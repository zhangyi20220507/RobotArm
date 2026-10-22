"""Review episodes, save individual trim ranges, and export a new LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import copy
import json
import mimetypes
import os
import re
import threading
import webbrowser
from datetime import datetime
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..paths import ARTIFACTS_ROOT
from .dataset_trim import export_dataset, export_plan, frame_bounds, trim_from_seconds


HTML = Path(__file__).with_name("dataset_browser.html").read_text(encoding="utf-8")
_UNSET = object()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class DatasetStore:
    def __init__(self, root: Path, reviews_path: Path) -> None:
        self.root = root.resolve()
        self.info = _read_json(self.root / "meta" / "info.json")
        self.reviews_path = reviews_path.resolve()
        self._lock = threading.Lock()
        self._export_lock = threading.Lock()
        self._export = {"status": "idle"}
        self.episodes = self._load_episode_rows()
        if len(self.episodes) != int(self.info["total_episodes"]):
            raise RuntimeError("episode metadata count does not match meta/info.json")
        if [int(row["episode_index"]) for row in self.episodes] != list(range(len(self.episodes))):
            raise ValueError("episode indices must be contiguous")
        self.reviews()  # Refuse a review file belonging to another dataset.

    def _load_episode_rows(self) -> list[dict]:
        import pyarrow.parquet as pq

        files = sorted((self.root / "meta" / "episodes").rglob("*.parquet"))
        if not files:
            raise FileNotFoundError("episode metadata parquet files not found")
        rows: list[dict] = []
        for path in files:
            rows.extend(pq.read_table(path).to_pylist())
        return sorted(rows, key=lambda row: int(row["episode_index"]))

    def reviews(self) -> dict:
        if not self.reviews_path.exists():
            return {}
        data = _read_json(self.reviews_path)
        if data.get("dataset_root") and Path(data["dataset_root"]).expanduser().resolve() != self.root:
            raise ValueError("quality-review file belongs to another dataset")
        reviews = data.get("episodes", {})
        if not isinstance(reviews, dict):
            raise ValueError("invalid review episodes")
        for index, review in reviews.items():
            if not str(index).isdigit() or not 0 <= int(index) < len(self.episodes) or not isinstance(review, dict):
                raise ValueError("invalid review episode")
            frame_bounds(review.get("trim"), int(self.episodes[int(index)]["length"]), float(self.info["fps"]))
        return reviews

    def save_review(self, episode: int, status: str, note: str, trim=_UNSET) -> dict:
        if not 0 <= episode < len(self.episodes):
            raise ValueError("invalid episode index")
        if status not in {"", "keep", "review", "reject"}:
            raise ValueError("invalid review status")
        with self._lock:
            reviews = self.reviews()
            review = dict(reviews.get(str(episode), {}))
            if trim is not _UNSET:
                if trim is None:
                    review.pop("trim", None)
                else:
                    if not isinstance(trim, dict) or set(trim) != {"start_s", "end_s"}:
                        raise ValueError("trim requires start_s and end_s")
                    saved = trim_from_seconds(trim["start_s"], trim["end_s"],
                                              int(self.episodes[episode]["length"]), float(self.info["fps"]))
                    if saved is None:
                        review.pop("trim", None)
                    else:
                        review["trim"] = saved
            review.update(status=status, note=note.strip())
            if status or note.strip() or review.get("trim"):
                reviews[str(episode)] = review
            else:
                reviews.pop(str(episode), None)
            payload = _read_json(self.reviews_path) if self.reviews_path.exists() else {}
            payload.update({
                "dataset_root": str(self.root),
                "episodes": reviews,
            })
            self.reviews_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.reviews_path.with_suffix(self.reviews_path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, self.reviews_path)
            return reviews

    def export_status(self) -> dict:
        with self._export_lock:
            return copy.deepcopy(self._export)

    def start_export(self, name: str) -> dict:
        # UI may only create a new sibling dataset, never arbitrary filesystem paths.
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", name):
            raise ValueError("新目录名只允许字母、数字、下划线、点和短横线，不能包含路径")
        output = self.root.parent / name
        if output.exists() or output.is_symlink() or output.resolve() == self.root:
            raise ValueError("输出目录已经存在，请换一个新名称")
        reviews = copy.deepcopy(self.reviews())
        plan = export_plan(self.info, self.episodes, reviews)
        with self._export_lock:
            if self._export["status"] == "running":
                raise ValueError("已有导出任务正在运行")
            self._export = {"status": "running", "output": str(output), "frames_done": 0,
                            "total_frames": sum(item["frames"] for item in plan),
                            "total_episodes": len(plan), "episodes_done": 0,
                            "current_episode": None, "episode_position": None,
                            "episodes": len(plan), "phase": "加载导出依赖"}
            initial = copy.deepcopy(self._export)

        def update(values):
            with self._export_lock:
                self._export.update(values)

        def run():
            try:
                result = export_dataset(self.root, output, self.info, plan, reviews, update)
                update({"status": "complete", **result, "phase": "导出完成",
                        "episodes_done": len(plan), "frames_done": result["frames"],
                        "current_episode": None, "episode_position": None})
            except Exception as exc:
                update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})

        threading.Thread(target=run, name="dataset-trim-export", daemon=True).start()
        return initial

    def meta_payload(self) -> dict:
        fps = float(self.info["fps"])
        return {
            "name": self.root.name,
            "fps": fps,
            "total_episodes": len(self.episodes),
            "episodes": [
                {
                    "index": int(row["episode_index"]),
                    "length": int(row["length"]),
                    "duration": int(row["length"]) / fps,
                    "tasks": row.get("tasks", []),
                }
                for row in self.episodes
            ],
            "reviews": self.reviews(),
            "reviews_path": str(self.reviews_path),
            "export_parent": str(self.root.parent),
            "suggested_export_name": self.root.name[:80] + "_trimmed_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        }

    @lru_cache(maxsize=8)
    def data_file(self, chunk: int, file: int):
        import pyarrow.parquet as pq

        template = self.info["data_path"]
        path = self.root / template.format(chunk_index=chunk, file_index=file)
        return pq.read_table(path, columns=["observation.state", "action", "timestamp", "episode_index"])

    def episode_payload(self, index: int) -> dict:
        import pyarrow.compute as pc
        import numpy as np

        if not 0 <= index < len(self.episodes):
            raise ValueError("invalid episode index")
        row = self.episodes[index]
        chunk, file = int(row["data/chunk_index"]), int(row["data/file_index"])
        table = self.data_file(chunk, file)
        table = table.filter(pc.equal(table["episode_index"], index))
        data = table.to_pylist()
        features = self.info["features"]
        names = features["observation.state"].get("names", {}).get("motors")
        names = names or [f"dim_{i}" for i in range(len(data[0]["observation.state"]))]
        video_keys = [key for key, feature in features.items() if feature.get("dtype") == "video"]
        videos = []
        for key in video_keys:
            prefix = f"videos/{key}"
            video_chunk = int(row[f"{prefix}/chunk_index"])
            video_file = int(row[f"{prefix}/file_index"])
            start = float(row[f"{prefix}/from_timestamp"])
            videos.append({"key": key, "url": f"/media/{index}/{video_keys.index(key)}", "from": start})
        timestamps = [float(item["timestamp"]) for item in data]
        state = np.asarray([item["observation.state"] for item in data], dtype=float)
        action = np.asarray([item["action"] for item in data], dtype=float)
        fps = float(self.info["fps"])
        action_step = np.diff(action, axis=0)
        tracking = action - state
        joint_step_deg = np.degrees(np.abs(action_step[:, :6])) if len(action_step) else np.zeros((0, 6))
        grip_step_mm = np.abs(action_step[:, 6]) * 1000 if len(action_step) else np.zeros(0)
        tracking_deg = np.degrees(np.abs(tracking[:, :6]))
        tracking_grip_mm = np.abs(tracking[:, 6]) * 1000
        timestamp_delta = np.diff(np.asarray(timestamps))
        max_joint_step = float(joint_step_deg.max(initial=0))
        max_grip_step = float(grip_step_mm.max(initial=0))
        p95_tracking = float(np.percentile(tracking_deg, 95)) if tracking_deg.size else 0.0
        p95_grip_tracking = float(np.percentile(tracking_grip_mm, 95)) if tracking_grip_mm.size else 0.0
        jitter_ms = float(np.max(np.abs(timestamp_delta - 1.0 / fps)) * 1000) if timestamp_delta.size else 0.0
        idle_ratio = float(np.mean((joint_step_deg.max(axis=1) < 0.03) & (grip_step_mm < 0.05))) if len(action_step) else 1.0
        warnings = []
        score = 100
        for condition, deduction, message in (
            (max_joint_step > 5.0, 20, "动作存在大于 5°/帧的关节跳变"),
            (max_grip_step > 10.0, 15, "夹爪动作存在大于 10 mm/帧的跳变"),
            (p95_tracking > 5.0, 20, "关节 state/action 跟随误差偏大"),
            (p95_grip_tracking > 10.0, 10, "夹爪 state/action 跟随误差偏大"),
            (jitter_ms > 5.0, 15, "时间戳间隔抖动超过 5 ms"),
            (idle_ratio > 0.45, 10, "静止帧比例较高，请检查无效等待"),
        ):
            if condition:
                score -= deduction
                warnings.append(message)
        metrics = {
            "score": max(0, score),
            "max_joint_step_deg": max_joint_step,
            "max_gripper_step_mm": max_grip_step,
            "p95_tracking_deg": p95_tracking,
            "p95_gripper_tracking_mm": p95_grip_tracking,
            "timestamp_jitter_ms": jitter_ms,
            "idle_ratio": idle_ratio,
            "warnings": warnings,
        }
        duration = int(row["length"]) / float(self.info["fps"])
        return {
            "index": index,
            "length": len(data),
            "duration": duration,
            "tasks": row.get("tasks", []),
            "names": names,
            "timestamps": timestamps,
            "state": state.tolist(),
            "action": action.tolist(),
            "metrics": metrics,
            "videos": videos,
        }

    def video_path(self, episode: int, ordinal: int) -> Path:
        row = self.episodes[episode]
        keys = [key for key, feature in self.info["features"].items() if feature.get("dtype") == "video"]
        key = keys[ordinal]
        chunk = int(row[f"videos/{key}/chunk_index"])
        file = int(row[f"videos/{key}/file_index"])
        path = self.root / self.info["video_path"].format(video_key=key, chunk_index=chunk, file_index=file)
        resolved = path.resolve()
        if not resolved.is_file() or self.root not in resolved.parents:
            raise FileNotFoundError(path)
        return resolved


def make_handler(store: DatasetStore):
    class Handler(BaseHTTPRequestHandler):
        server_version = "RobotArmDatasetBrowser/1.0"

        def log_message(self, format: str, *args) -> None:
            print(f"[dataset-browser] {self.address_string()} {format % args}")

        def json_response(self, payload: object, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            try:
                path = urlparse(self.path).path
                if path == "/":
                    body = HTML.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                elif path == "/api/meta":
                    self.json_response(store.meta_payload())
                elif path == "/api/export":
                    self.json_response(store.export_status())
                elif match := re.fullmatch(r"/api/episode/(\d+)", path):
                    self.json_response(store.episode_payload(int(match.group(1))))
                elif match := re.fullmatch(r"/media/(\d+)/(\d+)", path):
                    self.send_file(store.video_path(int(match.group(1)), int(match.group(2))))
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                # Chromium cancels an in-flight Range request when the user
                # seeks or switches episodes. The response socket is already
                # gone, so this is expected and must not trigger a second
                # attempt to send an HTTP 500 response.
                return
            except (ValueError, IndexError, FileNotFoundError) as exc:
                self.send_error(HTTPStatus.NOT_FOUND, str(exc))
            except Exception as exc:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path not in {"/api/review", "/api/export"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Content-Type must be application/json")
                origin = self.headers.get("Origin")
                if origin and urlparse(origin).netloc != self.headers.get("Host"):
                    raise ValueError("cross-origin writes are not allowed")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1_000_000:
                    raise ValueError("request too large")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("request must be an object")
                if path == "/api/export":
                    result = store.start_export(body["name"])
                else:
                    result = store.save_review(int(body["episode"]), str(body.get("status", "")),
                                               str(body.get("note", "")), body.get("trim", _UNSET))
                self.json_response(result)
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                self.json_response({"error": str(exc)}, 400)
            except OSError as exc:
                self.json_response({"error": str(exc)}, 500)

        def send_file(self, path: Path) -> None:
            size = path.stat().st_size
            start, end = 0, size - 1
            range_header = self.headers.get("Range")
            if range_header:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
                if not match:
                    self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    return
                if match.group(1):
                    start = int(match.group(1))
                if match.group(2):
                    end = min(int(match.group(2)), end)
                if start > end or start >= size:
                    self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    return
            length = end - start + 1
            self.send_response(HTTPStatus.PARTIAL_CONTENT if range_header else HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if range_header:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = length
                while remaining:
                    block = stream.read(min(1024 * 1024, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)

    return Handler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="robotarm modeling dataset-browser --", description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, default=None, help="quality-review JSON output")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-open", action="store_true", help="do not open the system browser")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.dataset_root.expanduser().resolve()
    reviews = args.reviews or ARTIFACTS_ROOT / "reports" / "dataset_quality" / f"{root.name}.json"
    store = DatasetStore(root, reviews)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    url = f"http://{args.host}:{args.port}"
    print(f"[dataset-browser] Dataset: {root}")
    print(f"[dataset-browser] Reviews: {store.reviews_path}")
    print(f"[dataset-browser] Open: {url}")
    print("[dataset-browser] Press Ctrl+C to stop.")
    if not args.no_open:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dataset-browser] stopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
