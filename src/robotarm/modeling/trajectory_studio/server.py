from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import re
import subprocess
import traceback
import urllib.parse
import xml.etree.ElementTree as ET
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ...paths import CHECKPOINT_ROOT, LEROBOT_DATA_ROOT, PIPER_MANISKILL_URDF, TRAINING_LOG_ROOT, WORKSPACE_ROOT
from .adapters import ADAPTERS
from .live_can import LIVE_CAN


PROJECT = Path(__file__).resolve().parent
WORKSPACE = WORKSPACE_ROOT.resolve()
FRONTEND = PROJECT / "frontend"
RUNS = TRAINING_LOG_ROOT / "trajectory_studio"
DEFAULT_URDF = PIPER_MANISKILL_URDF
PACKAGE_ROOTS = {}


def safe_path(value: str, *, must_exist: bool = True) -> Path:
    path = Path(value).expanduser().resolve()
    if path != WORKSPACE and WORKSPACE not in path.parents:
        raise ValueError(f"path must stay under {WORKSPACE}")
    if must_exist and not path.exists():
        raise ValueError(f"path does not exist: {path}")
    return path


def path_below(root: Path, value: str, *, must_exist: bool = True) -> Path:
    path = Path(value).resolve()
    resolved_root = root.resolve()
    if path != resolved_root and resolved_root not in path.parents:
        raise ValueError(f"path must stay under {resolved_root}")
    if must_exist and not path.exists():
        raise ValueError(f"path does not exist: {path}")
    return path


def vector(value: str | None, default: tuple[float, float, float]) -> list[float]:
    return [float(item) for item in value.split()] if value else list(default)


def resolve_mesh(uri: str, urdf: Path) -> Path:
    if uri.startswith("package://"):
        package, relative = uri[len("package://"):].split("/", 1)
        root = PACKAGE_ROOTS.get(package)
        if root is None:
            raise ValueError(f"unknown URDF package: {package}")
        # This Piper URDF uses package://agx_arm_description/agx_arm_urdf/...
        # while its checkout already starts at the agx_arm_urdf directory.
        if package == "agx_arm_description" and relative.startswith("agx_arm_urdf/"):
            relative = relative[len("agx_arm_urdf/"):]
        return safe_path(str(root / relative))
    return safe_path(str(urdf.parent / uri))


def expand_robot(path: Path) -> ET.Element:
    """Expand the simple xacro includes used by the bundled Piper model."""
    root = ET.parse(path).getroot()
    for include in list(root):
        if not include.tag.endswith("}include"):
            continue
        filename = include.get("filename", "")
        match = re.fullmatch(r"\$\(find ([^)]+)\)/(.*)", filename)
        if not match:
            raise ValueError(f"unsupported xacro include: {filename}")
        package, relative = match.groups()
        package_root = PACKAGE_ROOTS.get(package)
        if package_root is None:
            raise ValueError(f"unknown xacro package: {package}")
        if package == "agx_arm_description" and relative.startswith("agx_arm_urdf/"):
            relative = relative[len("agx_arm_urdf/"):]
        included = expand_robot(safe_path(str(package_root / relative)))
        position = list(root).index(include)
        root.remove(include)
        for child in list(included):
            root.insert(position, child)
            position += 1
    return root


def parse_geometry(link: ET.Element, urdf: Path) -> dict[str, Any] | None:
    # Browser renderer uses STL. Prefer collision STL when visual is DAE.
    candidates = [link.find("visual"), link.find("collision")]
    candidates.sort(key=lambda item: 0 if item is not None and (
        (item.find("geometry/mesh") is not None) and
        item.find("geometry/mesh").get("filename", "").lower().endswith(".stl")
    ) else 1)
    for node in candidates:
        if node is None:
            continue
        mesh = node.find("geometry/mesh")
        if mesh is None or not mesh.get("filename"):
            continue
        path = resolve_mesh(mesh.get("filename", ""), urdf)
        if path.suffix.lower() != ".stl":
            # The bundled ManiSkill URDF references visual DAE files, while
            # the browser renderer consumes their adjacent STL equivalents.
            stl = path.parent.parent / f"{path.stem}.stl"
            if not stl.is_file():
                continue
            path = stl
        origin = node.find("origin")
        return {
            "mesh": "/api/file?path=" + urllib.parse.quote(str(path)),
            "xyz": vector(origin.get("xyz") if origin is not None else None, (0, 0, 0)),
            "rpy": vector(origin.get("rpy") if origin is not None else None, (0, 0, 0)),
            "scale": vector(mesh.get("scale"), (1, 1, 1)),
        }
    return None


def parse_urdf(path: Path) -> dict[str, Any]:
    root = expand_robot(path)
    links = {}
    for link in root.findall("link"):
        links[link.get("name", "")] = {"name": link.get("name"), "geometry": parse_geometry(link, path)}
    joints = []
    child_links = set()
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        mimic = joint.find("mimic")
        child = joint.find("child").get("link")
        child_links.add(child)
        joints.append({
            "name": joint.get("name"), "type": joint.get("type"),
            "parent": joint.find("parent").get("link"), "child": child,
            "xyz": vector(origin.get("xyz") if origin is not None else None, (0, 0, 0)),
            "rpy": vector(origin.get("rpy") if origin is not None else None, (0, 0, 0)),
            "axis": vector(axis.get("xyz") if axis is not None else None, (1, 0, 0)),
            "lower": float(limit.get("lower", "0")) if limit is not None else None,
            "upper": float(limit.get("upper", "0")) if limit is not None else None,
            "mimic": ({
                "joint": mimic.get("joint"),
                "multiplier": float(mimic.get("multiplier", "1")),
                "offset": float(mimic.get("offset", "0")),
            } if mimic is not None else None),
        })
    roots = sorted(set(links) - child_links)
    return {"name": root.get("name"), "path": str(path), "roots": roots, "links": links, "joints": joints}


def catalog() -> dict[str, Any]:
    datasets = []
    for info in sorted(LEROBOT_DATA_ROOT.glob("*/meta/info.json")):
        value = json.loads(info.read_text(encoding="utf-8"))
        datasets.append({"path": str(info.parents[1]), "episodes": value.get("total_episodes"), "frames": value.get("total_frames")})
    # ``last`` usually resolves to the final numbered checkpoint. De-duplicate
    # resolved paths so the selector does not show the same model twice.
    checkpoints = sorted({
        str(path.resolve())
        for path in CHECKPOINT_ROOT.glob("*/checkpoints/*/pretrained_model")
    })
    return {
        "studio_version": 3,
        "workspace": str(WORKSPACE), "default_urdf": str(DEFAULT_URDF),
        "datasets": datasets, "checkpoints": checkpoints,
        "adapters": [adapter.describe() for adapter in ADAPTERS.values()],
    }


def ensure_camera_video(output: Path) -> bool:
    """Recover the camera-only stream from a legacy composite cache."""
    camera = output / "camera.mp4"
    if camera.is_file():
        return True
    comparison = output / "comparison.mp4"
    if not comparison.is_file():
        return False
    completed = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(comparison),
         "-vf", "crop=iw-720:ih:0:0", "-c:v", "libx264", "-preset", "fast",
         "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(camera)],
        check=False, capture_output=True, text=True,
    )
    return completed.returncode == 0 and camera.is_file()


class Handler(BaseHTTPRequestHandler):
    server_version = "RobotArmOpenLoopStudio/1.0"

    def send_json(self, value: Any, status: int = 200) -> None:
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def send_file(self, path: Path) -> None:
        size = path.stat().st_size
        start, end, status = 0, size - 1, HTTPStatus.OK
        requested = self.headers.get("Range")
        if requested:
            if not requested.startswith("bytes=") or "," in requested:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            first, last = requested[6:].split("-", 1)
            if first:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
            else:
                length = min(int(last), size)
                start, end = size - length, size - 1
            if start < 0 or start >= size or end < start:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = HTTPStatus.PARTIAL_CONTENT
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_GET(self) -> None:
        try:
            parsed = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed.query)
            if parsed.path == "/api/catalog":
                return self.send_json(catalog())
            if parsed.path == "/api/live/state":
                return self.send_json(LIVE_CAN.snapshot())
            if parsed.path == "/api/urdf":
                path = safe_path(query.get("path", [str(DEFAULT_URDF)])[0])
                return self.send_json(parse_urdf(path))
            if parsed.path == "/api/adapter":
                name = query.get("name", ["act"])[0]
                checkpoint = query.get("checkpoint", [None])[0]
                return self.send_json(ADAPTERS[name].describe(safe_path(checkpoint) if checkpoint else None))
            if parsed.path == "/api/file":
                return self.send_file(safe_path(query["path"][0]))
            if parsed.path.startswith("/api/runs/"):
                relative = parsed.path[len("/api/runs/"):]
                return self.send_file(path_below(RUNS, str(RUNS / relative)))
            relative = "index.html" if parsed.path == "/" else parsed.path.lstrip("/")
            static = (FRONTEND / relative).resolve()
            if FRONTEND.resolve() not in static.parents:
                raise ValueError("invalid static path")
            return self.send_file(static)
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            if self.path == "/api/live/start":
                return self.send_json(LIVE_CAN.start(request))
            if self.path == "/api/live/stop":
                return self.send_json(LIVE_CAN.stop())
            if self.path == "/api/source":
                video = safe_path(request["video_path"])
                trajectory = safe_path(request["trajectory_path"])
                return self.send_json({
                    "video_url": "/api/file?path=" + urllib.parse.quote(str(video)),
                    "trajectory_url": "/api/file?path=" + urllib.parse.quote(str(trajectory)),
                    "urdf": str(safe_path(request.get("urdf", str(DEFAULT_URDF)))),
                })
            if self.path != "/api/evaluate":
                return self.send_json({"error": "not found"}, 404)
            adapter = ADAPTERS[str(request.get("adapter", "act"))]
            # Resolve all external inputs before hashing or executing.
            request["dataset_root"] = str(safe_path(request["dataset_root"]))
            request["checkpoint"] = str(safe_path(request["checkpoint"]))
            urdf = str(safe_path(request.get("urdf", str(DEFAULT_URDF))))
            key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:16]
            output = RUNS / key
            output.mkdir(parents=True, exist_ok=True)
            summary_path = output / "summary.json"
            cached = summary_path.is_file() and (output / "trajectory.csv").is_file()
            if cached and ensure_camera_video(output):
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            else:
                summary = adapter.evaluate(request, output)
            self.send_json({
                "run_id": key, "summary": summary, "urdf": urdf,
                "trajectory_url": f"/api/runs/{key}/trajectory.csv",
                "video_url": f"/api/runs/{key}/camera.mp4",
            })
        except Exception as exc:
            traceback.print_exc()
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch the local open-loop policy validation studio."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    RUNS.mkdir(parents=True, exist_ok=True)
    print(f"RobotArm Open-loop Studio: http://{args.host}:{args.port}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    finally:
        LIVE_CAN.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    main()
