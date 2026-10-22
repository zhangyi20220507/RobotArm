"""Non-destructive, frame-aligned episode trimming for Dataset Browser.

Reviews store half-open frame ranges; a new LeRobot dataset materializes them.
No robot, camera, policy, or GPU is accessed by this module.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

BOOKKEEPING = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def trim_from_seconds(start: float, end: float, length: int, fps: float) -> dict | None:
    """Keep frame timestamps in [start, end); full episode is represented by None."""
    if isinstance(start, bool) or isinstance(end, bool):
        raise ValueError("起止时间必须是有限数字")
    start, end = float(start), float(end)
    if not (math.isfinite(start) and math.isfinite(end) and
            0 <= start < end <= length / fps + 1e-9):
        raise ValueError(f"保留范围须满足 0 ≤ 起点 < 终点 ≤ {length / fps:g} 秒")
    first = math.ceil(start * fps - 1e-8)
    stop = min(length, math.ceil(end * fps - 1e-8))
    if first >= stop:
        raise ValueError("所选时间段没有完整采样帧，请扩大范围")
    if first == 0 and stop == length:
        return None
    return {"start_frame": first, "end_frame": stop, "source_length": length, "fps": fps}


def frame_bounds(trim: dict | None, length: int, fps: float) -> tuple[int, int]:
    if trim is None:
        return 0, length
    if not isinstance(trim, dict):
        raise ValueError("invalid trim: expected frame range object")
    first, stop = trim.get("start_frame"), trim.get("end_frame")
    if type(first) is not int or type(stop) is not int or not 0 <= first < stop <= length:
        raise ValueError("invalid trim frame range")
    if trim.get("source_length") != length or trim.get("fps") != fps:
        raise ValueError("裁剪标记与当前 episode 长度/FPS 不符，请重新审核")
    return first, stop


def export_plan(info: dict, rows: list[dict], reviews: dict) -> list[dict]:
    plan = []
    for row in rows:
        index, length = int(row["episode_index"]), int(row["length"])
        review = reviews.get(str(index), {})
        if review.get("status", "") not in {"", "keep", "review", "reject"}:
            raise ValueError(f"invalid review status: episode {index}")
        if review.get("status") in {"review", "reject"}:
            continue
        first, stop = frame_bounds(review.get("trim"), length, float(info["fps"]))
        plan.append({"source_episode": index, "start_frame": first, "end_frame": stop,
                     "frames": stop - first, "source_length": length,
                     "output_episode": len(plan), "review": copy.deepcopy(review)})
    if not plan:
        raise ValueError("没有可导出的轨迹：复查和剔除条目不参与导出")
    return plan


def export_dataset(
    source: Path, output: Path, info: dict, plan: list[dict], reviews: dict,
    progress: Callable[[dict], None] = lambda _: None,
) -> dict:
    """Rebuild all selected features/stats. Output must be a new sibling directory."""
    source, output = source.resolve(), output.resolve()
    if output.parent != source.parent or output == source or output.exists():
        raise ValueError("导出必须是源数据集旁边尚不存在的新目录")
    if not plan:
        raise ValueError("empty export plan")
    if not math.isfinite(float(info["fps"])) or info["fps"] <= 0 or int(info["fps"]) != info["fps"]:
        raise ValueError("export requires a positive integer FPS")
    # Lazy imports: browsing and saving marks do not require importing LeRobot.
    import numpy as np
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {k: copy.deepcopy(v) for k, v in info["features"].items() if k not in BOOKKEEPING}
    for key, feature in features.items():
        if feature["dtype"] in {"video", "image"}:
            if len(feature["shape"]) != 3 or feature["shape"][-1] != 3:
                raise ValueError(f"目前裁剪导出支持 RGB HWC 相机，无法导出 {key}")
            feature.pop("info", None)  # Encoding metadata is regenerated.
    repo_id = f"local/{output.name}"
    writer = LeRobotDataset.create(
        repo_id=repo_id, root=output, fps=int(info["fps"]),
        robot_type=info.get("robot_type"), features=features, use_videos=True,
        image_writer_threads=2, encoder_threads=2,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=18, preset="fast"),
    )
    marker = output / "TRIM_INCOMPLETE.txt"
    marker.write_text("Export in progress or interrupted. Do not train on this directory.\n")
    total = sum(item["frames"] for item in plan)
    done = 0
    try:
        for position, item in enumerate(plan, start=1):
            episode = item["source_episode"]
            progress({"total_episodes": len(plan), "episodes_done": position - 1,
                      "current_episode": episode, "episode_position": position,
                      "frames_done": done, "total_frames": total, "phase": "复制帧"})
            reader = LeRobotDataset(
                repo_id=info.get("repo_id") or f"local/{source.name}", root=source,
                episodes=[episode], download_videos=False, return_uint8=True,
            )
            if len(reader) != item["source_length"]:
                raise ValueError(f"source episode {episode} length changed")
            for index in range(item["start_frame"], item["end_frame"]):
                sample = reader[index]
                frame = {}
                for key, feature in features.items():
                    value = sample[key]
                    if hasattr(value, "numpy"):
                        value = value.numpy()
                    if feature["dtype"] in {"image", "video"}:
                        value = np.moveaxis(value, 0, -1)  # LeRobot reader returns CHW.
                    frame[key] = value
                frame["task"] = sample["task"]
                writer.add_frame(frame)
                done += 1
                if done % 20 == 0 or done == total:
                    progress({"frames_done": done, "total_frames": total,
                              "episode": episode, "phase": "复制帧"})
            progress({"phase": "编码视频", "episode": episode, "frames_done": done})
            writer.save_episode(parallel_encoding=False)
            # Count a completed episode only after its frames and videos are saved.
            progress({"episodes_done": position, "current_episode": None,
                      "episode_position": None})
        progress({"phase": "生成统计与来源清单", "current_episode": None,
                  "episode_position": None})
        writer.finalize()
        manifest = {
            "format": "robotarm_dataset_trim_v1", "source_dataset": str(source),
            "output_dataset": str(output), "repo_id": repo_id, "fps": info["fps"],
            "range_semantics": "[start_frame, end_frame)",
            "source_info_sha256": hashlib.sha256((source / "meta/info.json").read_bytes()).hexdigest(),
            "reviews_snapshot": reviews, "episodes": plan, "total_frames": total,
            "videos_reencoded": True,
        }
        (output / "meta/trim_provenance.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # Preserve action semantics for Pi0/ACT without inheriting stale raw frame mappings.
        conversion = source / "meta/piper_conversion.json"
        if conversion.is_file():
            original = json.loads(conversion.read_text())
            derived = {k: original[k] for k in
                       ("representation", "compatible_policies", "camera_roles") if k in original}
            derived.update({"repo_id": repo_id, "converter": "robotarm.dataset_trim",
                            "source_conversion": original, "episodes": plan})
            (output / "meta/piper_conversion.json").write_text(
                json.dumps(derived, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        simulation = source / "simulation_provenance.json"
        if simulation.is_file():
            original = json.loads(simulation.read_text())
            (output / "simulation_provenance.json").write_text(json.dumps({
                **{k: v for k, v in original.items() if k != "episodes"},
                "source_provenance": original, "episodes": plan,
                "trim_provenance": "meta/trim_provenance.json",
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result_info = json.loads((output / "meta/info.json").read_text())
        if result_info["total_frames"] != total or result_info["total_episodes"] != len(plan):
            raise RuntimeError("exported frame/episode count mismatch")
        marker.unlink()
        return {"output": str(output), "episodes": len(plan), "frames": total}
    except BaseException as exc:
        marker.write_text(f"Export incomplete: {type(exc).__name__}: {exc}\nDo not train on this directory.\n")
        # Close writers without treating a partially exported dataset as valid.
        try:
            writer.finalize()
        except Exception:
            pass
        raise
