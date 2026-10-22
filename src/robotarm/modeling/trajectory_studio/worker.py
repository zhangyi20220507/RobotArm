#!/usr/bin/env python3
"""Isolated inference worker used by the open-loop validation studio.

The studio is the user-facing entry point. This worker keeps model inference in
an isolated process and writes only the artifacts consumed by the studio:
``summary.json``, ``trajectory.csv``, and ``camera.mp4``.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


NAMES = ("J1", "J2", "J3", "J4", "J5", "J6", "gripper")


def batch_sample(sample: dict, camera_keys: list[str]) -> dict:
    import torch

    batch = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            value = value.unsqueeze(0)
            if key in camera_keys and value.dtype == torch.uint8:
                value = value.float().div_(255.0)
            batch[key] = value
        elif isinstance(value, str):
            batch[key] = [value]
        else:
            batch[key] = value
    return batch


def tensor_image_to_bgr(value):
    import cv2
    import numpy as np

    image = value.detach().cpu().numpy()
    if image.shape[0] in (1, 3, 4):
        image = image.transpose(1, 2, 0)
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def display_units(values):
    import numpy as np

    values = np.asarray(values, dtype=np.float64).copy()
    values[..., :6] = np.degrees(values[..., :6])
    values[..., 6] *= 1000.0
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="0 uses the complete episode")
    parser.add_argument("--chunked", action="store_true",
                        help="consume cached action chunks; default replans every frame")
    parser.add_argument("--output-step", type=int, default=0,
                        help="action-chunk index k; compare prediction from t with record at t+k")
    args = parser.parse_args(argv)

    import cv2
    import numpy as np
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    checkpoint = args.checkpoint.expanduser().resolve()
    root = args.dataset_root.expanduser().resolve()
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    provenance_path = root / "meta" / "piper_conversion.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}
    repo_id = provenance.get("repo_id") or info.get("repo_id") or f"local/{root.name}"
    if not 0 <= args.episode < int(info["total_episodes"]):
        raise RuntimeError(f"episode must be in 0..{int(info['total_episodes']) - 1}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    dataset = LeRobotDataset(repo_id=repo_id, root=root, episodes=[args.episode])
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.pretrained_path = str(checkpoint)
    config.device = args.device
    policy = make_policy(config, ds_meta=dataset.meta)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    action_chunk_size = int(
        getattr(config, "chunk_size", getattr(config, "n_action_steps", 1))
    )
    if args.output_step < 0 or args.output_step >= action_chunk_size:
        raise RuntimeError(f"--output-step must be in 0..{action_chunk_size - 1}")
    if args.chunked and args.output_step:
        raise RuntimeError("--chunked and nonzero --output-step cannot be combined")
    available = len(dataset) - args.output_step
    total = available if args.max_frames <= 0 else min(available, args.max_frames)
    states, recorded, predicted, timestamps = [], [], [], []
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()
    with torch.inference_mode():
        for index in range(total):
            sample = dataset[index]
            processed = preprocessor(batch_sample(sample, dataset.meta.camera_keys))
            if args.chunked:
                action = postprocessor(policy.select_action(processed))
            elif args.output_step == 0:
                if callable(reset):
                    reset()
                action = postprocessor(policy.select_action(processed))
            else:
                if int(getattr(config, "n_obs_steps", 1)) > 1:
                    raise RuntimeError(
                        "nonzero --output-step is not supported for policies "
                        "that require multiple observation steps"
                    )
                chunk = policy.predict_action_chunk(processed)
                action = postprocessor(chunk[:, args.output_step])
            aligned = dataset[index + args.output_step]
            states.append(aligned["observation.state"].detach().cpu().numpy())
            recorded.append(aligned["action"].detach().cpu().numpy())
            predicted.append(action.detach().cpu().squeeze(0).numpy())
            timestamps.append(float(aligned["timestamp"].item()))
            if (index + 1) % 50 == 0 or index + 1 == total:
                print(f"Inference {index + 1}/{total}", flush=True)

    states = np.asarray(states)
    recorded = np.asarray(recorded)
    predicted = np.asarray(predicted)
    errors = np.abs(predicted - recorded)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(checkpoint),
        "dataset": str(root),
        "episode": args.episode,
        "frames": total,
        "inference_mode": "chunked" if args.chunked else "replan_each_frame",
        "output_step": args.output_step,
        "chunk_size": action_chunk_size,
        "mae": float(errors.mean()),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "max_abs_error": float(errors.max()),
        "per_dimension_mae": errors.mean(axis=0).tolist(),
        "per_dimension_mae_degrees_or_mm": display_units(errors.mean(axis=0)).tolist(),
        "per_dimension_rmse_degrees_or_mm": display_units(
            np.sqrt(np.mean(errors ** 2, axis=0))
        ).tolist(),
        "per_dimension_max_error_degrees_or_mm": display_units(
            errors.max(axis=0)
        ).tolist(),
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    with (output / "trajectory.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp"] + [f"state_{name}" for name in NAMES]
                        + [f"recorded_{name}" for name in NAMES]
                        + [f"predicted_{name}" for name in NAMES])
        for index in range(total):
            writer.writerow([
                timestamps[index], *states[index], *recorded[index], *predicted[index]
            ])

    first_image = tensor_image_to_bgr(dataset[args.output_step]["observation.images.global"])
    camera_video_path = output / "camera.mp4"
    camera_intermediate = output / "camera.intermediate.mp4"
    camera_writer = cv2.VideoWriter(
        str(camera_intermediate), cv2.VideoWriter_fourcc(*"mp4v"),
        float(info["fps"]), (first_image.shape[1], first_image.shape[0]),
    )
    if not camera_writer.isOpened():
        raise RuntimeError(f"cannot create video {camera_video_path}")
    try:
        for index in range(total):
            image = tensor_image_to_bgr(dataset[index + args.output_step]["observation.images.global"])
            camera_writer.write(image)
            if (index + 1) % 100 == 0 or index + 1 == total:
                print(f"Video {index + 1}/{total}", flush=True)
    finally:
        camera_writer.release()
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(camera_intermediate),
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(camera_video_path),
    ], check=True)
    camera_intermediate.unlink()
    print(json.dumps(report, indent=2))
    print(f"Camera video: {camera_video_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Open-loop evaluation failed: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
