#!/usr/bin/env python3
"""Check the unified RobotArm environment and optional LeRobot dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=None)
    args = parser.parse_args()

    import lerobot
    import torch
    import torchcodec

    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"LeRobot: {lerobot.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"TorchCodec: {torchcodec.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the current environment")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    probe = torch.randn(256, 256, device="cuda")
    result = probe @ probe
    torch.cuda.synchronize()
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("GPU smoke test produced non-finite values")
    print("GPU smoke test: OK")

    if args.dataset_root is not None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = args.dataset_root.expanduser().resolve()
        info_path = root / "meta" / "info.json"
        if not info_path.is_file():
            raise RuntimeError(f"not a LeRobot dataset: {root}")
        import json

        info = json.loads(info_path.read_text(encoding="utf-8"))
        provenance_path = root / "meta" / "piper_conversion.json"
        provenance = (
            json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance_path.is_file()
            else {}
        )
        repo_id = provenance.get("repo_id") or info.get("repo_id")
        repo_id = repo_id or f"local/{root.name}"
        dataset = LeRobotDataset(
            repo_id=repo_id, root=root, download_videos=False
        )
        print(f"Dataset: {root}")
        print(f"Dataset repo_id: {repo_id}")
        print(f"Frames: {len(dataset)}")
        print(f"Episodes: {dataset.meta.total_episodes}")
        print(f"FPS: {dataset.fps}")
        print(f"Cameras: {', '.join(dataset.meta.camera_keys)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Environment check failed: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
