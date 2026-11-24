"""Validate the shared cloud environment and an uploaded Piper LeRobot dataset."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path


def check_environment():
    import torch

    versions = {
        name: importlib.metadata.version(name)
        for name in ("lerobot", "torch", "torchvision", "transformers", "peft", "accelerate")
    }
    if versions["lerobot"] != "0.6.1":
        raise RuntimeError(f"Expected lerobot 0.6.1, got {versions['lerobot']}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support BF16")
    result = {
        "versions": versions,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
        "bf16": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def check_dataset(root: Path):
    import pyarrow.parquet as parquet

    info_path = root / "meta/info.json"
    if not info_path.is_file():
        raise RuntimeError(f"Not a LeRobot dataset: missing {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    for key in ("observation.state", "action"):
        if features.get(key, {}).get("shape") != [7]:
            raise RuntimeError(f"Expected {key} shape [7]")
    images = sorted(key for key in features if key.startswith("observation.images."))
    expected_images = ["observation.images.global", "observation.images.wrist"]
    if images != expected_images:
        raise RuntimeError(f"Expected dual cameras {expected_images}, got {images}")
    unexpected = sorted(
        key
        for key in features
        if key.startswith("observation.") and key not in {"observation.state", *expected_images}
    )
    if unexpected:
        raise RuntimeError(f"Unexpected policy observations: {unexpected}")
    if int(info.get("total_episodes", 0)) < 2 or int(info.get("total_frames", 0)) < 2:
        raise RuntimeError("Dataset is empty or too small to split")
    tasks_path = root / "meta/tasks.parquet"
    if not tasks_path.is_file():
        raise RuntimeError("VLA training requires meta/tasks.parquet with language instructions")
    task_rows = parquet.read_table(tasks_path, columns=["task"]).to_pylist()
    tasks = sorted({str(row.get("task", "")).strip() for row in task_rows if str(row.get("task", "")).strip()})
    if not tasks:
        raise RuntimeError("VLA training requires at least one non-empty language task")
    provenance_path = root / "meta/piper_conversion.json"
    if not provenance_path.is_file():
        raise RuntimeError("Missing meta/piper_conversion.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    representation = provenance.get("representation", {})
    if representation.get("space") != "joint" or representation.get("frame") != "absolute":
        raise RuntimeError("Expected absolute joint actions")
    result = {
        "dataset": str(root),
        "episodes": info["total_episodes"],
        "frames": info["total_frames"],
        "fps": info["fps"],
        "images": images,
        "state_action_dim": 7,
        "representation": representation,
        "tasks": tasks,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--environment-only", action="store_true")
    args = parser.parse_args(argv)
    check_environment()
    if not args.environment_only:
        if args.dataset_root is None:
            parser.error("--dataset-root is required unless --environment-only is used")
        check_dataset(args.dataset_root.expanduser().resolve())


if __name__ == "__main__":
    main()
