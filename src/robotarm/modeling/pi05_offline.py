"""Benchmark Pi0 or Pi0.5 inference from a local LeRobot dataset without hardware I/O."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..paths import ARTIFACTS_ROOT, LEROBOT_DATA_ROOT, PRETRAINED_ROOT, TRAINING_LOG_ROOT


DEFAULT_CHECKPOINT = PRETRAINED_ROOT / "pi05_base"
DEFAULT_DATASET = LEROBOT_DATA_ROOT / "piper_pick_cup_global_only"


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--policy-type", choices=("auto", "pi0", "pi05"), default="auto",
        help="expected policy type; auto reads checkpoint/config.json",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--index", type=nonnegative_int, default=0,
                        help="global frame index in the local dataset")
    parser.add_argument("--task", default=None,
                        help="override the task stored in the dataset sample")
    parser.add_argument(
        "--wrist-slot", choices=("left", "right"), default="right",
        help="map observation.images.wrist to this Pi wrist-camera slot",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--quantization", choices=("none", "int8-weight-only"), default="none",
        help="optional TorchAO inference-only quantization",
    )
    parser.add_argument("--warmup-runs", type=nonnegative_int, default=1)
    parser.add_argument("--runs", type=positive_int, default=5)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--datasets-cache", type=Path,
        default=ARTIFACTS_ROOT / "cache" / "huggingface" / "datasets",
        help="writable Hugging Face dataset metadata cache",
    )
    return parser


def make_dataset_batch(
    sample: dict[str, Any],
    task_override: str | None = None,
    wrist_slot: str = "right",
):
    """Map one 7D Piper/global-camera sample to the Pi0.5 base input schema."""
    import torch

    state = sample.get("observation.state")
    image = sample.get("observation.images.global")
    task = task_override if task_override is not None else sample.get("task")
    if not isinstance(state, torch.Tensor) or tuple(state.shape) != (7,):
        raise RuntimeError("offline Pi0.5 expects a 7D observation.state")
    if not torch.isfinite(state).all():
        raise RuntimeError("offline Pi0.5 state contains non-finite values")
    if not isinstance(image, torch.Tensor) or image.ndim != 3 or image.shape[0] != 3:
        raise RuntimeError("offline Pi0.5 expects a CHW global RGB image")
    if not torch.isfinite(image).all():
        raise RuntimeError("offline Pi0.5 image contains non-finite values")
    if not isinstance(task, str) or not task.strip():
        raise RuntimeError("offline Pi0.5 requires a non-empty task")
    if wrist_slot not in {"left", "right"}:
        raise ValueError("wrist_slot must be left or right")

    padded = torch.zeros((1, 32), dtype=torch.float32)
    padded[0, :7] = state.to(dtype=torch.float32)
    batch = {
        "observation.state": padded,
        "observation.images.base_0_rgb": image.to(dtype=torch.float32).unsqueeze(0),
        "task": [task.strip()],
    }
    wrist = sample.get("observation.images.wrist")
    if wrist is not None:
        if not isinstance(wrist, torch.Tensor) or wrist.ndim != 3 or wrist.shape[0] != 3:
            raise RuntimeError("offline Pi0.5 expects a CHW wrist RGB image")
        if not torch.isfinite(wrist).all():
            raise RuntimeError("offline Pi0.5 wrist image contains non-finite values")
        batch[f"observation.images.{wrist_slot}_wrist_0_rgb"] = (
            wrist.to(dtype=torch.float32).unsqueeze(0)
        )
    return batch


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def validate_base_schema(config: Any, policy_type: str) -> None:
    expected = {
        "observation.state",
        "observation.images.base_0_rgb",
        "observation.images.left_wrist_0_rgb",
        "observation.images.right_wrist_0_rgb",
    }
    state_shape = tuple(config.input_features["observation.state"].shape)
    action_shape = tuple(config.output_features["action"].shape)
    if set(config.input_features) != expected or state_shape != (32,) or action_shape != (32,):
        raise RuntimeError(
            f"offline {policy_type} supports only the known three-camera 32D base schema"
        )


def verify_weights_loaded(policy: Any, checkpoint: Path, policy_type: str) -> None:
    """Catch LeRobot silently returning a randomly initialized Pi policy."""
    import torch
    from safetensors import safe_open

    probes = {
        "pi0": ("action_in_proj.bias", "action_out_proj.bias", "state_proj.bias"),
        "pi05": ("action_in_proj.bias", "action_out_proj.bias", "time_mlp_in.bias"),
    }[policy_type]
    state = policy.state_dict()
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as weights:
        for key in probes:
            model_key = f"model.{key}"
            if key not in weights.keys() or model_key not in state:
                raise RuntimeError(f"{policy_type} checkpoint weight probe missing: {key}")
            expected = weights.get_tensor(key).to(dtype=state[model_key].dtype)
            actual = state[model_key].detach().cpu()
            if expected.shape != actual.shape or not torch.equal(expected, actual):
                raise RuntimeError(
                    f"{policy_type} weights were not loaded correctly ({key}); refusing inference"
                )


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.checkpoint.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not (checkpoint / "config.json").is_file() or not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"Pi checkpoint is incomplete: {checkpoint}")
    if not (dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"LeRobot dataset not found: {dataset_root}")

    cache = args.datasets_cache.expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache))

    import numpy as np
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Pi CUDA benchmark requires BF16 support")

    dataset_started = time.perf_counter()
    dataset = LeRobotDataset(f"local/{dataset_root.name}", root=dataset_root)
    if args.index >= len(dataset):
        raise IndexError(f"--index {args.index} is outside dataset length {len(dataset)}")
    sample_started = time.perf_counter()
    sample = dict(dataset[args.index])
    sample_decode_ms = (time.perf_counter() - sample_started) * 1000.0
    dataset_open_ms = (sample_started - dataset_started) * 1000.0
    batch = make_dataset_batch(sample, args.task, args.wrist_slot)

    requested_dtype = "bfloat16" if args.device == "cuda" else "float32"
    checkpoint_metadata = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    policy_type = checkpoint_metadata.get("type")
    if policy_type not in {"pi0", "pi05"}:
        raise RuntimeError(f"unsupported checkpoint policy type: {policy_type!r}")
    if args.policy_type != "auto" and args.policy_type != policy_type:
        raise RuntimeError(
            f"--policy-type {args.policy_type} does not match checkpoint type {policy_type}"
        )
    if policy_type == "pi0":
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy as PolicyClass
    else:
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy as PolicyClass

    config = PreTrainedConfig.from_pretrained(
        checkpoint, device=args.device, dtype=requested_dtype
    )
    validate_base_schema(config, policy_type)
    config.device = args.device
    config.dtype = requested_dtype

    print(
        f"OFFLINE ONLY: loading {policy_type} with one dataset frame; "
        "no CAN or camera connection is used.",
        flush=True,
    )
    model_started = time.perf_counter()
    policy = PolicyClass.from_pretrained(
        checkpoint, config=config, strict=True, local_files_only=True
    )
    verify_weights_loaded(policy, checkpoint, policy_type)
    policy.eval()
    model_load_s = time.perf_counter() - model_started

    quantization_s = 0.0
    quantized_linear_count = 0
    if args.quantization == "int8-weight-only":
        if args.device != "cuda":
            raise RuntimeError("INT8 weight-only benchmark currently requires CUDA")
        from torchao.quantization import Int8WeightOnlyConfig, quantize_

        quantization_started = time.perf_counter()
        quantize_(policy, Int8WeightOnlyConfig())
        torch.cuda.synchronize()
        quantization_s = time.perf_counter() - quantization_started
        quantized_linear_count = sum(
            type(module.weight).__name__ == "Int8Tensor"
            for module in policy.modules()
            if isinstance(module, torch.nn.Linear)
        )
        if quantized_linear_count == 0:
            raise RuntimeError("TorchAO did not quantize any Pi linear layers")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    def synchronize() -> None:
        if args.device == "cuda":
            torch.cuda.synchronize()

    def infer_once() -> tuple[float, Any]:
        reset = getattr(policy, "reset", None)
        if callable(reset):
            reset()  # Force a fresh 50-step chunk instead of consuming its queue.
        synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            action = postprocessor(policy.select_action(preprocessor(batch)))
        synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        output = action.detach().to(dtype=torch.float32, device="cpu").squeeze(0)
        if tuple(output.shape) != (32,) or not torch.isfinite(output).all():
            raise RuntimeError("Pi0.5 returned an invalid 32D base action")
        return elapsed_ms, output

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        resident_gib = torch.cuda.memory_allocated() / 1024**3
    else:
        resident_gib = None

    warmup_ms = []
    action = None
    for number in range(args.warmup_runs):
        latency, action = infer_once()
        warmup_ms.append(latency)
        print(f"warmup {number + 1}/{args.warmup_runs}: {latency:.1f} ms", flush=True)

    latencies_ms = []
    for number in range(args.runs):
        latency, action = infer_once()
        latencies_ms.append(latency)
        print(f"run {number + 1}/{args.runs}: {latency:.1f} ms", flush=True)

    assert action is not None
    report = {
        "mode": "offline_dataset_inference",
        "command_sent": False,
        "policy_type": policy_type,
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "dataset_length": len(dataset),
        "dataset_fps": dataset.fps,
        "sample_index": args.index,
        "episode_index": int(sample["episode_index"]),
        "frame_index": int(sample["frame_index"]),
        "task": batch["task"][0],
        "camera_keys": sorted(key for key in batch if key.startswith("observation.images.")),
        "wrist_slot": args.wrist_slot if "observation.images.wrist" in sample else None,
        "device": args.device,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "dtype": config.dtype,
        "quantization": args.quantization,
        "quantization_s": quantization_s,
        "quantized_linear_count": quantized_linear_count,
        "model_load_s": model_load_s,
        "dataset_open_ms": dataset_open_ms,
        "sample_decode_ms": sample_decode_ms,
        "warmup_ms": warmup_ms,
        "latencies_ms": latencies_ms,
        "mean_ms": statistics.fmean(latencies_ms),
        "median_ms": statistics.median(latencies_ms),
        "p95_ms": percentile(latencies_ms, 0.95),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "steady_replans_hz": 1000.0 / statistics.fmean(latencies_ms),
        "action_chunk_steps": config.chunk_size,
        "model_resident_gib": resident_gib,
        "peak_inference_gib": (
            torch.cuda.max_memory_allocated() / 1024**3 if args.device == "cuda" else None
        ),
        "output_shape": list(action.shape),
        "output_preview": action[:8].tolist(),
        "output_semantics": f"uncalibrated {policy_type} base output; not a Piper command",
    }
    return report


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    report = run(args)
    output = args.report
    if output is None:
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        output = TRAINING_LOG_ROOT / "pi05_offline" / f"benchmark_{stamp}.json"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Pi offline benchmark report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
