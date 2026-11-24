"""Train pretrained SmolVLA on a local dual-camera Piper dataset."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

from .pi0_lora import validate_dataset
from .quality_filter import default_reviews_path, select_training_episodes


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quality-reviews", type=Path)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--eval-split", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--save-freq", type=int, default=2000)
    parser.add_argument("--eval-steps", type=int, default=2000)
    parser.add_argument("--max-eval-samples", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=20)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--decay-lr", type=float, default=2.5e-6)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--training-mode",
        choices=("expert", "lora"),
        default="expert",
        help="expert trains the full action expert while freezing the VLM; lora is the lower-memory fallback",
    )
    parser.add_argument("--smoke", action="store_true", help="Run 20 updates with a small validation sample")
    parser.add_argument("--execute", action="store_true", help="Without this, validate and print the plan only")
    args = parser.parse_args(argv)
    positive = (
        args.batch_size,
        args.steps,
        args.save_freq,
        args.eval_steps,
        args.chunk_size,
        args.n_action_steps,
        args.rank,
        args.alpha,
        args.log_freq,
    )
    if min(positive) < 1 or args.num_workers < 0 or args.max_eval_samples < 0 or args.seed < 0:
        parser.error("positive counts and nonnegative workers/sample limit/seed required")
    if not 0 < args.eval_split < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("eval-split must be in (0,1); lr must be finite and positive")
    if args.n_action_steps > args.chunk_size:
        parser.error("n-action-steps must not exceed chunk-size")
    if not 0 <= args.warmup_steps < args.steps:
        parser.error("warmup-steps must be nonnegative and smaller than steps")
    if not math.isfinite(args.decay_lr) or not 0 < args.decay_lr <= args.lr:
        parser.error("decay-lr must be positive and no larger than lr")
    return args


def validate_language_tasks(root):
    """Check the task index representation that LeRobot actually reads."""
    import pyarrow.parquet as pq

    path = root / "meta/tasks.parquet"
    if not path.is_file():
        raise ValueError(f"Missing language instructions: {path}")
    tasks = pq.read_table(path).to_pandas()
    if tasks.index.name != "task" or "task_index" not in tasks:
        raise ValueError("tasks.parquet must have a pandas index named 'task' and a task_index column")
    texts = tasks.index.tolist()
    if not texts or any(not isinstance(task, str) or not task.strip() for task in texts):
        raise ValueError("Every language task must be a non-empty string")
    if not tasks.index.is_unique or tasks["task_index"].tolist() != list(range(len(tasks))):
        raise ValueError("Language tasks must be unique with contiguous task_index values starting at zero")
    return texts


def build_plan(args):
    root = args.dataset_root.expanduser().resolve()
    base = args.base_checkpoint.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise ValueError("Output directory already exists; choose a new run")
    info, provenance = validate_dataset(root)
    tasks = validate_language_tasks(root)
    if (root / "TRIM_INCOMPLETE.txt").exists():
        raise ValueError("Dataset trim export is incomplete")
    config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    if config.get("type") != "smolvla" or not (base / "model.safetensors").is_file():
        raise ValueError("Require local SmolVLA base config.json and model.safetensors")
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        if not (base / name).is_file():
            raise ValueError(f"Missing base processor: {name}")

    reviews = args.quality_reviews or default_reviews_path(root)
    if args.quality_reviews or reviews.exists() or (root / "meta/trim_provenance.json").exists():
        episodes, counts = select_training_episodes(root, reviews, args.seed)
        quality = dict(counts)
    else:
        episodes = list(range(info["total_episodes"]))
        random.Random(args.seed).shuffle(episodes)
        quality = {"unreviewed": len(episodes)}
    if len(episodes) < 2 or math.ceil(len(episodes) * args.eval_split) >= len(episodes):
        raise ValueError("Split would leave no training episodes")

    steps = 20 if args.smoke else args.steps
    # Adapt the base config to Piper instead of inheriting its 6D/three-camera
    # schema. Native names also work with our offline and online policy tools.
    input_features = {"observation.state": {"type": "STATE", "shape": [7]}}
    for key in ("observation.images.global", "observation.images.wrist"):
        height, width, channels = info["features"][key]["shape"]
        input_features[key] = {"type": "VISUAL", "shape": [channels, height, width]}
    repo_id = provenance.get("repo_id") or info.get("repo_id") or f"local/{root.name}"
    flags = {
        "dataset.repo_id": repo_id,
        "dataset.root": str(root),
        "dataset.episodes": json.dumps(episodes),
        "dataset.eval_split": args.eval_split,
        "dataset.image_transforms.enable": "false",
        "policy.path": str(base),
        "policy.input_features": json.dumps(input_features),
        "policy.output_features": "null",
        "policy.device": "cuda",
        "policy.use_amp": "true",
        "policy.use_peft": "false",
        "policy.load_vlm_weights": "true",
        "policy.freeze_vision_encoder": "true",
        "policy.train_expert_only": "true",
        "policy.train_state_proj": "true",
        "policy.empty_cameras": 0,
        "policy.chunk_size": args.chunk_size,
        "policy.n_action_steps": args.n_action_steps,
        "policy.optimizer_lr": args.lr,
        "policy.scheduler_decay_lr": args.decay_lr,
        "policy.scheduler_warmup_steps": min(args.warmup_steps, 2) if args.smoke else args.warmup_steps,
        "policy.scheduler_decay_steps": steps,
        "policy.push_to_hub": "false",
        "policy.repo_id": f"local/{output.name}",
        "batch_size": args.batch_size,
        "steps": steps,
        "seed": args.seed,
        "num_workers": 0 if args.smoke else args.num_workers,
        "eval_steps": steps if args.smoke else args.eval_steps,
        "save_freq": steps if args.smoke else args.save_freq,
        "max_eval_samples": 8 if args.smoke else args.max_eval_samples,
        "log_freq": 1 if args.smoke else args.log_freq,
        "env_eval_freq": 0,
        "wandb.enable": "true",
        "wandb.mode": "offline",
        "output_dir": str(output),
        "job_name": output.name,
    }
    if args.training_mode == "lora":
        flags.update(
            {
                "peft.method_type": "LORA",
                "peft.r": args.rank,
                "peft.lora_alpha": args.alpha,
            }
        )
    command = [sys.executable, "-m", "robotarm.modeling.smolvla_lora_driver"]
    command.extend(f"--{key}={value}" for key, value in flags.items())
    return {
        "dataset": str(root),
        "base_checkpoint": str(base),
        "output_dir": str(output),
        "total_episodes": info["total_episodes"],
        "total_frames": info["total_frames"],
        "selected_episodes": episodes,
        "quality_counts": quality,
        "eval_split": args.eval_split,
        "dataset_fps": info["fps"],
        "tasks": tasks,
        "smoke": args.smoke,
        "training_mode": args.training_mode,
        "flags": flags,
        "command": command,
    }


def check_dependencies():
    required = ("lerobot", "peft", "torch", "transformers", "accelerate", "safetensors")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Missing {missing}; install: pip install -e '.[smolvla-training]'")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Requires a CUDA GPU")


def main(argv=None):
    args = parse_args(argv)
    plan = build_plan(args)
    summary = {key: value for key, value in plan.items() if key not in {"command", "flags", "selected_episodes"}}
    summary["selected_episode_count"] = len(plan["selected_episodes"])
    summary["training_settings"] = {
        key: value
        for key, value in plan["flags"].items()
        if key not in {"dataset.episodes", "dataset.root", "policy.path"}
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        print("DRY-RUN: no model loading, CUDA allocation or training; add --execute to start training")
        return 0
    check_dependencies()
    output = Path(plan["output_dir"])
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = output.parent / (output.name + ".launch.json")
    if manifest.exists():
        raise ValueError(f"Launch manifest already exists: {manifest}; choose a new run")
    manifest.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    env = dict(os.environ)
    env.setdefault("HF_DATASETS_CACHE", str(output.parent / ".hf_datasets_cache"))
    return subprocess.call(plan["command"], env=env)


if __name__ == "__main__":
    raise SystemExit(main())
