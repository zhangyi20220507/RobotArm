"""Cloud-only Pi0 LoRA launch plan. Existing train/online-policy are unchanged."""
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

from .quality_filter import default_reviews_path, select_training_episodes


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--quality-reviews", type=Path)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--eval-split", type=float, default=.2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--save-freq", type=int, default=1000)
    p.add_argument("--eval-steps", type=int, default=1000)
    p.add_argument("--max-eval-samples", type=int, default=256)
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--n-action-steps", type=int, default=5)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--smoke", action="store_true", help="20 updates, final save/eval; run reload checker afterward")
    p.add_argument("--execute", action="store_true", help="Without this only validate files and print plan")
    args = p.parse_args(argv)
    positive = (args.batch_size, args.steps, args.save_freq, args.eval_steps,
                args.chunk_size, args.n_action_steps, args.rank, args.alpha)
    if min(positive) < 1 or args.num_workers < 0 or args.max_eval_samples < 0 or args.seed < 0:
        p.error("positive counts and nonnegative workers/sample limit/seed required")
    if not 0 < args.eval_split < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        p.error("eval-split must be in (0,1); lr must be finite and positive")
    if args.n_action_steps > args.chunk_size:
        p.error("n-action-steps must not exceed chunk-size")
    return args


def validate_dataset(root):
    info = json.loads((root / "meta/info.json").read_text())
    features = info["features"]
    for key in ("observation.state", "action"):
        if features.get(key, {}).get("shape") != [7]:
            raise ValueError(f"Expected Piper {key} shape [7]")
    images = sorted(k for k in features if k.startswith("observation.images."))
    if images != ["observation.images.global", "observation.images.wrist"]:
        raise ValueError("This preset requires exactly global+wrist images")
    observations = {k for k in features if k.startswith("observation.")}
    if observations != {"observation.state", *images}:
        raise ValueError("Unexpected observation features; require the model-safe joint schema")
    for key in images:
        shape = features[key]["shape"]
        if len(shape) != 3 or shape[-1] != 3:
            raise ValueError(f"Expected RGB HWC image metadata for {key}")
    conversion = root / "meta/piper_conversion.json"
    if not conversion.is_file():
        raise ValueError("Missing Piper conversion provenance")
    provenance = json.loads(conversion.read_text())
    representation = provenance.get("representation", {})
    if representation.get("space") != "joint" or representation.get("frame") != "absolute":
        raise ValueError("This preset requires absolute joint actions, not TCP/deltas")
    if info["total_episodes"] < 2 or info["fps"] <= 0:
        raise ValueError("Need at least two episodes and a positive FPS")
    return info, provenance


def build_plan(args):
    root = args.dataset_root.expanduser().resolve()
    base = args.base_checkpoint.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise ValueError("Output directory already exists; choose a new run (resume is not implemented here)")
    info, provenance = validate_dataset(root)
    config = json.loads((base / "config.json").read_text())
    if config.get("type") != "pi0" or not (base / "model.safetensors").is_file():
        raise ValueError("Require local Pi0 base config.json and model.safetensors")
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        if not (base / name).is_file():
            raise ValueError(f"Missing base processor: {name}")
    reviews = args.quality_reviews or default_reviews_path(root)
    if args.quality_reviews or reviews.exists():
        episodes, counts = select_training_episodes(root, reviews, args.seed)
        quality = dict(counts)
    else:
        episodes = list(range(info["total_episodes"]))
        random.Random(args.seed).shuffle(episodes)
        quality = {"unreviewed": len(episodes)}
    if len(episodes) < 2 or math.ceil(len(episodes) * args.eval_split) >= len(episodes):
        raise ValueError("Split would leave no training episodes")
    steps = 20 if args.smoke else args.steps
    repo_id = provenance.get("repo_id") or f"local/{root.name}"
    flags = {
        "dataset.repo_id": repo_id, "dataset.root": str(root),
        "dataset.episodes": json.dumps(episodes), "dataset.eval_split": args.eval_split,
        "dataset.image_transforms.enable": "false",
        "policy.path": str(base), "policy.input_features": "null", "policy.output_features": "null",
        "policy.device": "cuda", "policy.dtype": "bfloat16", "policy.use_amp": "true",
        "policy.use_peft": "false",  # Base weights load first; trainer wraps LoRA.
        "policy.empty_cameras": 1, "policy.max_state_dim": 32, "policy.max_action_dim": 32,
        "policy.chunk_size": args.chunk_size, "policy.n_action_steps": args.n_action_steps,
        "policy.gradient_checkpointing": "true", "policy.use_relative_actions": "false",
        "policy.optimizer_lr": args.lr, "policy.scheduler_decay_lr": args.lr / 10,
        "policy.push_to_hub": "false", "policy.repo_id": f"local/{output.name}",
        "peft.method_type": "LORA", "peft.r": args.rank, "peft.lora_alpha": args.alpha,
        "batch_size": args.batch_size, "steps": steps, "seed": args.seed,
        "num_workers": 0 if args.smoke else args.num_workers,
        "eval_steps": steps if args.smoke else args.eval_steps,
        "save_freq": steps if args.smoke else args.save_freq,
        "max_eval_samples": 8 if args.smoke else args.max_eval_samples,
        "wandb.enable": "true", "wandb.mode": "offline", "output_dir": str(output),
        "job_name": output.name,
    }
    command = [sys.executable, "-m", "robotarm.modeling.pi0_lora_driver"]
    command += [f"--{key}={value}" for key, value in flags.items()]
    return {"dataset": str(root), "base_checkpoint": str(base), "output_dir": str(output),
            "total_episodes": info["total_episodes"], "total_frames": info["total_frames"],
            "selected_episodes": episodes, "quality_counts": quality,
            "eval_split": args.eval_split, "dataset_fps": info["fps"],
            "smoke": args.smoke, "flags": flags, "command": command,
            "split_note": "LeRobot splits the shuffled list per task; counts are determined by dataset task metadata"}


def check_dependencies():
    missing = [name for name in ("lerobot", "peft", "torch", "transformers", "accelerate")
               if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Missing {missing}; install the cloud extra: pip install -e '.[pi-lora]'")
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requires a CUDA GPU supporting BF16")


def main(argv=None):
    args = parse_args(argv)
    plan = build_plan(args)
    summary = {key: value for key, value in plan.items() if key not in {"command", "flags", "selected_episodes"}}
    summary["selected_episode_count"] = len(plan["selected_episodes"])
    summary["training_settings"] = {key: value for key, value in plan["flags"].items()
                                    if key not in {"dataset.episodes", "dataset.root", "policy.path"}}
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        print("DRY-RUN: no model loading, CUDA allocation or training; add --execute on the cloud GPU")
        return 0
    check_dependencies()
    output = Path(plan["output_dir"])
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = output.parent / (output.name + ".launch.json")
    if manifest.exists():
        raise ValueError(f"Launch manifest already exists: {manifest}; choose a new run")
    manifest.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    env = dict(os.environ)
    env.setdefault("HF_DATASETS_CACHE", str(output.parent / ".hf_datasets_cache"))
    return subprocess.call(plan["command"], env=env)


if __name__ == "__main__":
    raise SystemExit(main())
