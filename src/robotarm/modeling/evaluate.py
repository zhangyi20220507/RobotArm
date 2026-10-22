#!/usr/bin/env python3
"""Evaluate a policy on held-out episodes without touching robot hardware."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_episodes(values: list[str]) -> list[int]:
    """Parse space- and/or comma-separated episode indices."""
    try:
        episodes = [
            int(item)
            for value in values
            for item in value.split(",")
            if item.strip()
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "episodes must be space- or comma-separated integers"
        ) from exc
    if not episodes or min(episodes) < 0 or len(set(episodes)) != len(episodes):
        raise argparse.ArgumentTypeError("episodes must be unique non-negative integers")
    return episodes


def batch_sample(sample: dict, camera_keys: list[str]) -> dict:
    import torch

    result = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            value = value.unsqueeze(0)
            if key in camera_keys and value.dtype == torch.uint8:
                value = value.float().div_(255.0)
            result[key] = value
        elif isinstance(value, str):
            result[key] = [value]
        else:
            result[key] = value
    return result


def main() -> int:
    from ..config import defaults

    cfg = defaults("modeling", "evaluate")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--episodes", nargs="+", required=True,
        metavar="EPISODE",
        help="unique episode indices, separated by spaces and/or commas",
    )
    parser.add_argument("--samples", type=int, default=cfg["samples"])
    parser.add_argument("--device", choices=("cuda", "cpu"), default=cfg["device"])
    parser.add_argument("--seed", type=int, default=cfg["seed"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    try:
        args.episodes = parse_episodes(args.episodes)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if args.samples < 1:
        raise RuntimeError("--samples must be positive")

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
    total_episodes = int(info["total_episodes"])
    if max(args.episodes) >= total_episodes:
        raise RuntimeError(f"episode index exceeds dataset range 0..{total_episodes - 1}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    dataset = LeRobotDataset(repo_id=repo_id, root=root, episodes=args.episodes)
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.pretrained_path = str(checkpoint)
    config.device = args.device
    policy = make_policy(config, ds_meta=dataset.meta)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    count = min(args.samples, len(dataset))
    indices = sorted({min(len(dataset) - 1, math.floor(i * len(dataset) / count)) for i in range(count)})
    absolute_errors = []
    hold_state_errors = []
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    with torch.inference_mode():
        for number, index in enumerate(indices, start=1):
            sample = dataset[index]
            reset = getattr(policy, "reset", None)
            if callable(reset):
                reset()
            prediction = postprocessor(policy.select_action(preprocessor(batch_sample(sample, dataset.meta.camera_keys))))
            prediction = prediction.detach().cpu().squeeze(0)
            target = sample["action"].detach().cpu()
            absolute_errors.append(torch.abs(prediction - target))
            state = sample.get("observation.state")
            if state is not None and tuple(state.shape) == tuple(target.shape):
                hold_state_errors.append(torch.abs(state.detach().cpu() - target))
            if number % 25 == 0 or number == len(indices):
                print(f"Evaluated {number}/{len(indices)}", flush=True)

    errors = torch.stack(absolute_errors)
    per_dimension_mae = errors.mean(dim=0)
    per_dimension_rmse = torch.sqrt(torch.mean(errors.square(), dim=0))
    per_dimension_max = errors.max(dim=0).values
    report = {
        "checkpoint": str(checkpoint),
        "dataset": str(root),
        "held_out_episodes": args.episodes,
        "samples": len(indices),
        "seed": args.seed,
        "mae": errors.mean().item(),
        "rmse": torch.sqrt(torch.mean(errors.square())).item(),
        "max_abs_error": errors.max().item(),
        "per_dimension_mae": per_dimension_mae.tolist(),
        "per_dimension_rmse": per_dimension_rmse.tolist(),
        "per_dimension_max_abs_error": per_dimension_max.tolist(),
    }
    if len(hold_state_errors) == len(indices):
        hold_errors = torch.stack(hold_state_errors)
        hold_mae = hold_errors.mean().item()
        report["hold_state_baseline"] = {
            "description": "predict action = observation.state",
            "mae": hold_mae,
            "rmse": torch.sqrt(torch.mean(hold_errors.square())).item(),
            "max_abs_error": hold_errors.max().item(),
            "per_dimension_mae": hold_errors.mean(dim=0).tolist(),
        }
        report["model_to_hold_mae_ratio"] = report["mae"] / hold_mae if hold_mae > 0 else None
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Evaluation failed: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
