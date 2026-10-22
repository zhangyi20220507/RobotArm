"""Reload a portable Pi0 adapter and evaluate one dataset observation, offline."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .pi0_lora import check_dependencies, validate_dataset
from .pi0_lora_driver import load_verified_base


def load_pi0_lora(checkpoint, base, config=None, device="cuda"):
    """Load a portable adapter on verified base weights; never edits artifacts."""
    from peft import PeftConfig, PeftModel
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.configs.policies import PreTrainedConfig

    checkpoint, base = Path(checkpoint).resolve(), Path(base).resolve()
    config = config or PreTrainedConfig.from_pretrained(checkpoint)
    if config.type != "pi0" or tuple(config.output_features["action"].shape) != (7,):
        raise ValueError("Require saved Piper Pi0 7D config")
    if not (checkpoint / "adapter_config.json").is_file():
        raise ValueError("Require LoRA adapter_config.json in checkpoint")
    config.device, config.use_peft, config.pretrained_path = device, False, str(base)
    policy = load_verified_base(PI0Policy.from_pretrained, base, base, config=config).to(device)
    adapter_config = PeftConfig.from_pretrained(checkpoint)
    adapter_config.base_model_name_or_path = str(base)
    policy = PeftModel.from_pretrained(policy, checkpoint, config=adapter_config, is_trainable=False)
    policy.eval()
    config.use_peft = True
    return policy


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True, help="saved adapter pretrained_model directory")
    p.add_argument("--base-checkpoint", type=Path, required=True, help="local base location, including after migration")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    check_dependencies()
    root, checkpoint, base = (x.expanduser().resolve() for x in
                              (args.dataset_root, args.checkpoint, args.base_checkpoint))
    info, provenance = validate_dataset(root)
    if not 0 <= args.index < info["total_frames"]:
        raise ValueError("index outside dataset")
    if not (checkpoint / "adapter_config.json").is_file():
        raise ValueError("Require LoRA adapter_config.json in checkpoint")
    os.environ.setdefault("HF_DATASETS_CACHE", str(root.parent / ".hf_datasets_cache"))
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.policies.factory import make_pre_post_processors
    from .evaluate import batch_sample

    config = PreTrainedConfig.from_pretrained(checkpoint)
    if config.type != "pi0" or tuple(config.output_features["action"].shape) != (7,):
        raise ValueError("Require saved Piper Pi0 7D config")
    policy = load_pi0_lora(checkpoint, base, config)
    pre, post = make_pre_post_processors(
        policy_cfg=config, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    dataset = LeRobotDataset(provenance.get("repo_id") or f"local/{root.name}", root=root)
    sample = dataset[args.index]
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        policy.reset()
        action = post(policy.select_action(pre(batch_sample(sample, dataset.meta.camera_keys))))
    action = action.detach().float().cpu().reshape(-1)
    if action.shape != (7,) or not torch.isfinite(action).all():
        raise RuntimeError("Reloaded adapter produced invalid 7D action")
    report = {"mode": "offline_pi0_lora_reload", "command_sent": False,
              "checkpoint": str(checkpoint), "base_checkpoint": str(base),
              "dataset": str(root), "index": args.index, "seed": args.seed,
              "action_rad_m": action.tolist(), "recorded_action_rad_m": sample["action"].tolist(),
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "note": "One-frame reload smoke check; not a task-success or safety approval"}
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
