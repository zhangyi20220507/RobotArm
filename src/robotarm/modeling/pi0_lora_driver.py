"""Isolated LeRobot training process with a Pi0 pretrained-weight guard."""
from __future__ import annotations

from contextlib import redirect_stdout
import io


def load_verified_base(loader, checkpoint, *args, **kwargs):
    # This LeRobot version catches loading errors and prints instead of
    # raising. Preserve its output and turn that fallback into a hard failure.
    output = io.StringIO()
    try:
        with redirect_stdout(output):
            policy = loader(*args, **kwargs)
    finally:
        print(output.getvalue(), end="", flush=True)
    if any(message in output.getvalue() for message in (
        "Returning model without loading pretrained weights",
        "Warning: Could not load state dict:",
    )):
        raise RuntimeError("Pi0 pretrained loader reported a failure; refusing to continue")
    verify_base_weights(policy, checkpoint)
    return policy


def verify_base_weights(policy, checkpoint):
    import torch
    from pathlib import Path
    from safetensors import safe_open

    state = policy.state_dict()
    with safe_open(str(Path(checkpoint) / "model.safetensors"), framework="pt", device="cpu") as weights:
        for name in ("state_proj.weight", "action_in_proj.weight", "action_out_proj.weight"):
            source = name if name in weights.keys() else "model." + name
            actual = state["model." + name].detach().cpu()
            expected = weights.get_tensor(source).to(dtype=actual.dtype)
            if actual.shape != expected.shape or not torch.equal(actual, expected):
                raise RuntimeError(f"Pi0 base weights not loaded correctly: {name}; refusing random-weight training")


def main():
    from lerobot.policies.pi0.configuration_pi0 import PI0Config  # noqa: F401: register before parsing policy.path
    from lerobot.scripts import lerobot_train

    original = lerobot_train.make_policy
    original_optimizer = lerobot_train.make_optimizer_and_scheduler
    def guarded_policy(*args, **kwargs):
        cfg = kwargs.get("cfg", args[0] if args else None)
        if cfg is None or cfg.type != "pi0" or cfg.use_peft:
            raise RuntimeError("This isolated driver only accepts a Pi0 base before LoRA wrapping")
        policy = load_verified_base(original, cfg.pretrained_path, *args, **kwargs)
        print("Pi0 base projection weights verified before LoRA wrapping", flush=True)
        return policy
    def guarded_optimizer(cfg, policy):
        trainable = [(name, p.numel()) for name, p in policy.named_parameters() if p.requires_grad]
        if not trainable or any("lora_" not in name for name, _ in trainable):
            raise RuntimeError("Expected only LoRA parameters to be trainable; refusing accidental full tuning")
        if tuple(policy.config.output_features["action"].shape) != (7,):
            raise RuntimeError("Expected Piper 7D output features")
        print(f"Verified LoRA-only trainable parameters: {sum(n for _, n in trainable):,}", flush=True)
        return original_optimizer(cfg, policy)
    # Patch only this isolated process's imported function; no installed files
    # or ACT/other policy paths are modified.
    lerobot_train.make_policy = guarded_policy
    lerobot_train.make_optimizer_and_scheduler = guarded_optimizer
    try:
        lerobot_train.train()
    finally:
        lerobot_train.make_policy = original
        lerobot_train.make_optimizer_and_scheduler = original_optimizer


if __name__ == "__main__":
    main()
