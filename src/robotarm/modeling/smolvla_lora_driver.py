"""Isolated SmolVLA process with base-weight and trainable-parameter guards."""
from __future__ import annotations


def verify_base_weights(policy, checkpoint):
    import torch
    from pathlib import Path
    from safetensors import safe_open

    state = policy.state_dict()
    names = ("model.state_proj.weight", "model.action_in_proj.weight", "model.action_out_proj.weight")
    verified = 0
    with safe_open(str(Path(checkpoint) / "model.safetensors"), framework="pt", device="cpu") as weights:
        available = set(weights.keys())
        for name in names:
            candidates = (name, name.removeprefix("model."), "model." + name)
            source = next((candidate for candidate in candidates if candidate in available), None)
            if source is None or name not in state:
                raise RuntimeError(f"Missing SmolVLA base projection: {name}")
            actual = state[name].detach().cpu()
            expected = weights.get_tensor(source).to(dtype=actual.dtype)
            if actual.shape != expected.shape or not torch.equal(actual, expected):
                raise RuntimeError(f"SmolVLA base weights not loaded correctly: {name}")
            verified += 1
    if verified == 0:
        raise RuntimeError("Could not verify any SmolVLA base projection weights")


def main():
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401
    from lerobot.scripts import lerobot_train

    original_policy = lerobot_train.make_policy
    original_optimizer = lerobot_train.make_optimizer_and_scheduler

    def guarded_policy(*args, **kwargs):
        cfg = kwargs.get("cfg", args[0] if args else None)
        if cfg is None or cfg.type != "smolvla" or cfg.use_peft:
            raise RuntimeError("This isolated driver only accepts a SmolVLA base before LoRA wrapping")
        policy = original_policy(*args, **kwargs)
        verify_base_weights(policy, cfg.pretrained_path)
        print("SmolVLA base projection weights verified before LoRA wrapping", flush=True)
        return policy

    def guarded_optimizer(cfg, policy):
        trainable = [(name, parameter.numel()) for name, parameter in policy.named_parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("SmolVLA has no trainable parameters")
        if cfg.peft is not None:
            if any("lora_" not in name for name, _ in trainable):
                raise RuntimeError("Expected only LoRA parameters to be trainable")
            mode = "LoRA-only"
        else:
            if not any("lm_expert" in name for name, _ in trainable):
                raise RuntimeError("Expected the SmolVLA action expert to be trainable")
            if any("vlm_with_expert.vlm." in name for name, _ in trainable):
                raise RuntimeError("SmolVLA VLM unexpectedly trainable in expert mode")
            mode = "action-expert"
        if tuple(policy.config.output_features["action"].shape) != (7,):
            raise RuntimeError("Expected Piper 7D output features")
        print(f"Verified {mode} trainable parameters: {sum(size for _, size in trainable):,}", flush=True)
        return original_optimizer(cfg, policy)

    lerobot_train.make_policy = guarded_policy
    lerobot_train.make_optimizer_and_scheduler = guarded_optimizer
    try:
        lerobot_train.train()
    finally:
        lerobot_train.make_policy = original_policy
        lerobot_train.make_optimizer_and_scheduler = original_optimizer


if __name__ == "__main__":
    main()
