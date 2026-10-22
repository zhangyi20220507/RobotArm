"""Read-only live Pi0.5 shadow inference for the uncalibrated 32D base model."""

from __future__ import annotations

import json
import math
import time

from ..core.camera.realsense import RoleCameras
from ..core.robot import piper as base


def make_batch(joints, gripper_width, image_bgr, task):
    """Pad physical Piper feedback to base Pi0.5 state; never decode output."""
    import cv2
    import numpy as np
    import torch

    state = np.asarray([*joints, gripper_width], dtype=np.float32)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise RuntimeError("Pi0.5 shadow requires finite 7D Piper feedback")
    padded = np.zeros((1, 32), dtype=np.float32)
    padded[0, :7] = state
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return {
        "observation.state": torch.from_numpy(padded),
        "observation.images.base_0_rgb": (
            torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0)
        ),
        "task": [task],
    }


def validate_base_schema(config):
    """Only the known base schema can enter this uncalibrated shadow path."""
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
            "Pi0.5 shadow currently supports only the 32D base schema; "
            "fine-tuned Piper checkpoints need a separate validated action contract"
        )


def verify_weights_loaded(policy, checkpoint):
    """Catch LeRobot 0.6.x silently returning randomly initialized weights."""
    import torch
    from safetensors import safe_open

    probes = ("action_in_proj.bias", "action_out_proj.bias", "time_mlp_in.bias")
    state = policy.state_dict()
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as weights:
        for key in probes:
            model_key = f"model.{key}"
            if key not in weights.keys() or model_key not in state:
                raise RuntimeError(f"Pi0.5 checkpoint weight probe missing: {key}")
            expected = weights.get_tensor(key).to(dtype=state[model_key].dtype)
            actual = state[model_key].detach().cpu()
            if expected.shape != actual.shape or not torch.equal(expected, actual):
                raise RuntimeError(
                    f"Pi0.5 weights were not loaded correctly ({key}); "
                    "refusing even shadow inference"
                )


def run(args, config, checkpoint, digest, task):
    """Observe one Piper and one camera; never call any motor/CAN command API."""
    if args.execute:
        raise RuntimeError("Pi0.5 base shadow runner cannot execute CAN actions")

    import numpy as np
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    from .online_policy import validate_joint_feedback

    validate_base_schema(config)
    if args.device == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Pi0.5 shadow requires CUDA BF16 support on this host")
        config.dtype = "bfloat16"
    config.device = args.device
    print("Pi0.5 base: SHADOW ONLY; 32D output has no Piper action semantics or CAN path.", flush=True)
    policy = PI05Policy.from_pretrained(checkpoint, config=config, strict=True)
    verify_weights_loaded(policy, checkpoint)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    robot = None
    cameras = None
    report = args.report.resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    try:
        robot, robot_config = base.connect_robot(args)
        base.ensure_arm_status_normal(robot, args.feedback_timeout)
        gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        cameras = RoleCameras({"global": args.camera_serial}, 640, 480, args.camera_fps, preview=True)
        cameras.wait_ready()
        reset = getattr(policy, "reset", None)
        deadline = time.monotonic() + args.duration
        count = 0
        with report.open("w", encoding="utf-8") as log:
            while time.monotonic() < deadline:
                tick = time.monotonic()
                joints = [float(v) for v in base.wait_for_feedback(robot, min(args.feedback_timeout, 0.2)).msg]
                validate_joint_feedback(joints, robot_config)
                width = float(base.wait_gripper_width(gripper, min(args.feedback_timeout, 0.2)))
                if not math.isfinite(width) or not -0.003 <= width <= 0.153:
                    raise RuntimeError("Pi0.5 shadow received implausible gripper feedback")
                frames = cameras.latest()
                image_bgr, captured_at, _, _ = frames["global"]
                camera_age_ms = (time.monotonic() - captured_at) * 1000.0
                if camera_age_ms < 0 or camera_age_ms > args.max_camera_age_ms:
                    raise RuntimeError(f"Pi0.5 shadow camera frame stale: {camera_age_ms:.1f}ms")
                batch = make_batch(joints, width, image_bgr, task)
                if callable(reset):
                    reset()  # Replan from the current observation; no stale queued actions.
                started = time.monotonic()
                with torch.inference_mode():
                    raw = postprocessor(policy.select_action(preprocessor(batch)))
                inference_ms = (time.monotonic() - started) * 1000.0
                action = raw.detach().cpu().squeeze(0).numpy()
                if action.shape != (32,) or not np.isfinite(action).all():
                    raise RuntimeError("Pi0.5 base returned invalid 32D action")
                row = {
                    "time": time.time(),
                    "checkpoint_sha256": digest,
                    "policy_type": "pi05",
                    "mode": "shadow",
                    "calibrated": False,
                    "command_sent": False,
                    "state": [*joints, width],
                    "raw_action": action.tolist(),
                    "safe_action": None,
                    "task": task,
                    "camera_serial": args.camera_serial,
                    "camera_age_ms": camera_age_ms,
                    "inference_ms": inference_ms,
                    "healthy": True,
                }
                log.write(json.dumps(row) + "\n")
                log.flush()
                cameras.show(frames, False)
                count += 1
                print(f"\rPi0.5 SHADOW frames={count} infer={inference_ms:.0f}ms camera={camera_age_ms:.0f}ms", end="", flush=True)
                remaining = 1.0 / args.rate - (time.monotonic() - tick)
                if remaining > 0:
                    time.sleep(remaining)
        print(f"\nPi0.5 shadow report: {report}")
        return 0
    except KeyboardInterrupt:
        print(f"\nPi0.5 shadow stopped; no CAN motion command was sent. Report: {report}")
        return 0
    finally:
        if cameras is not None:
            cameras.close()
        base.disconnect_quietly(robot)
