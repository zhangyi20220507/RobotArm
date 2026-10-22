"""Run a joint-relative or joint-absolute LeRobot ACT checkpoint in ManiSkill."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch

from . import environment  # noqa: F401
from ..paths import WORKSPACE_ROOT

# datasets reads these variables during import, so configure them before
# importing the shared collector helpers (which import LeRobotDataset).
_CACHE_ROOT = WORKSPACE_ROOT / ".cache" / "huggingface"
(_CACHE_ROOT / "datasets").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_CACHE_ROOT))
os.environ.setdefault("HF_DATASETS_CACHE", str(_CACHE_ROOT / "datasets"))

from .collect_lerobot import capture_rgb, robot_state
from .scripted_pick_place import to_numpy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--action-frame", choices=("auto", "absolute", "relative"), default="auto",
        help="auto reads simulation_provenance.json from the dataset",
    )
    parser.add_argument("--max-joint-step-rad", type=float, default=0.10)
    parser.add_argument("--max-gripper-step-m", type=float, default=0.005)
    parser.add_argument("--display", default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/simulation/act_eval.json"))
    args = parser.parse_args()
    if args.display:
        os.environ["DISPLAY"] = args.display
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; use --device cpu")

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    checkpoint = args.checkpoint.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not checkpoint.is_dir():
        checkpoint_root = checkpoint
        while checkpoint_root != checkpoint_root.parent and checkpoint_root.name != "checkpoints":
            checkpoint_root = checkpoint_root.parent
        candidates = (
            sorted(checkpoint_root.glob("*/pretrained_model"))
            if checkpoint_root.name == "checkpoints" and checkpoint_root.is_dir()
            else []
        )
        hint = ""
        if candidates:
            hint = "\navailable checkpoints:\n  " + "\n  ".join(map(str, candidates))
        raise SystemExit(f"checkpoint directory does not exist: {checkpoint}{hint}")
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset directory does not exist: {dataset_root}")
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    provenance_path = dataset_root / "simulation_provenance.json"
    provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.is_file() else {}
    )
    action_frame = args.action_frame
    if action_frame == "auto":
        action_space = provenance.get("action_space")
        if action_space == "joint_absolute":
            action_frame = "absolute"
        elif action_space == "joint_relative":
            action_frame = "relative"
        else:
            raise SystemExit(
                "cannot infer action frame; pass --action-frame absolute or relative"
            )
    repo_id = info.get("repo_id") or f"local/{dataset_root.name}"
    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)
    config = PreTrainedConfig.from_pretrained(checkpoint)
    if getattr(config, "type", None) != "act":
        raise SystemExit(f"checkpoint is not ACT: {getattr(config, 'type', None)!r}")
    config.pretrained_path = str(checkpoint)
    config.device = args.device
    policy = make_policy(config, ds_meta=dataset.meta)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    height, width, _ = info["features"]["observation.images.global"]["shape"]
    env = gym.make(
        "PiperPickCupToContainer-v1", obs_mode="state",
        control_mode=("pd_joint_pos" if action_frame == "absolute" else "pd_joint_delta_pos"),
        reward_mode="normalized_dense",
        render_mode="rgb_array", num_envs=1,
        sensor_configs={"width": width, "height": height},
        robot_init_qpos_noise=0.0, max_episode_steps=args.max_steps + 100,
    )
    rng = np.random.default_rng(args.seed)
    results = []
    window = f"ACT joint-{action_frame} simulation"
    if args.render:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 960, 720)
    try:
        for episode in range(args.episodes):
            episode_seed = int(rng.integers(0, 1_000_000))
            env.reset(seed=episode_seed)
            # Match collection: remove reset/controller transients before policy input.
            for _ in range(20):
                settled_state = robot_state(env)
                arm_hold = settled_state[:6] if action_frame == "absolute" else np.zeros(6)
                env.step(np.concatenate([
                    arm_hold, [settled_state[6] / 2, -settled_state[6] / 2]
                ]).astype(np.float32))
            reset = getattr(policy, "reset", None)
            if callable(reset):
                reset()
            success_streak = 0
            ever_grasped = False
            ever_lifted = False
            inference_ms = []
            for step in range(args.max_steps):
                state = robot_state(env)
                image = capture_rgb(env, "observation.images.global")
                batch = {
                    "observation.state": torch.from_numpy(state).unsqueeze(0),
                    "observation.images.global": torch.from_numpy(image).permute(2, 0, 1).float().div(255).unsqueeze(0),
                }
                start = time.perf_counter()
                with torch.inference_mode():
                    action = postprocessor(policy.select_action(preprocessor(batch)))
                inference_ms.append((time.perf_counter() - start) * 1000)
                predicted = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
                if action_frame == "absolute":
                    arm_command = state[:6] + np.clip(
                        predicted[:6] - state[:6],
                        -args.max_joint_step_rad, args.max_joint_step_rad,
                    )
                    width_target = float(np.clip(
                        state[6] + np.clip(
                            predicted[6] - state[6],
                            -args.max_gripper_step_m, args.max_gripper_step_m,
                        ), 0.0, 0.07,
                    ))
                else:
                    arm_delta = np.clip(
                        predicted[:6], -args.max_joint_step_rad, args.max_joint_step_rad
                    )
                    # Normalized delta controller maps [-1, 1] to [-0.1, 0.1] rad.
                    arm_command = np.clip(arm_delta / 0.1, -1.0, 1.0)
                    width_target = float(np.clip(
                        state[6] + np.clip(
                            predicted[6], -args.max_gripper_step_m, args.max_gripper_step_m,
                        ), 0.0, 0.07,
                    ))
                control = np.concatenate([arm_command, [width_target / 2, -width_target / 2]]).astype(np.float32)
                env.step(control)
                evaluation = env.unwrapped.evaluate()
                flags = {
                    key: bool(to_numpy(evaluation[key])[0])
                    for key in (
                        "success", "is_grasped", "is_lifted", "is_in_container",
                        "is_released", "is_upright", "is_object_static",
                    )
                }
                ever_grasped = ever_grasped or flags["is_grasped"]
                ever_lifted = ever_lifted or flags["is_lifted"]
                # A pick-place rollout must actually complete the pick phase.
                # The environment's instantaneous predicate alone can also be
                # reached by accidentally pushing an untouched cup into the box.
                success = flags["success"] and ever_grasped and ever_lifted
                success_streak = success_streak + 1 if success else 0
                if args.render:
                    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    cv2.putText(frame, f"{action_frame} | episode {episode} step {step} strict_success={success}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, .55, (80, 255, 120), 2)
                    cv2.putText(
                        frame,
                        f"grasped_once={ever_grasped} lifted_once={ever_lifted} "
                        f"in_box={flags['is_in_container']} released={flags['is_released']}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .48,
                        (80, 255, 120) if success else (80, 210, 255), 1,
                    )
                    cv2.imshow(window, frame)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        raise KeyboardInterrupt
                if success_streak >= 10:
                    break
            final_eval = env.unwrapped.evaluate()
            environment_success = bool(to_numpy(final_eval["success"])[0])
            strict_success = environment_success and ever_grasped and ever_lifted
            result = {
                "episode": episode, "seed": episode_seed, "steps": step + 1,
                "success": strict_success,
                "environment_success": environment_success,
                "ever_grasped": ever_grasped,
                "ever_lifted": ever_lifted,
                "in_container": bool(to_numpy(final_eval["is_in_container"])[0]),
                "upright": bool(to_numpy(final_eval["is_upright"])[0]),
                "inference_ms_mean": float(np.mean(inference_ms)),
                "inference_ms_p95": float(np.percentile(inference_ms, 95)),
            }
            results.append(result)
            print(json.dumps(result), flush=True)
    finally:
        env.close()
        if args.render:
            cv2.destroyAllWindows()
    report = {
        "checkpoint": str(checkpoint), "dataset": str(dataset_root),
        "action_frame": action_frame,
        "episodes": args.episodes,
        "success_rate": sum(row["success"] for row in results) / len(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"success={sum(row['success'] for row in results)}/{len(results)} report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
