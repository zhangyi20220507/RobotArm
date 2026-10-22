"""Collect qualified Piper cup demonstrations as a LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from ..paths import SIMULATION_DATA_ROOT, WORKSPACE_ROOT

# Hugging Face datasets creates lock files during resume. Keep them in the
# writable project cache instead of a potentially read-only user cache.
_HF_CACHE = WORKSPACE_ROOT / ".cache" / "huggingface"
(_HF_CACHE / "datasets").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_HF_CACHE))
os.environ.setdefault("HF_DATASETS_CACHE", str(_HF_CACHE / "datasets"))

from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from . import environment  # noqa: F401
from .force_pick_cup import run_episode, tune_grip_contacts
from .scripted_pick_place import OPEN_GRIPPER, make_action, to_numpy


JOINT_NAMES = [f"joint_{i}" for i in range(1, 7)] + ["gripper_width"]
TASK = "Pick up the cup and place it upright in the target container"


def robot_state(env) -> np.ndarray:
    qpos = to_numpy(env.unwrapped.agent.robot.get_qpos())[0].astype(np.float32)
    width = np.clip(qpos[6] - qpos[7], 0.0, 0.07)
    return np.concatenate([qpos[:6], np.array([width], dtype=np.float32)])


def capture_rgb(env, uid: str) -> np.ndarray:
    base = env.unwrapped
    base.scene.update_render(update_sensors=True, update_human_render_cameras=False)
    sensor = base.scene.sensors[uid]
    sensor.capture()
    obs = sensor.get_obs(rgb=True, depth=False, position=False, segmentation=False)
    image = obs.get("rgb", obs.get("Color"))
    if image is None:
        image = next(iter(obs.values()))
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


class LeRobotEpisodeRecorder:
    """Intercept env.step and commit only qualified episodes."""

    def __init__(
        self, env, dataset, camera_uids: dict[str, str], task: str,
        settle_min_steps: int, settle_max_steps: int,
        settle_delta_rad: float, settle_stable_steps: int,
        max_joint_delta_rad: float, max_gripper_delta_m: float,
        max_episode_frames: int, action_frame: str, start_episode_index: int = 0,
    ):
        self.env = env
        self.dataset = dataset
        self.camera_uids = camera_uids
        self.task = task
        self.active = False
        self.seed = None
        self.frames = 0
        self.accepted: list[dict] = []
        self.rejected: list[dict] = []
        self.settle_min_steps = settle_min_steps
        self.settle_max_steps = settle_max_steps
        self.settle_delta_rad = settle_delta_rad
        self.settle_stable_steps = settle_stable_steps
        self.max_joint_delta_rad = max_joint_delta_rad
        self.max_gripper_delta_m = max_gripper_delta_m
        self.max_episode_frames = max_episode_frames
        self.action_frame = action_frame
        self.start_episode_index = start_episode_index
        self.settle_result: dict = {}
        self.max_observed_joint_delta = 0.0
        self.max_observed_gripper_delta = 0.0
        self._raw_step = env.step
        env.step = self._step

    def stabilize_after_reset(self):
        """Let the PD controller and physics settle before recording an episode."""
        previous = robot_state(self.env)
        stable = 0
        peak_delta = 0.0
        last_delta = float("inf")
        steps = 0
        for steps in range(1, self.settle_max_steps + 1):
            self._raw_step(make_action([0.0, 0.0, 0.0], OPEN_GRIPPER))
            current = robot_state(self.env)
            last_delta = float(np.max(np.abs(current[:6] - previous[:6])))
            peak_delta = max(peak_delta, last_delta)
            previous = current
            stable = stable + 1 if last_delta <= self.settle_delta_rad else 0
            if steps >= self.settle_min_steps and stable >= self.settle_stable_steps:
                break
        converged = stable >= self.settle_stable_steps
        self.settle_result = {
            "reset_settle_steps": steps,
            "reset_settle_converged": converged,
            "reset_settle_final_delta_rad": last_delta,
            "reset_settle_peak_delta_rad": peak_delta,
        }
        print(
            f"reset settled: steps={steps} converged={converged} "
            f"final_joint_delta={last_delta:.6f} rad"
        )

    def start(self, seed: int):
        self.active = True
        self.seed = int(seed)
        self.frames = 0
        self.max_observed_joint_delta = 0.0
        self.max_observed_gripper_delta = 0.0

    def _step(self, control_action):
        if not self.active:
            return self._raw_step(control_action)
        state = robot_state(self.env)
        images = {
            role: capture_rgb(self.env, uid) for role, uid in self.camera_uids.items()
        }
        result = self._raw_step(control_action)
        next_state = robot_state(self.env)
        if self.action_frame == "absolute":
            # Record what the expert actually achieved on the next simulation
            # frame. The Cartesian IK controller's internal target can switch
            # solution branches even though the physical trajectory is smooth;
            # those jumps are unsuitable behavior-cloning labels.
            action = next_state.copy()
            commanded_delta = next_state - state
        else:
            action = next_state - state
            commanded_delta = action
        self.max_observed_joint_delta = max(
            self.max_observed_joint_delta,
            float(np.max(np.abs(commanded_delta[:6]))),
        )
        self.max_observed_gripper_delta = max(
            self.max_observed_gripper_delta, float(abs(commanded_delta[6]))
        )
        frame = {
            "observation.state": state,
            "action": action.astype(np.float32),
            "task": self.task,
        }
        for role, image in images.items():
            frame[f"observation.images.{role}"] = image
        self.dataset.add_frame(frame)
        self.frames += 1
        return result

    def finish(self, qualified: bool, outcome: dict):
        self.active = False
        joint_ok = self.max_observed_joint_delta <= self.max_joint_delta_rad
        gripper_ok = self.max_observed_gripper_delta <= self.max_gripper_delta_m
        length_ok = self.frames <= self.max_episode_frames
        quality_ok = joint_ok and gripper_ok and length_ok
        accepted = qualified and quality_ok and self.settle_result["reset_settle_converged"]
        record = {
            "seed": self.seed, "frames": self.frames,
            "max_joint_delta_rad": self.max_observed_joint_delta,
            "max_gripper_delta_m": self.max_observed_gripper_delta,
            **self.settle_result, **outcome,
        }
        if accepted:
            self.dataset.save_episode()
            record["episode_index"] = self.start_episode_index + len(self.accepted)
            self.accepted.append(record)
            print(f"dataset: accepted episode {record['episode_index']} ({self.frames} frames)")
        else:
            self.dataset.clear_episode_buffer(delete_images=True)
            if qualified and not joint_ok:
                record["quality_rejection"] = "joint_delta_exceeded"
            elif qualified and not gripper_ok:
                record["quality_rejection"] = "gripper_delta_exceeded"
            elif qualified and not length_ok:
                record["quality_rejection"] = "episode_too_long"
            elif qualified and not self.settle_result["reset_settle_converged"]:
                record["quality_rejection"] = "reset_not_settled"
            self.rejected.append(record)
            reason = record.get("quality_rejection", outcome["status"])
            print(f"dataset: rejected trial seed={self.seed} reason={reason}")
        return accepted

    def close(self):
        self.active = False
        self.env.step = self._raw_step


def features(width: int, height: int, roles: list[str], action_frame: str) -> dict:
    action_names = (
        [f"next_{name}" for name in JOINT_NAMES]
        if action_frame == "absolute"
        else [f"delta_{name}" for name in JOINT_NAMES]
    )
    result = {
        "observation.state": {
            "dtype": "float32", "shape": (7,), "names": {"motors": JOINT_NAMES},
        },
        "action": {
            "dtype": "float32", "shape": (7,),
            "names": {"motors": action_names},
        },
    }
    for role in roles:
        result[f"observation.images.{role}"] = {
            "dtype": "video", "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        }
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100, help="target total qualified episodes")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--max-trials", type=int, default=0, help="0 uses 2x requested episodes")
    parser.add_argument(
        "--output", type=Path,
        default=SIMULATION_DATA_ROOT / "lerobot" / "piper_pick_cup_joint_absolute",
    )
    parser.add_argument("--repo-id", default="local/sim-piper-pick-cup-joint-absolute")
    parser.add_argument(
        "--action-frame", choices=("absolute", "relative"), default="absolute",
        help="absolute records measured state[t+1]; relative records state[t+1]-state[t]",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--include-wrist", action="store_true")
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--encoder-threads", type=int, default=4)
    parser.add_argument("--reset-settle-min-steps", type=int, default=20)
    parser.add_argument("--reset-settle-max-steps", type=int, default=100)
    parser.add_argument("--reset-settle-stable-steps", type=int, default=8)
    parser.add_argument("--reset-settle-delta-rad", type=float, default=0.003)
    parser.add_argument(
        "--max-joint-delta-rad", type=float, default=None,
        help="maximum measured single-frame joint motion; default 0.4 rad",
    )
    parser.add_argument("--max-gripper-delta-m", type=float, default=0.01)
    parser.add_argument("--max-episode-frames", type=int, default=650)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="append until --episodes total episodes; requires an existing finalized dataset",
    )
    parser.add_argument("--sim-backend", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--render-backend", default="gpu", choices=["cpu", "gpu"])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.episodes <= 0 or args.fps <= 0 or args.width <= 0 or args.height <= 0:
        raise SystemExit("episodes, fps, width and height must be positive")
    if args.max_gripper_delta_m <= 0 or args.max_episode_frames <= 0:
        raise SystemExit("quality limits must be positive")
    if args.overwrite and args.resume:
        raise SystemExit("--overwrite and --resume are mutually exclusive")
    max_joint_delta_rad = args.max_joint_delta_rad
    if max_joint_delta_rad is None:
        max_joint_delta_rad = 0.4
    existing_episodes = 0
    if args.output.exists() and not args.resume:
        if not args.overwrite:
            raise SystemExit(f"output exists: {args.output}; use --overwrite or --resume")
        shutil.rmtree(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    roles = ["global"] + (["wrist"] if args.include_wrist else [])
    camera_uids = {"global": "observation.images.global"}
    if args.include_wrist:
        camera_uids["wrist"] = "observation.images.wrist"
    encoder = RGBEncoderConfig(vcodec="h264", crf=23, preset="fast")
    if args.resume:
        if not (args.output / "meta" / "info.json").is_file():
            raise SystemExit(f"cannot resume incomplete/nonexistent dataset: {args.output}")
        dataset = LeRobotDataset.resume(
            repo_id=args.repo_id, root=args.output,
            image_writer_threads=args.image_writer_threads,
            encoder_threads=args.encoder_threads, rgb_encoder=encoder,
        )
        existing_episodes = int(dataset.meta.total_episodes)
        if existing_episodes >= args.episodes:
            print(f"dataset already has {existing_episodes}/{args.episodes} episodes; nothing to do")
            return
    else:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id, root=args.output, fps=args.fps,
            robot_type="piper_simulation", features=features(args.width, args.height, roles, args.action_frame),
            use_videos=True, image_writer_threads=args.image_writer_threads,
            encoder_threads=args.encoder_threads, rgb_encoder=encoder,
        )
    episodes_to_collect = args.episodes - existing_episodes
    max_trials = args.max_trials or episodes_to_collect * 4
    if max_trials < episodes_to_collect:
        raise SystemExit("--max-trials cannot be smaller than the episodes still needed")
    env = gym.make(
        "PiperPickCupToContainer-v1", obs_mode="state",
        control_mode="pd_ee_delta_pos", reward_mode="normalized_dense",
        render_mode="rgb_array", num_envs=1,
        sim_backend=args.sim_backend, render_backend=args.render_backend,
        sensor_configs={"width": args.width, "height": args.height},
        robot_init_qpos_noise=0.0, max_episode_steps=1300,
    )
    # Reuse the force policy defaults without maintaining a second parameter set.
    policy_args = argparse.Namespace(
        force_threshold=2.0, contact_stable_steps=8, close_steps=28,
        gripper_step_mm=2.0, gripper_preload_mm=2.0,
        grasp_retries=1, pre_grasp_z=.17, grasp_z=.07, min_grasp_z=.06,
        grasp_offset_x=.035, grasp_offset_y=.010, centering_iterations=4,
        centering_tolerance=.003, centering_step_budget=25,
        pre_grasp_step_budget=90, grasp_step_budget=180,
        max_grasp_z_error=.06, retry_down_step=.004, transport_z=.22,
        lift_distance=.12, transfer_segments=10, transfer_step_budget=40,
        place_step_budget=100, place_segments=6, retreat_step_budget=90,
        release_steps=35, settle_steps=45, abort_open_steps=20,
        max_attempt_steps=1100, abort_on_force_loss=True, slow_lift=True,
        exhaustive_grasp_search=False, grasp_offsets=None, grasp_zs=None,
        grip_static_friction=2.0, grip_dynamic_friction=1.5,
    )
    tune_grip_contacts(env, policy_args)
    recorder = LeRobotEpisodeRecorder(
        env, dataset, camera_uids, TASK,
        settle_min_steps=args.reset_settle_min_steps,
        settle_max_steps=args.reset_settle_max_steps,
        settle_delta_rad=args.reset_settle_delta_rad,
        settle_stable_steps=args.reset_settle_stable_steps,
        max_joint_delta_rad=max_joint_delta_rad,
        max_gripper_delta_m=args.max_gripper_delta_m,
        max_episode_frames=args.max_episode_frames,
        action_frame=args.action_frame,
        start_episode_index=existing_episodes,
    )
    rng = np.random.default_rng(args.seed)
    trials = 0
    incomplete = False
    try:
        while len(recorder.accepted) < episodes_to_collect and trials < max_trials:
            policy_args.episode_seed = int(rng.integers(0, 1_000_000))
            trials += 1
            print(
                f"trial {trials}/{max_trials}: accepted "
                f"{existing_episodes + len(recorder.accepted)}/{args.episodes} "
                f"seed={policy_args.episode_seed}"
            )
            run_episode(env, policy_args, recorder=recorder)
        if len(recorder.accepted) != episodes_to_collect:
            incomplete = True
        dataset.finalize()
    finally:
        recorder.close()
        env.close()

    provenance = {
        "format": "LeRobot v3", "source": "simulation",
        "environment": "PiperPickCupToContainer-v1",
        "controller": "force_pick_cup", "state_space": "joint_absolute",
        "action_space": f"joint_{args.action_frame}",
        "action_definition": (
            "measured state[t+1] (absolute joints + gripper width)"
            if args.action_frame == "absolute" else "state[t+1]-state[t]"
        ),
        "joint_unit": "rad", "gripper_unit": "m", "fps": args.fps,
        "reset_stabilization": {
            "min_steps": args.reset_settle_min_steps,
            "max_steps": args.reset_settle_max_steps,
            "stable_steps": args.reset_settle_stable_steps,
            "joint_delta_rad": args.reset_settle_delta_rad,
        },
        "max_joint_delta_rad": max_joint_delta_rad,
        "max_gripper_delta_m": args.max_gripper_delta_m,
        "max_episode_frames": args.max_episode_frames,
        "task": TASK, "seed": args.seed, "trials": trials,
        "resumed_from_episodes": existing_episodes,
        "accepted": recorder.accepted, "rejected": recorder.rejected,
        "created_at": datetime.now().astimezone().isoformat(),
    }
    (args.output / "simulation_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    status = "partial" if incomplete else "complete"
    print(
        f"{status}: {existing_episodes + len(recorder.accepted)} total episodes "
        f"({len(recorder.accepted)} added, "
        f"{sum(x['frames'] for x in recorder.accepted)} new frames) -> {args.output}"
    )
    if incomplete:
        raise RuntimeError(
            f"only collected {existing_episodes + len(recorder.accepted)}/"
            f"{args.episodes} total qualified episodes in this run's {trials} trials; "
            "rerun with --resume to continue"
        )


if __name__ == "__main__":
    main()
