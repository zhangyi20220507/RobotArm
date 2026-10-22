"""Train PPO for PiperPickCubeToBox with force feedback.

This trains a real PPO policy. It does not use scripted binding or ideal grasp.
The state observation includes gripper contact-force feedback:
``left_finger_force``, ``right_finger_force``, and ``finger_force_norms``.

Recommended GPU batched training on a 32 GB vGPU:
  robotarm simulation train-ppo -- \
    --sim-backend gpu \
    --vec-env maniskill \
    --n-envs 128 \
    --total-timesteps 5000000 \
    --n-steps 64 \
    --batch-size 4096 \
    --n-epochs 4 \
    --device cuda \
    --no-eval

CPU smoke test:
  robotarm simulation train-ppo -- \
    --total-timesteps 32 \
    --n-envs 1 \
    --vec-env dummy \
    --n-steps 16 \
    --batch-size 16 \
    --device cpu

Outputs:
  - Checkpoints: ``artifacts/simulation/piper_ppo_force/checkpoints/``
  - Best model if eval is enabled: ``artifacts/simulation/piper_ppo_force/best/``
  - Final model: ``artifacts/simulation/piper_ppo_force/piper_ppo_force_final.zip``
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv

from . import environment  # noqa: F401 - registers PiperPickCubeToBox-v1
from ..paths import SIMULATION_ARTIFACTS_ROOT

warnings.filterwarnings(
    "ignore",
    message="The `render_mode` attribute is not defined in your environment.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="You are trying to run PPO on the GPU.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*Evaluation environment is not wrapped with a ``Monitor`` wrapper.*",
    category=UserWarning,
)


class SingleManiSkillEnv(gym.Wrapper):
    """Convert ManiSkill's batched torch env with num_envs=1 to SB3 Gym API."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        base_env = env.unwrapped
        self.single_observation_space = base_env.single_observation_space
        self.single_action_space = base_env.single_action_space
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space

    @staticmethod
    def _to_numpy(x: Any):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _squeeze_obs(self, obs):
        obs = self._to_numpy(obs)
        if obs.ndim > 1 and obs.shape[0] == 1:
            obs = obs[0]
        return obs.astype(np.float32, copy=False)

    def _squeeze_info(self, info: dict):
        out = {}
        for key, value in info.items():
            if isinstance(value, torch.Tensor):
                arr = value.detach().cpu().numpy()
                out[key] = arr[0] if arr.shape[:1] == (1,) else arr
            elif isinstance(value, np.ndarray):
                out[key] = value[0] if value.shape[:1] == (1,) else value
            else:
                out[key] = value
        if "success" in out:
            out["is_success"] = bool(np.asarray(out["success"]).item())
        return out

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._squeeze_obs(obs), self._squeeze_info(info)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(self._to_numpy(reward).reshape(-1)[0])
        terminated = bool(self._to_numpy(terminated).reshape(-1)[0])
        truncated = bool(self._to_numpy(truncated).reshape(-1)[0])
        return self._squeeze_obs(obs), reward, terminated, truncated, self._squeeze_info(info)


def make_env(rank: int, args):
    def _init():
        env = gym.make(
            args.env_id,
            obs_mode="state",
            reward_mode=args.reward_mode,
            control_mode=args.control_mode,
            render_mode="rgb_array",
            num_envs=1,
            sim_backend=args.sim_backend,
            render_backend=args.render_backend,
            human_render_camera_configs={"width": 256, "height": 256},
        )
        env = SingleManiSkillEnv(env)
        env.reset(seed=args.seed + rank)
        return env

    return _init


def build_mani_skill_gpu_vec_env(args, n_envs: int, seed_offset: int = 0):
    env = gym.make(
        args.env_id,
        obs_mode="state",
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode="rgb_array",
        num_envs=n_envs,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        human_render_camera_configs={"width": 256, "height": 256},
    )
    env.reset(seed=args.seed + seed_offset)
    base_env = env.unwrapped
    base_env.render_mode = "rgb_array"
    return ManiSkillSB3VectorEnv(base_env)


def build_vec_env(args, n_envs: int, start_rank: int = 0):
    if args.vec_env == "maniskill":
        if args.sim_backend != "gpu":
            raise ValueError("--vec-env maniskill is intended for --sim-backend gpu")
        return build_mani_skill_gpu_vec_env(args, n_envs, seed_offset=start_rank)

    env_fns = [make_env(start_rank + i, args) for i in range(n_envs)]
    if args.vec_env == "subproc" and n_envs > 1:
        env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        env = DummyVecEnv(env_fns)
    return VecMonitor(env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="PiperPickCubeToBox-v1")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--vec-env", choices=["dummy", "subproc", "maniskill"], default="subproc")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control-mode", default="pd_ee_delta_pos")
    parser.add_argument("--reward-mode", default="normalized_dense", choices=["dense", "normalized_dense", "sparse"])
    parser.add_argument("--sim-backend", default="cpu", choices=["cpu", "gpu", "auto"])
    parser.add_argument("--render-backend", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--gae-lambda", type=float, default=0.9)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--no-eval", action="store_true", help="Disable EvalCallback for maximum training throughput.")
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument("--log-dir", default=str(SIMULATION_ARTIFACTS_ROOT / "piper_ppo_force"))
    parser.add_argument("--model-name", default="piper_ppo_force")
    parser.add_argument("--resume", default=None, help="Path to a saved SB3 .zip model to continue training.")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    ckpt_dir = Path(args.log_dir) / "checkpoints"
    best_dir = Path(args.log_dir) / "best"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    env = build_vec_env(args, args.n_envs, start_rank=0)
    eval_env = build_vec_env(args, 1, start_rank=10_000)

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Force feedback is included in the state observation as:")
    print("  left_finger_force, right_finger_force, finger_force_norms")

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    if args.resume:
        model = PPO.load(args.resume, env=env, device=args.device)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            verbose=1,
            tensorboard_log=args.log_dir,
            device=args.device,
            policy_kwargs=policy_kwargs,
        )

    callbacks = [
        CheckpointCallback(
            save_freq=max(args.save_freq // max(args.n_envs, 1), 1),
            save_path=str(ckpt_dir),
            name_prefix=args.model_name,
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
    ]
    if not args.no_eval and args.eval_freq > 0:
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(best_dir),
                log_path=str(Path(args.log_dir) / "eval"),
                eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
                n_eval_episodes=5,
                deterministic=True,
                render=False,
            )
        )

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            tb_log_name=args.model_name,
            reset_num_timesteps=args.resume is None,
            progress_bar=True,
        )
        final_path = Path(args.log_dir) / f"{args.model_name}_final"
        model.save(final_path)
        print(f"Saved final model to {final_path}.zip")
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
