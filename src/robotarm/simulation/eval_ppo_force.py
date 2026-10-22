"""Evaluate a trained PPO policy on PiperPickCubeToBox.

This mirrors the scripted demo UI, but actions come from a trained SB3 PPO
model instead of waypoint rules. No ideal grasp or object binding is used.

Interactive VNC evaluation:
  robotarm simulation evaluate-ppo -- \
    --model artifacts/simulation/piper_ppo_force/piper_ppo_force_final.zip \
    --display :1 \
    --interactive \
    --view split \
    --device cuda

Batch success-rate evaluation:
  robotarm simulation evaluate-ppo -- \
    --model artifacts/simulation/piper_ppo_force/piper_ppo_force_final.zip \
    --episodes 100 \
    --device cuda

Interactive keys:
  g        run PPO policy on current positions
  r        replay same positions
  n        randomize positions and run PPO policy
  1/2/3/4  switch viewer: render / D405 / D435 / split
  q / ESC  quit
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from . import environment  # noqa: F401 - registers env/robot
from mani_skill.utils.structs.pose import Pose
from .viewer import render_view
from .scripted_pick_place import OPEN_GRIPPER, go_home, move_blue_box, set_actor_xyz, to_numpy
from .train_ppo_force import SingleManiSkillEnv
from ..paths import SIMULATION_ARTIFACTS_ROOT


def randomize_eval_task(env, rng):
    e = env.unwrapped
    cube_xy = rng.uniform([-0.07, -0.08], [0.055, 0.08])
    box_xy = rng.uniform([0.13, -0.08], [0.25, 0.08])
    if np.linalg.norm(cube_xy - box_xy) < 0.16:
        box_xy[0] = 0.24

    cube_xyz = np.array([cube_xy[0], cube_xy[1], e.cube_half_size], dtype=np.float32)
    set_actor_xyz(e.cube, cube_xyz, device=e.device)
    move_blue_box(env, box_xy)
    return cube_xyz, box_xy.copy()


def restore_eval_task(env, task_state):
    cube_xyz, box_xy = task_state
    set_actor_xyz(env.unwrapped.cube, cube_xyz, device=env.unwrapped.device)
    move_blue_box(env, box_xy)


def evaluate_info(env):
    info = env.unwrapped.evaluate()
    return {
        "success": bool(to_numpy(info["success"])[0]),
        "in_box": bool(to_numpy(info["is_cube_in_box"])[0]),
        "grasped": bool(to_numpy(info["is_grasped"])[0]),
    }


def step_policy(env, model, obs, deterministic=True):
    action, _ = model.predict(obs, deterministic=deterministic)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    return obs, reward, done, info


def render_frame(env, view, label, task_state=None):
    frame = render_view(env.env, view)
    cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(
        frame,
        "g policy | r replay | n new+policy | 1/2/3/4 view | q quit",
        (12, frame.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    if task_state is not None:
        cube_xyz, box_xy = task_state
        text = f"cube=({cube_xyz[0]:.2f},{cube_xyz[1]:.2f}) box=({box_xy[0]:.2f},{box_xy[1]:.2f})"
        cv2.putText(frame, text, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return frame


def run_policy_episode(env, model, task_state, args, view_ref, render=True):
    restore_eval_task(env.env, task_state)
    go_home(env.env, render_cb=None)
    obs, _ = env.reset()
    restore_eval_task(env.env, task_state)

    total_reward = 0.0
    success_once = False
    in_box_once = False
    label = "policy"
    for step in range(args.max_steps):
        obs, reward, done, info = step_policy(env, model, obs, deterministic=not args.stochastic)
        total_reward += reward
        metrics = evaluate_info(env.env)
        success_once = success_once or metrics["success"]
        in_box_once = in_box_once or metrics["in_box"]

        if render:
            frame = render_frame(
                env,
                view_ref[0],
                f"{label} step={step} r={total_reward:.2f} success={metrics['success']}",
                task_state,
            )
            cv2.imshow("step06_eval_piper_ppo_force", frame)
            key = cv2.waitKey(max(1, int(1000 / args.fps))) & 0xFF
            if key in (27, ord("q")):
                raise KeyboardInterrupt
            if key == ord("1"):
                view_ref[0] = "render"
            elif key == ord("2"):
                view_ref[0] = "d405"
            elif key == ord("3"):
                view_ref[0] = "d435"
            elif key == ord("4"):
                view_ref[0] = "split"
            time.sleep(0.001)

        if done:
            break

    go_home(env.env, render_cb=None)
    final_metrics = evaluate_info(env.env)
    result = dict(
        reward=total_reward,
        success=final_metrics["success"],
        success_once=success_once,
        in_box=final_metrics["in_box"],
        in_box_once=in_box_once,
        steps=step + 1,
    )
    print(result)
    return result


def make_env(args):
    raw_env = gym.make(
        "PiperPickCubeToBox-v1",
        obs_mode="state",
        reward_mode="normalized_dense",
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        human_render_camera_configs={"width": args.render_width, "height": args.render_height},
    )
    return SingleManiSkillEnv(raw_env)


def interactive_eval(env, model, args):
    rng = np.random.default_rng(args.seed)
    cv2.namedWindow("step06_eval_piper_ppo_force", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("step06_eval_piper_ppo_force", args.width, args.height)
    view_ref = [args.view]

    obs, _ = env.reset(seed=args.seed)
    task_state = randomize_eval_task(env.env, rng)
    go_home(env.env, render_cb=None)
    status = "ready"

    while True:
        frame = render_frame(env, view_ref[0], status, task_state)
        cv2.imshow("step06_eval_piper_ppo_force", frame)
        key = cv2.waitKey(max(1, int(1000 / args.fps))) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("1"):
            view_ref[0] = "render"
        elif key == ord("2"):
            view_ref[0] = "d405"
        elif key == ord("3"):
            view_ref[0] = "d435"
        elif key == ord("4"):
            view_ref[0] = "split"
        elif key == ord("g"):
            status = "running policy current"
            run_policy_episode(env, model, task_state, args, view_ref, render=True)
            status = "done - press r replay or n new"
        elif key == ord("r"):
            status = "replay same"
            run_policy_episode(env, model, task_state, args, view_ref, render=True)
            status = "done - same positions"
        elif key == ord("n"):
            obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
            task_state = randomize_eval_task(env.env, rng)
            status = "new random running"
            run_policy_episode(env, model, task_state, args, view_ref, render=True)
            status = "done - new positions"


def batch_eval(env, model, args):
    rng = np.random.default_rng(args.seed)
    results = []
    for ep in range(args.episodes):
        env.reset(seed=int(rng.integers(0, 1_000_000)))
        task_state = randomize_eval_task(env.env, rng)
        print(f"episode {ep}: cube={task_state[0][:2].round(3).tolist()} box={task_state[1].round(3).tolist()}")
        results.append(run_policy_episode(env, model, task_state, args, [args.view], render=args.render))
    success = np.mean([r["success"] for r in results]) if results else 0.0
    success_once = np.mean([r["success_once"] for r in results]) if results else 0.0
    in_box = np.mean([r["in_box"] for r in results]) if results else 0.0
    reward = np.mean([r["reward"] for r in results]) if results else 0.0
    print(
        f"summary episodes={len(results)} success={success:.3f} "
        f"success_once={success_once:.3f} in_box={in_box:.3f} mean_reward={reward:.3f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(SIMULATION_ARTIFACTS_ROOT / "piper_ppo_force/best/best_model.zip"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--display", default=None)
    parser.add_argument("--view", default="split", choices=["render", "d405", "d435", "split"])
    parser.add_argument("--fps", type=float, default=20)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--render-width", type=int, default=512)
    parser.add_argument("--render-height", type=int, default=384)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--sim-backend", default="cpu", choices=["cpu", "gpu", "auto"])
    parser.add_argument("--render-backend", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.display is not None:
        os.environ["DISPLAY"] = args.display

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    env = make_env(args)
    model = PPO.load(model_path, device=args.device)

    try:
        if args.interactive:
            interactive_eval(env, model, args)
        else:
            if args.render:
                cv2.namedWindow("step06_eval_piper_ppo_force", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("step06_eval_piper_ppo_force", args.width, args.height)
            batch_eval(env, model, args)
    finally:
        env.close()
        if args.interactive or args.render:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
