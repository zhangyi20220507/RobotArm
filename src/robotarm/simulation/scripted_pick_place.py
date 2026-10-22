"""Scripted Piper pick-and-place demo and interactive controller.

This is a rule/waypoint controller for debugging the scene and collecting
successful trajectories. It uses Cartesian TCP waypoints through ManiSkill's
``pd_ee_delta_pos`` controller. By default it uses an ``ideal_grasp`` attachment
after gripper close to make the demonstration stable; pass ``--no-ideal-grasp``
to test pure physical contact.

Interactive VNC demo:
  robotarm simulation scripted-pick-place -- --display :1 --interactive

Automatic visual rollout:
  robotarm simulation scripted-pick-place -- --display :1 --render --view split

Batch run without rendering:
  robotarm simulation scripted-pick-place -- --episodes 10 --seed 0

Interactive keys:
  g        execute policy on current red-cube / blue-bin positions
  r        replay the same positions
  n        randomize positions and execute
  1/2/3/4  switch viewer: render / D405 / D435 / split
  q / ESC  quit

After every execution, the robot returns exactly to its Home joint pose.
"""

from __future__ import annotations

import argparse
import os
import time

import cv2
import gymnasium as gym
import numpy as np
import sapien
import torch

from . import environment  # noqa: F401 - registers env/robot
from mani_skill.utils.structs.pose import Pose
from .viewer import render_view


OPEN_GRIPPER = np.array([0.035, -0.035], dtype=np.float32)
CLOSED_GRIPPER = np.array([0.0, 0.0], dtype=np.float32)
GRASP_TCP_XY_OFFSET = np.array([0.035, 0.010], dtype=np.float32)
CARRIED_OBJ_XY_OFFSET_FROM_TCP = np.array([-0.026, -0.030], dtype=np.float32)
CARRIED_OBJ_OFFSET_FROM_TCP = np.array([-0.026, -0.030, 0.0], dtype=np.float32)
HOME_TCP = np.array([-0.16, 0.0, 0.34], dtype=np.float32)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def set_actor_xyz(actor, xyz, quat=None, device="cpu"):
    xyz_t = torch.tensor(xyz, dtype=torch.float32, device=device).reshape(1, 3)
    if quat is None:
        quat_t = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    else:
        quat_t = torch.tensor(quat, dtype=torch.float32, device=device).reshape(1, 4)
    actor.set_pose(Pose.create_from_pq(xyz_t, quat_t))


def move_blue_box(env, center):
    e = env.unwrapped
    spec = e.box_specs[0]
    spec["center"] = (float(center[0]), float(center[1]))
    e.box_center = spec["center"]
    e.box_inner_half_size = spec["inner_half_size"]
    e.box_wall_height = spec["height"]

    cx, cy = spec["center"]
    hx, hy = spec["inner_half_size"]
    t = e.box_wall_thickness
    h = spec["height"]
    bottom_z = t / 2
    wall_z = t + h / 2
    poses = [
        [cx, cy, bottom_z],
        [cx, cy + hy + t / 2, wall_z],
        [cx, cy - hy - t / 2, wall_z],
        [cx - hx - t / 2, cy, wall_z],
        [cx + hx + t / 2, cy, wall_z],
    ]
    for actor, pos in zip(e.box_parts[:5], poses):
        actor.set_pose(sapien.Pose(p=pos))

    if hasattr(e, "target_box_centers"):
        e.target_box_centers = torch.tensor([[cx, cy]], dtype=torch.float32, device=e.device)
    goal = [cx, cy, e.box_wall_thickness + e.cube_half_size]
    set_actor_xyz(e.goal_site, goal, device=e.device)


def randomize_task(env, rng):
    e = env.unwrapped
    cube_xy = rng.uniform([-0.07, -0.08], [0.055, 0.08])
    box_xy = rng.uniform([0.13, -0.08], [0.25, 0.08])

    if np.linalg.norm(cube_xy - box_xy) < 0.16:
        box_xy[0] = 0.24

    set_actor_xyz(
        e.cube,
        [cube_xy[0], cube_xy[1], e.cube_half_size],
        device=e.device,
    )
    move_blue_box(env, box_xy)
    return np.array([cube_xy[0], cube_xy[1], e.cube_half_size], dtype=np.float32), box_xy


def make_action(delta_xyz, gripper, pos_scale=0.1, max_xyz_step=0.1):
    """Build a bounded Cartesian delta action.

    Bound Cartesian motion to the controller's configured 100 mm action range.
    """
    delta = np.asarray(delta_xyz, dtype=np.float32)
    norm = float(np.linalg.norm(delta))
    if norm > max_xyz_step:
        delta = delta * (max_xyz_step / norm)
    arm = np.clip(delta / pos_scale, -1.0, 1.0)
    return np.concatenate([arm, gripper]).astype(np.float32)


def current_tcp(env):
    return to_numpy(env.unwrapped.agent.tcp_pose.p)[0].astype(np.float32)


def attach_cube_to_tcp(env, offset=CARRIED_OBJ_OFFSET_FROM_TCP):
    e = env.unwrapped
    tcp = current_tcp(env)
    cube_xyz = tcp + offset
    cube_xyz[2] = max(float(cube_xyz[2]), e.cube_half_size)
    set_actor_xyz(e.cube, cube_xyz, device=e.device)


def step_to(
    env,
    target_xyz,
    gripper,
    max_steps=80,
    tol=0.012,
    render_cb=None,
    label="",
    attach_cube=False,
    max_xyz_step=0.1,
):
    target_xyz = np.asarray(target_xyz, dtype=np.float32)
    last_info = None
    for i in range(max_steps):
        tcp = current_tcp(env)
        delta = target_xyz - tcp
        action = make_action(delta, gripper, max_xyz_step=max_xyz_step)
        _, reward, terminated, truncated, info = env.step(action)
        if attach_cube:
            attach_cube_to_tcp(env)
        last_info = info
        if render_cb is not None:
            render_cb(label)
        if np.linalg.norm(delta) < tol:
            break
        if bool(np.asarray(terminated).any() or np.asarray(truncated).any()):
            break
    return last_info


def hold(env, gripper, steps=15, render_cb=None, label="", attach_cube=False):
    info = None
    for _ in range(steps):
        action = make_action([0, 0, 0], gripper)
        _, _, _, _, info = env.step(action)
        if attach_cube:
            attach_cube_to_tcp(env)
        if render_cb is not None:
            render_cb(label)
    return info


def snap_home(env, render_cb=None):
    e = env.unwrapped
    qpos = e.agent.keyframes["rest"].qpos.copy()
    qpos[-2:] = OPEN_GRIPPER
    e.agent.reset(qpos)
    e.agent.robot.set_pose(sapien.Pose([-0.52, 0, 0]))
    if render_cb is not None:
        render_cb("home")


def go_home(env, render_cb=None):
    step_to(env, HOME_TCP, OPEN_GRIPPER, max_steps=160, tol=0.004, render_cb=render_cb, label="home")
    snap_home(env, render_cb=render_cb)
    hold(env, OPEN_GRIPPER, steps=10, render_cb=render_cb, label="home")


def scripted_episode(env, rng, render_cb=None, ideal_grasp=True, reset_scene=True, randomize=True, task_state=None):
    if reset_scene:
        env.reset(seed=int(rng.integers(0, 1_000_000)))
    if randomize or task_state is None:
        cube_xyz, box_xy = randomize_task(env, rng)
        task_state = (cube_xyz.copy(), box_xy.copy())
    else:
        cube_xyz, box_xy = task_state
        set_actor_xyz(env.unwrapped.cube, cube_xyz, device=env.unwrapped.device)
        move_blue_box(env, box_xy)
    e = env.unwrapped

    print(f"cube={cube_xyz[:2].round(3).tolist()} box={box_xy.round(3).tolist()}")

    grasp_xy = cube_xyz[:2] + GRASP_TCP_XY_OFFSET
    place_tcp_xy = box_xy - CARRIED_OBJ_XY_OFFSET_FROM_TCP

    pre_grasp = np.array([grasp_xy[0], grasp_xy[1], 0.16], dtype=np.float32)
    grasp = np.array([grasp_xy[0], grasp_xy[1], 0.015], dtype=np.float32)
    lift = np.array([grasp_xy[0], grasp_xy[1], 0.20], dtype=np.float32)
    pre_place = np.array([place_tcp_xy[0], place_tcp_xy[1], 0.22], dtype=np.float32)
    place = np.array([place_tcp_xy[0], place_tcp_xy[1], 0.075], dtype=np.float32)
    retreat = np.array([place_tcp_xy[0], place_tcp_xy[1], 0.22], dtype=np.float32)

    step_to(env, pre_grasp, OPEN_GRIPPER, max_steps=100, render_cb=render_cb, label="pre_grasp")
    step_to(env, grasp, OPEN_GRIPPER, max_steps=70, render_cb=render_cb, label="grasp_down")
    hold(env, CLOSED_GRIPPER, steps=25, render_cb=render_cb, label="close", attach_cube=ideal_grasp)
    step_to(env, lift, CLOSED_GRIPPER, max_steps=90, render_cb=render_cb, label="lift", attach_cube=ideal_grasp)
    step_to(env, pre_place, CLOSED_GRIPPER, max_steps=120, render_cb=render_cb, label="pre_place", attach_cube=ideal_grasp)
    step_to(env, place, CLOSED_GRIPPER, max_steps=80, render_cb=render_cb, label="place_down", attach_cube=ideal_grasp)
    if ideal_grasp:
        set_actor_xyz(
            e.cube,
            [box_xy[0], box_xy[1], e.box_wall_thickness + e.cube_half_size],
            device=e.device,
        )
    hold(env, OPEN_GRIPPER, steps=25, render_cb=render_cb, label="open")
    step_to(env, retreat, OPEN_GRIPPER, max_steps=80, render_cb=render_cb, label="retreat")
    hold(env, OPEN_GRIPPER, steps=40, render_cb=render_cb, label="settle")

    info = e.evaluate()
    print(
        "success=", bool(to_numpy(info["success"])[0]),
        "in_box=", bool(to_numpy(info["is_cube_in_box"])[0]),
        "grasped=", bool(to_numpy(info["is_grasped"])[0]),
    )
    go_home(env, render_cb=render_cb)
    return info, task_state


def draw_interactive_frame(env, view, label):
    frame = render_view(env, view)
    cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(
        frame,
        "g grasp | r replay | n new+grasp | 1/2/3/4 view | q quit",
        (12, frame.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    return frame


def interactive_loop(env, rng, args):
    cv2.namedWindow("step03_scripted_piper_pick_place", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("step03_scripted_piper_pick_place", args.width, args.height)
    delay_ms = max(1, int(1000 / args.fps))
    view = args.view
    task_state = None
    status = "home"

    def render_cb(label):
        frame = draw_interactive_frame(env, view, label)
        cv2.imshow("step03_scripted_piper_pick_place", frame)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key in (27, ord("q")):
            raise KeyboardInterrupt
        time.sleep(0.001)

    env.reset(seed=int(rng.integers(0, 1_000_000)))
    task_state = randomize_task(env, rng)
    go_home(env, render_cb=render_cb)

    while True:
        frame = draw_interactive_frame(env, view, status)
        cv2.imshow("step03_scripted_piper_pick_place", frame)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("1"):
            view = "render"
        elif key == ord("2"):
            view = "d405"
        elif key == ord("3"):
            view = "d435"
        elif key == ord("4"):
            view = "split"
        elif key == ord("g"):
            status = "grasp current"
            _, task_state = scripted_episode(
                env, rng, render_cb=render_cb, ideal_grasp=not args.no_ideal_grasp, reset_scene=False, randomize=False, task_state=task_state
            )
            status = "done - press r replay or n new"
        elif key == ord("r"):
            status = "replay same"
            env.reset(seed=int(rng.integers(0, 1_000_000)))
            _, task_state = scripted_episode(
                env, rng, render_cb=render_cb, ideal_grasp=not args.no_ideal_grasp, reset_scene=False, randomize=False, task_state=task_state
            )
            status = "done - same positions"
        elif key == ord("n"):
            status = "new random"
            env.reset(seed=int(rng.integers(0, 1_000_000)))
            _, task_state = scripted_episode(
                env, rng, render_cb=render_cb, ideal_grasp=not args.no_ideal_grasp, reset_scene=False, randomize=True
            )
            status = "done - new positions"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--display", default=None)
    parser.add_argument("--view", default="split", choices=["render", "d405", "d435", "split"])
    parser.add_argument("--fps", type=float, default=20)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--render", action="store_true", help="Show the OpenCV viewer while running.")
    parser.add_argument("--interactive", action="store_true", help="Open a keyboard controlled demo window.")
    parser.add_argument("--no-ideal-grasp", action="store_true", help="Use only physical contacts; less reliable for this URDF.")
    args = parser.parse_args()

    if args.display is not None:
        os.environ["DISPLAY"] = args.display

    env = gym.make(
        "PiperPickCubeToBox-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
        num_envs=1,
        human_render_camera_configs={"width": 512, "height": 384},
    )

    delay_ms = max(1, int(1000 / args.fps))
    rng = np.random.default_rng(args.seed)
    try:
        if args.interactive:
            interactive_loop(env, rng, args)
            return

        if args.render:
            cv2.namedWindow("step03_scripted_piper_pick_place", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("step03_scripted_piper_pick_place", args.width, args.height)

        def render_cb(label):
            if not args.render:
                return
            frame = render_view(env, args.view)
            cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("step03_scripted_piper_pick_place", frame)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (27, ord("q")):
                raise KeyboardInterrupt
            time.sleep(0.001)

        for ep in range(args.episodes):
            print(f"episode {ep}")
            scripted_episode(env, rng, render_cb=render_cb, ideal_grasp=not args.no_ideal_grasp)
    finally:
        env.close()
        if args.render or args.interactive:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
