"""Run and record the PiperLiftCup-v1 physical-contact grasp baseline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from collections import Counter

import cv2
import gymnasium as gym
import numpy as np

from . import environment  # noqa: F401
from .force_scripted_pick_place import (
    StepBudget, StepLimitExceeded, active_gripper_target, contact_force_norms,
    find_force_grasp, guarded_lift,
)
from .scripted_pick_place import OPEN_GRIPPER, current_tcp, hold, make_action, to_numpy
from .viewer import render_view


class CartesianGraspAdapter(gym.Wrapper):
    """Bounded numerical IK feeding the simulated joint position drives."""

    def __init__(self, env):
        super().__init__(env)
        self.target_rotation = None
        self.last_joints = None
        from sapien.wrapper.pinocchio_model import PinocchioModel

        robot = env.unwrapped.agent.robot
        self.model = PinocchioModel(Path(env.unwrapped.agent.urdf_path).read_text(), [0, 0, -9.81])
        self.model.set_joint_order([j.name for j in robot.active_joints])
        names = [link.name for link in robot.get_links()]
        self.model.set_link_order(names)
        self.tcp_index = names.index("piper_tcp")
        limits = to_numpy(robot.get_qlimits())[0, :6]
        self.lower, self.upper = limits[:, 0] + 1e-4, limits[:, 1] - 1e-4

    def reset(self, **kwargs):
        self.target_rotation = None
        self.last_joints = None
        return self.env.reset(**kwargs)

    def step(self, action):
        from scipy.spatial.transform import Rotation
        from scipy.optimize import least_squares

        action = np.asarray(action, dtype=np.float32)
        if action.shape != (5,):
            raise ValueError("grasp adapter expects XYZ delta plus two gripper targets")
        e = self.unwrapped
        qpos = to_numpy(e.agent.robot.get_qpos())[0].astype(np.float64)
        if not np.any(action[:3]) and self.last_joints is not None:
            return self.env.step(np.r_[self.last_joints, action[-2:]].astype(np.float32))
        target = current_tcp(self) + np.clip(action[:3], -0.15, 0.15) * 0.1
        target -= to_numpy(e.agent.robot.pose.p)[0]
        rotation = self.target_rotation or Rotation.from_matrix(
            to_numpy(e.agent.tcp.pose.to_transformation_matrix())[0, :3, :3]
        )

        def residual(joints):
            self.model.compute_forward_kinematics(np.r_[joints, qpos[6:]])
            pose = self.model.get_link_pose(self.tcp_index)
            matrix = pose.to_transformation_matrix()
            error = (rotation * Rotation.from_matrix(matrix[:3, :3]).inv()).as_rotvec()
            return np.r_[(pose.p - target) * 10, error, (joints - qpos[:6]) * 0.005]

        solution = least_squares(
            residual, np.clip(qpos[:6], self.lower, self.upper),
            bounds=(self.lower, self.upper), diff_step=1e-3, max_nfev=35,
        )
        joints = qpos[:6] + np.clip(solution.x - qpos[:6], -0.08, 0.08)
        self.last_joints = joints.copy()
        return self.env.step(np.r_[joints, action[-2:]].astype(np.float32))


def grasp_options():
    return SimpleNamespace(
        active_gripper_target=OPEN_GRIPPER.copy(), force_threshold=1.0,
        contact_stable_steps=5, close_steps=75, gripper_step_mm=1.0,
        gripper_preload_mm=2.0, grasp_retries=1, min_grasp_z=0.025,
        retry_down_step=0.004, grasp_offset_x=0.035, grasp_offset_y=0.010,
        pre_grasp_z=0.18, grasp_z=0.060, centering_iterations=5,
        centering_tolerance=0.0025, centering_step_budget=25,
        pre_grasp_step_budget=90, grasp_step_budget=100,
        max_grasp_z_error=0.04, exhaustive_grasp_search=False,
        slow_lift=True, abort_on_force_loss=True,
        motion_stagnation_steps=8,
    )


def align_side_grasp(env, render_cb):
    """Approach vertically; place the jaw opening across a lying cup's axis."""
    from scipy.spatial.transform import Rotation

    e = env.unwrapped
    axis = to_numpy(e.cup.pose.to_transformation_matrix())[0, :3, 2]
    current = to_numpy(e.agent.tcp.pose.to_transformation_matrix())[0, :3, :3]
    heading = np.arctan2(current[1, 0], current[0, 0])
    yaw = np.arctan2(axis[1], axis[0]) if abs(axis[2]) < 0.7 else np.pi
    yaw = heading + (yaw - heading + np.pi / 2) % np.pi - np.pi / 2
    desired = Rotation.from_euler("z", yaw) * Rotation.from_euler("x", np.pi)
    env.target_rotation = desired
    target = to_numpy(e.cup.pose.p)[0].copy()
    target[2] = 0.22
    stagnant_steps = 0
    previous_tcp = None
    previous_rotation = None
    for _ in range(120):
        current = Rotation.from_matrix(to_numpy(e.agent.tcp.pose.to_transformation_matrix())[0, :3, :3])
        orientation_error = (desired * current.inv()).magnitude()
        tcp = current_tcp(env)
        delta = target - tcp
        if orientation_error < 0.04 and np.linalg.norm(delta) < 0.008:
            break
        # A joint limit can make the exact overhead attitude unreachable.
        # Continue the approach once IK has stopped improving, while keeping
        # the same orientation target throughout descent and physical grasp.
        if (
            previous_tcp is not None
            and np.linalg.norm(tcp - previous_tcp) < 0.0002
            and (current * previous_rotation.inv()).magnitude() < 0.002
        ):
            stagnant_steps += 1
        else:
            stagnant_steps = 0
        if stagnant_steps >= 8:
            break
        previous_tcp, previous_rotation = tcp.copy(), current
        action = make_action(delta, OPEN_GRIPPER, max_xyz_step=0.015)
        env.step(action)
        render_cb("align_grasp")


def run_episode(env, seed, render_cb):
    env.reset(seed=seed)
    opts = grasp_options()
    status = "no_bilateral_contact"
    peak_clearance = 0.0
    stage_steps = Counter()

    def on_step(label):
        nonlocal peak_clearance
        stage_steps[label] += 1
        peak_clearance = max(peak_clearance, float(env.unwrapped.evaluate()["cup_clearance"][0]))
        render_cb(label)

    render_cb("reset")
    with StepBudget(env, 700) as budget:
        try:
            hold(env, OPEN_GRIPPER, 5, on_step, "settle")
            align_side_grasp(env, on_step)
            # Cup origin is its centre, unlike the older cup-to-container task.
            # Keep the finger contact band near that centre for either pose.
            centre = to_numpy(env.unwrapped.cup.pose.p)[0]
            axis = to_numpy(env.unwrapped.cup.pose.to_transformation_matrix())[0, :3, 2]
            sideways = abs(axis[2]) < 0.7
            opts.grasp_z = float(centre[2] - 0.012 if sideways else centre[2] + 0.002)
            opts.min_grasp_z = 0.008 if sideways else 0.025
            grasped, *_ = find_force_grasp(env, centre, opts, render_cb=on_step)
            if grasped:
                target = current_tcp(env).copy()
                target[2] += 0.15
                lifted, _ = guarded_lift(env, target, opts, render_cb=on_step)
                status = "hold_not_stable" if lifted else "force_lost_during_lift"
                if lifted:
                    hold(env, active_gripper_target(opts), 35, on_step, "hold_lift")
            info = env.unwrapped.evaluate()
            success = bool(to_numpy(info["success"])[0])
            if success:
                status = "success"
        except StepLimitExceeded:
            status, success = "step_limit", False
            info = env.unwrapped.evaluate()
    return dict(
        seed=seed, status=status, success=success, steps=budget.steps,
        peak_clearance_m=peak_clearance,
        final_clearance_m=float(to_numpy(info["cup_clearance"])[0]),
        hold_count=int(to_numpy(info["hold_count"])[0]),
        final_contact_force_n=list(contact_force_norms(env)),
        stage_steps=dict(stage_steps),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cup-pose", choices=["upright", "side", "mixed"], default="upright")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, help="consecutive episode seeds start here")
    parser.add_argument("--cup-top-diameter-mm", type=float, default=60)
    parser.add_argument("--cup-bottom-diameter-mm", type=float, default=48)
    parser.add_argument("--cup-height-mm", type=float, default=90)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--display")
    parser.add_argument("--view", choices=["render", "d405", "d435", "split"], default="split")
    parser.add_argument("--output", type=Path, help="new directory for MP4, preview PNG and results JSON")
    parser.add_argument("--record", action="store_true", help="record video; requires --output")
    args = parser.parse_args()
    if args.episodes < 1 or (args.record and args.output is None):
        parser.error("episodes must be positive; --record requires --output")
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
    if args.display:
        os.environ["DISPLAY"] = args.display
    env = CartesianGraspAdapter(gym.make(
        "PiperLiftCup-v1", cup_pose=args.cup_pose, obs_mode="state",
        control_mode="pd_joint_pos", reward_mode="normalized_dense",
        sim_backend="cpu", render_backend="gpu" if args.record or args.render else "none",
        render_mode="rgb_array" if args.record or args.render else None,
        robot_init_qpos_noise=0.0, max_episode_steps=700,
        cup_top_radius=args.cup_top_diameter_mm / 2000,
        cup_bottom_radius=args.cup_bottom_diameter_mm / 2000,
        cup_height=args.cup_height_mm / 1000,
    ))
    results = []
    writer = None
    window = "Piper - lift cup"
    try:
        if args.render:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 1100, 620)
        for episode in range(args.episodes):
            seed = args.seed + episode
            frame_index = 0

            def render_cb(label):
                nonlocal writer, frame_index
                if not args.render and not args.record:
                    return
                frame = render_view(env, args.view)
                info = env.unwrapped.evaluate()
                force = contact_force_norms(env)
                overlay = [
                    f"PiperLiftCup-v1 | seed={seed} | {label}",
                    f"clearance={float(info['cup_clearance'][0])*1000:.0f} mm / 70 mm"
                    f"  hold={int(info['hold_count'][0])}/10  success={bool(info['success'][0])}",
                    f"contact L/R={force[0]:.2f}/{force[1]:.2f} N",
                ]
                for row, text in enumerate(overlay):
                    cv2.putText(frame, text, (10, 24 + 24 * row), cv2.FONT_HERSHEY_SIMPLEX,
                                0.52, (255, 255, 255), 1, cv2.LINE_AA)
                if args.record:
                    if writer is None:
                        writer = cv2.VideoWriter(
                            str(args.output / f"episode_{seed}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                            env.unwrapped.control_freq, (frame.shape[1], frame.shape[0]),
                        )
                        if not writer.isOpened():
                            raise RuntimeError("Unable to open MP4 video encoder")
                    writer.write(frame)
                    if frame_index == 0:
                        cv2.imwrite(str(args.output / f"scene_{seed}.png"), frame)
                frame_index += 1
                if args.render:
                    cv2.imshow(window, frame)
                    if cv2.waitKey(50) & 0xFF in (27, ord("q")):
                        raise KeyboardInterrupt

            result = run_episode(env, seed, render_cb)
            results.append(result)
            print(json.dumps(result), flush=True)
            if args.record:
                cv2.imwrite(str(args.output / f"final_{seed}.png"), render_view(env, args.view))
                writer.release()
                writer = None
    finally:
        if writer is not None:
            writer.release()
        env.close()
        if args.render:
            cv2.destroyAllWindows()
        if args.output:
            report = dict(
                environment="PiperLiftCup-v1", source="simulation", cup_pose=args.cup_pose,
                control_mode="pd_joint_pos", sim_freq=500, control_freq=20,
                requested_episodes=args.episodes, completed_episodes=len(results),
                geometry_mm=dict(top=args.cup_top_diameter_mm, bottom=args.cup_bottom_diameter_mm,
                                 height=args.cup_height_mm),
                policy="scripted_state_feedback", episodes=results,
                successes=sum(r["success"] for r in results),
            )
            (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"summary: {sum(r['success'] for r in results)}/{len(results)} successful")


if __name__ == "__main__":
    main()
