"""Force-guarded scripted policy for PiperPickCupToContainer-v1.

The cup is never attached to the TCP. A grasp is accepted only after both
fingers maintain physical contact force, and force loss aborts the transfer.
"""

from __future__ import annotations

import argparse
import os
import time

if os.path.isdir("/usr/share/fonts/truetype/dejavu"):
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import gymnasium as gym
import numpy as np

from . import environment  # noqa: F401 - register robot and environments
from .force_scripted_pick_place import (
    StepBudget,
    StepLimitExceeded,
    contact_force_norms,
    active_gripper_target,
    find_force_grasp,
    guarded_lift,
    guarded_transfer,
    is_force_grasped,
    iter_rigid_components,
    ramp_gripper,
)
from .scripted_pick_place import (
    CLOSED_GRIPPER,
    OPEN_GRIPPER,
    current_tcp,
    go_home,
    hold,
    make_action,
    step_to,
    to_numpy,
)
from .viewer import render_view


def tune_grip_contacts(env, args):
    """Model rubber finger pads without changing cup mass or task geometry."""
    e = env.unwrapped
    material = e.scene.sub_scenes[0].create_physical_material(
        args.grip_static_friction, args.grip_dynamic_friction, 0.0
    )
    for actor in (e.cup, e.agent.finger1_link, e.agent.finger2_link):
        for component in iter_rigid_components(actor):
            for shape in component.collision_shapes:
                shape.set_physical_material(material)


def retreat_after_release(env, target_xyz, args, render_cb=None):
    """Retreat even if placement briefly raised the environment success flag."""
    target_xyz = np.asarray(target_xyz, dtype=np.float32)
    for _ in range(args.retreat_step_budget):
        delta = target_xyz - current_tcp(env)
        env.step(make_action(delta, OPEN_GRIPPER))
        if render_cb is not None:
            render_cb("retreat")
        if np.linalg.norm(delta) < 0.01:
            break


def guarded_vertical_place(env, target_xyz, args, render_cb=None):
    """Lower through short vertical segments and keep checking the grasp."""
    start = current_tcp(env)
    target_xyz = np.asarray(target_xyz, dtype=np.float32)
    for i in range(args.place_segments):
        alpha = (i + 1) / args.place_segments
        waypoint = start * (1 - alpha) + target_xyz * alpha
        step_to(
            env, waypoint, active_gripper_target(args),
            max_steps=args.place_step_budget, tol=0.006,
            render_cb=render_cb, label="lower_into_container",
        )
        grasped, left, right, total = is_force_grasped(env, args.force_threshold)
        print(
            f"place force: left={left:.4f} right={right:.4f} total={total:.4f}"
        )
        if not grasped and i + 1 < args.place_segments:
            return False
    return True


def run_episode(env, args, render_cb=None, recorder=None):
    env.reset(seed=args.episode_seed)
    args.active_gripper_target = OPEN_GRIPPER.copy()
    if recorder is not None:
        recorder.stabilize_after_reset()
        recorder.start(args.episode_seed)
    e = env.unwrapped
    cup_start = to_numpy(e.cup.pose.p)[0].astype(np.float32)
    container = to_numpy(e.target_container_centers)[0].astype(np.float32)
    print(
        f"cup={np.round(cup_start, 3).tolist()} "
        f"container={np.round(container, 3).tolist()}"
    )

    status = "running"
    steps = 0
    try:
        with StepBudget(env, args.max_attempt_steps) as budget:
            grasped, _, _, _ = find_force_grasp(
                env, cup_start, args, render_cb=render_cb
            )
            if not grasped:
                status = "no_bilateral_contact"
                hold(env, OPEN_GRIPPER, args.abort_open_steps, render_cb, "open_failed")
            else:
                cup_at_grasp = to_numpy(e.cup.pose.p)[0].astype(np.float32)
                tcp_at_grasp = current_tcp(env)
                carried_offset = cup_at_grasp - tcp_at_grasp
                lift_target = tcp_at_grasp.copy()
                lift_target[2] = max(args.transport_z, tcp_at_grasp[2] + args.lift_distance)
                lift_ok, _ = guarded_lift(env, lift_target, args, render_cb=render_cb)

                if lift_ok or not args.abort_on_force_loss:
                    # Preserve the measured TCP-to-cup offset instead of assuming
                    # an ideal grasp point.
                    desired_cup = np.array(
                        # Put the cup close to the bottom before releasing; the
                        # container is wide enough for the opened 70 mm gripper.
                        [container[0], container[1], e.container_wall_thickness + 0.006],
                        dtype=np.float32,
                    )
                    place_tcp = desired_cup - carried_offset
                    pre_place = place_tcp.copy()
                    pre_place[2] = max(args.transport_z, place_tcp[2] + 0.10)
                    transfer_ok, _ = guarded_transfer(
                        env, pre_place, args, render_cb=render_cb,
                        label="force_transfer_to_container",
                    )
                    if transfer_ok or not args.abort_on_force_loss:
                        # The cup can settle a few millimetres inside the fingers
                        # during transfer. Re-measure instead of reusing the
                        # pre-lift offset, otherwise the final drop is off-centre.
                        live_offset = (
                            to_numpy(e.cup.pose.p)[0].astype(np.float32)
                            - current_tcp(env)
                        )
                        place_tcp = desired_cup - live_offset
                        guarded_vertical_place(env, place_tcp, args, render_cb)
                        release_ramp_steps = ramp_gripper(
                            env, 0.07, args.gripper_step_mm,
                            render_cb=render_cb, label="release_cup",
                        )
                        remaining_release = max(
                            0, args.release_steps - release_ramp_steps
                        )
                        hold(env, OPEN_GRIPPER, remaining_release, render_cb, "release_hold")
                        retreat = pre_place.copy()
                        retreat_after_release(env, retreat, args, render_cb)
                        hold(env, OPEN_GRIPPER, args.settle_steps, render_cb, "settle")
                        status = "placed"
                    else:
                        status = "force_lost_during_transfer"
                else:
                    status = "force_lost_during_lift"

                if status.startswith("force_lost"):
                    hold(env, OPEN_GRIPPER, args.abort_open_steps, render_cb, "safe_open")
            steps = budget.steps
    except StepLimitExceeded as exc:
        status = "step_limit"
        steps = args.max_attempt_steps
        print(f"attempt aborted: {exc}")
        hold(env, OPEN_GRIPPER, args.abort_open_steps, render_cb, "safe_open")

    info = e.evaluate()
    left, right, total = contact_force_norms(env)
    cup_final = to_numpy(e.cup.pose.p)[0]
    gripper_qpos = to_numpy(e.agent.robot.get_qpos())[0, -2:]
    qualified = status == "placed" and bool(to_numpy(info["success"])[0])
    if recorder is not None:
        qualified = recorder.finish(
            qualified=qualified,
            outcome={
                "status": status,
                "success": bool(to_numpy(info["success"])[0]),
                "in_container": bool(to_numpy(info["is_in_container"])[0]),
                "upright": bool(to_numpy(info["is_upright"])[0]),
                "steps": steps,
            },
        )
    print(
        f"status={status} steps={steps}/{args.max_attempt_steps} "
        f"success={bool(to_numpy(info['success'])[0])} "
        f"qualified={qualified} "
        f"in_container={bool(to_numpy(info['is_in_container'])[0])} "
        f"upright={bool(to_numpy(info['is_upright'])[0])} "
        f"cup_final={np.round(cup_final, 3).tolist()} "
        f"tcp_final={np.round(current_tcp(env), 3).tolist()} "
        f"gripper_qpos={np.round(gripper_qpos, 4).tolist()} "
        f"force_L/R/sum={left:.3f}/{right:.3f}/{total:.3f}N"
    )
    go_home(env, render_cb=render_cb)
    return qualified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--episode-seed", type=int, default=None,
        help="use one exact environment seed for reproducing a rollout",
    )
    parser.add_argument("--display", default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--view", default="split", choices=["render", "d405", "d435", "split"])
    parser.add_argument("--fps", type=float, default=20)
    parser.add_argument("--width", type=int, default=1100)
    parser.add_argument("--height", type=int, default=620)
    parser.add_argument("--force-threshold", type=float, default=2.0)
    parser.add_argument("--contact-stable-steps", type=int, default=8)
    parser.add_argument("--close-steps", type=int, default=28)
    parser.add_argument("--gripper-step-mm", type=float, default=2.0)
    parser.add_argument("--gripper-preload-mm", type=float, default=2.0)
    parser.add_argument("--grasp-retries", type=int, default=1)
    parser.add_argument("--pre-grasp-z", type=float, default=0.17)
    parser.add_argument("--grasp-z", type=float, default=0.070)
    parser.add_argument("--min-grasp-z", type=float, default=0.060)
    parser.add_argument("--grasp-offset-x", type=float, default=0.035)
    parser.add_argument("--grasp-offset-y", type=float, default=0.010)
    parser.add_argument("--centering-iterations", type=int, default=4)
    parser.add_argument("--centering-tolerance", type=float, default=0.003)
    parser.add_argument("--centering-step-budget", type=int, default=25)
    parser.add_argument("--pre-grasp-step-budget", type=int, default=90)
    parser.add_argument("--grasp-step-budget", type=int, default=180)
    parser.add_argument("--max-grasp-z-error", type=float, default=0.06)
    parser.add_argument("--retry-down-step", type=float, default=0.004)
    parser.add_argument("--transport-z", type=float, default=0.22)
    parser.add_argument("--lift-distance", type=float, default=0.12)
    parser.add_argument("--transfer-segments", type=int, default=10)
    parser.add_argument("--transfer-step-budget", type=int, default=40)
    parser.add_argument("--place-step-budget", type=int, default=100)
    parser.add_argument("--place-segments", type=int, default=6)
    parser.add_argument("--retreat-step-budget", type=int, default=90)
    parser.add_argument("--release-steps", type=int, default=35)
    parser.add_argument("--settle-steps", type=int, default=45)
    parser.add_argument("--abort-open-steps", type=int, default=20)
    parser.add_argument("--max-attempt-steps", type=int, default=900)
    parser.add_argument("--grip-static-friction", type=float, default=2.0)
    parser.add_argument("--grip-dynamic-friction", type=float, default=1.5)
    parser.add_argument("--abort-on-force-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--slow-lift", action=argparse.BooleanOptionalAction, default=True)
    # Compatibility fields consumed by the shared grasp search helper.
    parser.set_defaults(exhaustive_grasp_search=False, grasp_offsets=None, grasp_zs=None)
    args = parser.parse_args()
    if args.episodes <= 0 or args.force_threshold <= 0:
        parser.error("--episodes and --force-threshold must be positive")
    if args.display is not None:
        os.environ["DISPLAY"] = args.display

    env = gym.make(
        "PiperPickCupToContainer-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pos",
        reward_mode="normalized_dense",
        render_mode="rgb_array",
        num_envs=1,
        max_episode_steps=max(1200, args.max_attempt_steps + 200),
        robot_init_qpos_noise=0.0,
        human_render_camera_configs={"width": 640, "height": 480},
    )
    tune_grip_contacts(env, args)
    rng = np.random.default_rng(args.seed)
    window = "Piper force-guided cup pick-place"
    if args.render:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, args.width, args.height)
    delay_ms = max(1, int(1000 / args.fps))

    def render_cb(label):
        if not args.render:
            return
        frame = render_view(env, args.view)
        left, right, total = contact_force_norms(env)
        cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
        cv2.putText(
            frame, f"contact L={left:.2f} R={right:.2f} sum={total:.2f} N",
            (12, 56), cv2.FONT_HERSHEY_SIMPLEX, .58, (80, 230, 255), 2,
        )
        qpos = to_numpy(env.unwrapped.agent.robot.get_qpos())[0]
        arm_actual = np.rad2deg(qpos[:6])
        arm_ctrl = env.unwrapped.agent.controller.controllers["arm"]
        arm_target = to_numpy(arm_ctrl._target_qpos)[0]
        grip_ctrl = env.unwrapped.agent.controller.controllers["gripper"]
        grip_target = to_numpy(grip_ctrl._target_qpos)[0]
        actual_width = (qpos[6] - qpos[7]) * 1000.0
        target_width = (grip_target[0] - grip_target[1]) * 1000.0
        cv2.putText(
            frame,
            "joint actual deg: " + " ".join(f"{v:6.1f}" for v in arm_actual),
            (12, 84), cv2.FONT_HERSHEY_SIMPLEX, .48, (210, 210, 210), 1,
        )
        cv2.putText(
            frame,
            "joint target deg: " + " ".join(f"{v:6.1f}" for v in np.rad2deg(arm_target)),
            (12, 108), cv2.FONT_HERSHEY_SIMPLEX, .48, (80, 235, 145), 1,
        )
        cv2.putText(
            frame, f"gripper actual/target: {actual_width:.1f}/{target_width:.1f} mm",
            (12, 132), cv2.FONT_HERSHEY_SIMPLEX, .52, (80, 235, 145), 1,
        )
        cv2.imshow(window, frame)
        if cv2.waitKey(delay_ms) & 0xFF in (27, ord("q")):
            raise KeyboardInterrupt
        time.sleep(0.001)

    successes = 0
    fixed_episode_seed = args.episode_seed
    try:
        for episode in range(args.episodes):
            args.episode_seed = (
                fixed_episode_seed
                if fixed_episode_seed is not None
                else int(rng.integers(0, 1_000_000))
            )
            print(f"episode {episode + 1}/{args.episodes} seed={args.episode_seed}")
            successes += run_episode(env, args, render_cb=render_cb)
    finally:
        env.close()
        if args.render:
            cv2.destroyAllWindows()
    print(f"summary: {successes}/{args.episodes} successful")


if __name__ == "__main__":
    main()
