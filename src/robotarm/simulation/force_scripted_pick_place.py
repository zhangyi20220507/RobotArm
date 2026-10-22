"""Force-feedback scripted Piper pick-and-place demo.

This is the physical-contact version of step03. It uses the same waypoint
controller, but it does NOT bind the cube to the TCP after closing the gripper.
Instead, it reads the contact-force signals from the two gripper fingers and
only treats the grasp as valid when force exceeds a threshold.

Interactive VNC demo:
  robotarm simulation force-scripted-pick-place -- --display :1 --interactive

Automatic visual rollout:
  robotarm simulation force-scripted-pick-place -- --display :1 --render --view split

Batch run without rendering:
  robotarm simulation force-scripted-pick-place -- --episodes 10 --seed 0

Useful tuning options:
  --force-threshold 5.0      minimum combined finger-force norm for grasp
  --exhaustive-grasp-search   use the legacy search grid for debugging
  --grasp-z 0.006             TCP z used at grasp pose
  --slow-lift                 lift in smaller increments after force contact
  --no-abort-on-force-loss    continue placing after force loss (debug only)
  --max-attempt-steps 650     hard env.step budget for one grasp attempt

Interactive keys:
  g        execute policy on current red-cube / blue-bin positions
  r        replay the same positions
  n        randomize positions and execute
  1/2/3/4  switch viewer: render / D405 / D435 / split
  q / ESC  quit

After every execution attempt, the robot returns exactly to its Home joint pose.
"""

from __future__ import annotations

import argparse
import os
import time

import cv2
import gymnasium as gym
import numpy as np

from . import environment  # noqa: F401 - registers env/robot
from .viewer import render_view
from .scripted_pick_place import (
    CARRIED_OBJ_XY_OFFSET_FROM_TCP,
    CLOSED_GRIPPER,
    OPEN_GRIPPER,
    current_tcp,
    go_home,
    hold,
    move_blue_box,
    set_actor_xyz,
    step_to,
    to_numpy,
)




class StepLimitExceeded(RuntimeError):
    pass


class StepBudget:
    """Temporarily count env.step calls and stop one grasp attempt at a hard limit."""

    def __init__(self, env, max_steps):
        self.env = env
        self.max_steps = int(max_steps)
        self.steps = 0
        self._orig_step = None

    def __enter__(self):
        self._orig_step = self.env.step

        def counted_step(action):
            if self.steps >= self.max_steps:
                raise StepLimitExceeded(f"attempt step budget exhausted: {self.steps}/{self.max_steps}")
            self.steps += 1
            return self._orig_step(action)

        self.env.step = counted_step
        return self

    def __exit__(self, exc_type, exc, tb):
        self.env.step = self._orig_step
        return False

def iter_rigid_components(actor):
    for obj in getattr(actor, "_objs", []):
        for comp in getattr(obj, "components", []):
            if hasattr(comp, "collision_shapes"):
                yield comp


def apply_contact_tuning(env, args):
    """Increase contact friction for this debug controller without binding objects."""
    if args.no_contact_tuning:
        return
    e = env.unwrapped
    mat = e.scene.sub_scenes[0].create_physical_material(
        args.static_friction,
        args.dynamic_friction,
        args.restitution,
    )
    actors = [e.cube, e.agent.finger1_link, e.agent.finger2_link]
    for actor in actors:
        for comp in iter_rigid_components(actor):
            for shape in comp.collision_shapes:
                shape.set_physical_material(mat)
    for comp in iter_rigid_components(e.cube):
        if hasattr(comp, "set_mass"):
            comp.set_mass(args.cube_mass)
    print(
        "contact tuning:",
        f"static_friction={args.static_friction}",
        f"dynamic_friction={args.dynamic_friction}",
        f"cube_mass={args.cube_mass}",
    )


def randomize_force_task(env, rng, max_cube_x):
    """Randomize within the low-grasp workspace of the fixed TCP orientation."""
    e = env.unwrapped
    cube_xy = rng.uniform([-0.07, -0.08], [max_cube_x, 0.08])
    box_xy = rng.uniform([0.13, -0.08], [0.25, 0.08])
    if np.linalg.norm(cube_xy - box_xy) < 0.16:
        box_xy[0] = 0.24
    cube_xyz = np.array(
        [cube_xy[0], cube_xy[1], e.cube_half_size], dtype=np.float32
    )
    set_actor_xyz(e.cube, cube_xyz, device=e.device)
    move_blue_box(env, box_xy)
    return cube_xyz, box_xy.astype(np.float32)


def contact_force_norms(env):
    """Return left, right, and combined contact-force norms against the red cube."""
    e = env.unwrapped
    left_force = e.scene.get_pairwise_contact_forces(e.agent.finger1_link, e.cube)
    right_force = e.scene.get_pairwise_contact_forces(e.agent.finger2_link, e.cube)
    left = float(np.linalg.norm(to_numpy(left_force)[0]))
    right = float(np.linalg.norm(to_numpy(right_force)[0]))
    return left, right, left + right


def is_force_grasped(env, force_threshold):
    left, right, total = contact_force_norms(env)
    # A large force on only one finger is a collision, not a stable grasp.
    per_finger = force_threshold * 0.2
    grasped = total >= force_threshold and min(left, right) >= per_finger
    return grasped, left, right, total


def gripper_target(width_m):
    half = float(np.clip(width_m, 0.0, 0.07)) / 2.0
    return np.array([half, -half], dtype=np.float32)


def current_gripper_width(env):
    qpos = to_numpy(env.unwrapped.agent.robot.get_qpos())[0]
    return float(np.clip(qpos[6] - qpos[7], 0.0, 0.07))


def active_gripper_target(args):
    return getattr(args, "active_gripper_target", CLOSED_GRIPPER)


def ramp_gripper(env, target_width, step_mm, render_cb=None, label="gripper_ramp"):
    """Move the absolute width target gradually, matching real teleoperation."""
    controller = env.unwrapped.agent.controller.controllers["gripper"]
    target_qpos = to_numpy(controller._target_qpos)[0]
    width = float(np.clip(target_qpos[0] - target_qpos[1], 0.0, 0.07))
    target_width = float(np.clip(target_width, 0.0, 0.07))
    step_m = max(float(step_mm) / 1000.0, 1e-5)
    direction = 1.0 if target_width > width else -1.0
    steps = 0
    while abs(target_width - width) > 1e-6:
        width = target_width if abs(target_width - width) <= step_m else width + direction * step_m
        hold(env, gripper_target(width), steps=1, render_cb=render_cb, label=label)
        steps += 1
    return steps


def close_until_force(env, args, render_cb=None):
    """Close until bilateral contact is stable instead of waiting a fixed hold."""
    best = (False, 0.0, 0.0, 0.0)
    for retry in range(args.grasp_retries + 1):
        stable_steps = 0
        info = None
        left = right = total = 0.0
        command_width = current_gripper_width(env)
        for close_step in range(args.close_steps):
            command_width = max(0.0, command_width - args.gripper_step_mm / 1000.0)
            info = hold(
                env,
                gripper_target(command_width),
                steps=1,
                render_cb=render_cb,
                label=f"close force retry={retry}",
            )
            grasped, left, right, total = is_force_grasped(
                env, args.force_threshold
            )
            best = max(best, (grasped, left, right, total), key=lambda item: item[3])
            stable_steps = stable_steps + 1 if grasped else 0
            if stable_steps >= args.contact_stable_steps:
                hold_width = max(
                    0.0,
                    current_gripper_width(env) - args.gripper_preload_mm / 1000.0,
                )
                args.active_gripper_target = gripper_target(hold_width)
                hold(env, args.active_gripper_target, steps=2, render_cb=render_cb, label="force_hold")
                print(
                    f"force contact stabilized after {close_step + 1} close steps; "
                    f"hold_target={hold_width * 1000:.1f} mm"
                )
                return True, info
        print(
            f"force retry={retry}: left={left:.4f} right={right:.4f} "
            f"total={total:.4f} threshold={args.force_threshold:.4f}"
        )

        if retry < args.grasp_retries:
            # Re-open smoothly. A one-frame 0 -> 70 mm target jump is unlike the
            # real teleoperation data and teaches an unnecessarily hard action.
            ramp_gripper(
                env, 0.07, args.gripper_step_mm,
                render_cb=render_cb, label="retry_open",
            )
            tcp = current_tcp(env)
            # A tiny downward nudge helps when the fingers close just above the cube.
            target = tcp.copy()
            target[2] = max(args.min_grasp_z, target[2] - args.retry_down_step)
            step_to(
                env,
                target,
                OPEN_GRIPPER,
                max_steps=18,
                tol=0.004,
                render_cb=render_cb,
                label="retry_down",
            )

    print(
        "force grasp failed: "
        f"best_left={best[1]:.4f} best_right={best[2]:.4f} best_total={best[3]:.4f}"
    )
    return False, info


def finger_center_xy(env):
    """Return the live midpoint between the two gripper link origins."""
    e = env.unwrapped
    left = to_numpy(e.agent.finger1_link.pose.p)[0]
    right = to_numpy(e.agent.finger2_link.pose.p)[0]
    return ((left + right) * 0.5)[:2].astype(np.float32)


def align_fingers_over_cube(env, target_z, args, render_cb=None, label="align"):
    """Servo the live finger midpoint over the cube at a fixed TCP height."""
    for _ in range(args.centering_iterations):
        cube = to_numpy(env.unwrapped.cube.pose.p)[0].astype(np.float32)
        error_xy = cube[:2] - finger_center_xy(env)
        if np.linalg.norm(error_xy) <= args.centering_tolerance:
            break
        tcp = current_tcp(env)
        target = np.array(
            [tcp[0] + error_xy[0], tcp[1] + error_xy[1], target_z],
            dtype=np.float32,
        )
        step_to(
            env,
            target,
            OPEN_GRIPPER,
            max_steps=args.centering_step_budget,
            tol=args.centering_tolerance,
            render_cb=render_cb,
            label=label,
        )


def try_centered_force_grasp(env, args, render_cb=None):
    """Use live finger geometry for one fast grasp attempt."""
    cube = to_numpy(env.unwrapped.cube.pose.p)[0].astype(np.float32)
    initial_xy = cube[:2] + np.array(
        [args.grasp_offset_x, args.grasp_offset_y], dtype=np.float32
    )
    pre_grasp = np.array(
        [initial_xy[0], initial_xy[1], args.pre_grasp_z], dtype=np.float32
    )
    step_to(
        env,
        pre_grasp,
        OPEN_GRIPPER,
        max_steps=args.pre_grasp_step_budget,
        tol=0.008,
        render_cb=render_cb,
        label="pre_grasp_centered",
        max_xyz_step=0.01,
    )
    align_fingers_over_cube(
        env, args.pre_grasp_z, args, render_cb=render_cb, label="center_over_cube"
    )

    tcp = current_tcp(env)
    cube = to_numpy(env.unwrapped.cube.pose.p)[0].astype(np.float32)
    error_xy = cube[:2] - finger_center_xy(env)
    grasp = np.array(
        [tcp[0] + error_xy[0], tcp[1] + error_xy[1], args.grasp_z],
        dtype=np.float32,
    )
    step_to(
        env,
        grasp,
        OPEN_GRIPPER,
        max_steps=args.grasp_step_budget,
        tol=0.004,
        render_cb=render_cb,
        label="grasp_centered",
    )
    tcp = current_tcp(env)
    if tcp[2] > args.grasp_z + args.max_grasp_z_error:
        contacted, left, right, total = is_force_grasped(
            env, args.force_threshold
        )
        if not contacted:
            print(
                "force grasp unreachable: "
                f"requested_z={args.grasp_z:.3f} reached_z={tcp[2]:.3f}"
            )
            cube = to_numpy(env.unwrapped.cube.pose.p)[0].astype(np.float32)
            return False, None, args.grasp_z, cube[:2].copy()
        print(
            "grasp target height not reached, but bilateral contact exists; "
            f"continuing close (left={left:.2f}, right={right:.2f}, total={total:.2f})"
        )

    align_fingers_over_cube(
        env, args.grasp_z, args, render_cb=render_cb, label="center_at_grasp"
    )
    ok, _ = close_until_force(env, args, render_cb=render_cb)
    return ok, np.zeros(2, dtype=np.float32), args.grasp_z, current_tcp(env)[:2].copy()


def guarded_lift(env, target_xyz, args, render_cb=None):
    """Lift while monitoring force.

    Force loss aborts the transfer by default to avoid wasting steps. The current Piper gripper collision
    model is not always able to physically retain the cube even after contact is
    detected, so use --no-abort-on-force-loss only for contact-model debugging.
    """
    steps = 8 if args.slow_lift else 1
    start = current_tcp(env)
    ok = True
    info = None
    for i in range(steps):
        alpha = (i + 1) / steps
        waypoint = start * (1 - alpha) + np.asarray(target_xyz, dtype=np.float32) * alpha
        info = step_to(
            env,
            waypoint,
            active_gripper_target(args),
            max_steps=45,
            tol=0.008,
            render_cb=render_cb,
            label="force_lift",
        )
        grasped, left, right, total = is_force_grasped(env, args.force_threshold * 0.5)
        print(f"lift force: left={left:.4f} right={right:.4f} total={total:.4f}")
        if not grasped:
            ok = False
            if args.abort_on_force_loss:
                break
    return ok, info




def guarded_transfer(env, target_xyz, args, render_cb=None, label="force_transfer"):
    start = current_tcp(env)
    target_xyz = np.asarray(target_xyz, dtype=np.float32)
    steps = max(1, args.transfer_segments)
    ok = True
    info = None
    for i in range(steps):
        alpha = (i + 1) / steps
        waypoint = start * (1 - alpha) + target_xyz * alpha
        info = step_to(
            env,
            waypoint,
            active_gripper_target(args),
            max_steps=args.transfer_step_budget,
            tol=0.008,
            render_cb=render_cb,
            label=label,
        )
        grasped, left, right, total = is_force_grasped(env, args.force_threshold)
        print(f"transfer force: left={left:.4f} right={right:.4f} total={total:.4f}")
        if not grasped:
            ok = False
            if args.abort_on_force_loss:
                break
    return ok, info

def parse_grasp_offsets(args):
    if args.grasp_offsets:
        offsets = []
        for item in args.grasp_offsets.split(";"):
            if not item.strip():
                continue
            x, y = item.split(",")
            offsets.append(np.array([float(x), float(y)], dtype=np.float32))
        return offsets
    offsets = [
        np.array([args.grasp_offset_x, args.grasp_offset_y], dtype=np.float32),
        np.array([0.035, 0.010], dtype=np.float32),
        np.array([0.040, 0.000], dtype=np.float32),
        np.array([0.020, 0.000], dtype=np.float32),
    ]
    unique = []
    for item in offsets:
        if not any(np.linalg.norm(item - old) < 1e-6 for old in unique):
            unique.append(item)
    return unique


def parse_grasp_zs(args):
    if args.grasp_zs:
        return [float(item) for item in args.grasp_zs.split(",") if item.strip()]
    vals = []
    for v in [args.grasp_z, 0.006, 0.014]:
        if all(abs(v - old) > 1e-6 for old in vals):
            vals.append(v)
    return vals


def find_force_grasp(env, cube_xyz, args, render_cb=None):
    """Try a small set of grasp offsets/heights and keep the first force contact."""
    if not args.exhaustive_grasp_search:
        return try_centered_force_grasp(env, args, render_cb=render_cb)
    best = None
    tested = set()
    for xy_offset in parse_grasp_offsets(args):
        for grasp_z in parse_grasp_zs(args):
            live_cube_xyz = to_numpy(env.unwrapped.cube.pose.p)[0].astype(np.float32)
            key = (round(float(xy_offset[0]), 4), round(float(xy_offset[1]), 4), round(float(grasp_z), 4))
            if key in tested:
                continue
            tested.add(key)
            grasp_xy = live_cube_xyz[:2] + xy_offset
            pre_grasp = np.array([grasp_xy[0], grasp_xy[1], 0.16], dtype=np.float32)
            grasp = np.array([grasp_xy[0], grasp_xy[1], grasp_z], dtype=np.float32)
            step_to(env, pre_grasp, OPEN_GRIPPER, max_steps=80, tol=0.01, render_cb=render_cb, label="pre_grasp_scan")
            step_to(env, grasp, OPEN_GRIPPER, max_steps=60, tol=0.005, render_cb=render_cb, label="grasp_scan")
            ok, _ = close_until_force(env, args, render_cb=render_cb)
            left, right, total = contact_force_norms(env)
            print(
                f"candidate offset={xy_offset.round(3).tolist()} z={grasp_z:.3f} "
                f"force_total={total:.3f}"
            )
            if best is None or total > best[0]:
                best = (total, xy_offset.copy(), grasp_z, grasp_xy.copy())
            if ok:
                print(f"selected grasp offset={xy_offset.round(3).tolist()} z={grasp_z:.3f}")
                return True, xy_offset.copy(), grasp_z, grasp_xy.copy()
            hold(env, OPEN_GRIPPER, steps=8, render_cb=render_cb, label="open_scan")
    if best is not None:
        print(
            f"best failed candidate: force={best[0]:.3f} "
            f"offset={best[1].round(3).tolist()} z={best[2]:.3f}"
        )
    return False, None, None, None

def force_scripted_episode(
    env,
    rng,
    args,
    render_cb=None,
    reset_scene=True,
    randomize=True,
    task_state=None,
):
    if reset_scene:
        env.reset(seed=int(rng.integers(0, 1_000_000)))
    if randomize or task_state is None:
        cube_xyz, box_xy = randomize_force_task(env, rng, args.max_cube_x)
        task_state = (cube_xyz.copy(), box_xy.copy())
    else:
        cube_xyz, box_xy = task_state
        set_actor_xyz(env.unwrapped.cube, cube_xyz, device=env.unwrapped.device)
        move_blue_box(env, box_xy)

    e = env.unwrapped
    print(f"cube={cube_xyz[:2].round(3).tolist()} box={box_xy.round(3).tolist()}")

    attempt_status = "not_started"
    budget_steps = 0
    try:
        with StepBudget(env, args.max_attempt_steps) as budget:
            attempt_status = "running"
            grasp_ok, selected_offset, selected_z, grasp_xy = find_force_grasp(
                env, cube_xyz, args, render_cb=render_cb
            )
            if selected_offset is None:
                selected_offset = np.array([args.grasp_offset_x, args.grasp_offset_y], dtype=np.float32)
                grasp_xy = cube_xyz[:2] + selected_offset

            lift = np.array([grasp_xy[0], grasp_xy[1], 0.20], dtype=np.float32)
            lift_ok = False
            if grasp_ok:
                lift_ok, _ = guarded_lift(env, lift, args, render_cb=render_cb)

            if args.use_live_carried_offset:
                live_cube = to_numpy(e.cube.pose.p)[0].astype(np.float32)
                live_tcp = current_tcp(env)
                carried_xy_offset = live_cube[:2] - live_tcp[:2]
                offset_is_reasonable = (
                    np.all(np.isfinite(carried_xy_offset))
                    and -0.08 <= carried_xy_offset[0] <= 0.04
                    and -0.08 <= carried_xy_offset[1] <= 0.04
                )
                if not offset_is_reasonable:
                    print(f"carried offset fallback: measured={np.round(carried_xy_offset, 3).tolist()}")
                    carried_xy_offset = CARRIED_OBJ_XY_OFFSET_FROM_TCP
            else:
                carried_xy_offset = CARRIED_OBJ_XY_OFFSET_FROM_TCP
            place_tcp_xy = box_xy - carried_xy_offset
            pre_place = np.array([place_tcp_xy[0], place_tcp_xy[1], args.transport_z], dtype=np.float32)
            place = np.array([place_tcp_xy[0], place_tcp_xy[1], args.place_z], dtype=np.float32)
            retreat = np.array([place_tcp_xy[0], place_tcp_xy[1], args.transport_z], dtype=np.float32)

            if grasp_ok and (lift_ok or not args.abort_on_force_loss):
                guarded_transfer(env, pre_place, args, render_cb=render_cb, label="pre_place")
                if args.drop_from_transport:
                    hold(env, OPEN_GRIPPER, steps=45, render_cb=render_cb, label="drop_open")
                else:
                    step_to(env, place, CLOSED_GRIPPER, max_steps=90, render_cb=render_cb, label="place_down")
                    hold(env, OPEN_GRIPPER, steps=30, render_cb=render_cb, label="open")
                step_to(env, retreat, OPEN_GRIPPER, max_steps=80, render_cb=render_cb, label="retreat")
                hold(env, OPEN_GRIPPER, steps=45, render_cb=render_cb, label="settle")
            else:
                print("skip place: no reliable force grasp")
                hold(env, OPEN_GRIPPER, steps=20, render_cb=render_cb, label="open_failed_grasp")

            budget_steps = budget.steps
            attempt_status = "finished"
    except StepLimitExceeded as exc:
        attempt_status = "step_limit"
        budget_steps = args.max_attempt_steps
        print(f"attempt aborted: {exc}; opening gripper and returning Home")
        hold(env, OPEN_GRIPPER, steps=args.abort_open_steps, render_cb=render_cb, label="abort_open")

    info = e.evaluate()
    left, right, total = contact_force_norms(env)
    final_cube = to_numpy(e.cube.pose.p)[0]
    print(
        "success=", bool(to_numpy(info["success"])[0]),
        "in_box=", bool(to_numpy(info["is_cube_in_box"])[0]),
        "mani_grasped=", bool(to_numpy(info["is_grasped"])[0]),
        f"attempt_status={attempt_status}",
        f"attempt_steps={budget_steps}/{args.max_attempt_steps}",
        f"cube_final={np.round(final_cube, 3).tolist()}",
        f"force_total={total:.4f}",
        f"force_left={left:.4f}",
        f"force_right={right:.4f}",
    )
    go_home(env, render_cb=render_cb)
    return info, task_state

def draw_interactive_frame(env, view, label):
    frame = render_view(env, view)
    left, right, total = contact_force_norms(env)
    cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(
        frame,
        f"force L={left:.3f} R={right:.3f} sum={total:.3f}",
        (12, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
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
    cv2.namedWindow("step03_2_force_scripted_piper_pick_place", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("step03_2_force_scripted_piper_pick_place", args.width, args.height)
    delay_ms = max(1, int(1000 / args.fps))
    view = args.view
    status = "home"

    def render_cb(label):
        frame = draw_interactive_frame(env, view, label)
        cv2.imshow("step03_2_force_scripted_piper_pick_place", frame)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key in (27, ord("q")):
            raise KeyboardInterrupt
        time.sleep(0.001)

    env.reset(seed=int(rng.integers(0, 1_000_000)))
    task_state = randomize_force_task(env, rng, args.max_cube_x)
    go_home(env, render_cb=render_cb)

    while True:
        frame = draw_interactive_frame(env, view, status)
        cv2.imshow("step03_2_force_scripted_piper_pick_place", frame)
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
            status = "force grasp current"
            _, task_state = force_scripted_episode(
                env,
                rng,
                args,
                render_cb=render_cb,
                reset_scene=False,
                randomize=False,
                task_state=task_state,
            )
            status = "done - press r replay or n new"
        elif key == ord("r"):
            status = "force replay same"
            env.reset(seed=int(rng.integers(0, 1_000_000)))
            _, task_state = force_scripted_episode(
                env,
                rng,
                args,
                render_cb=render_cb,
                reset_scene=False,
                randomize=False,
                task_state=task_state,
            )
            status = "done - same positions"
        elif key == ord("n"):
            status = "force new random"
            env.reset(seed=int(rng.integers(0, 1_000_000)))
            _, task_state = force_scripted_episode(
                env,
                rng,
                args,
                render_cb=render_cb,
                reset_scene=False,
                randomize=True,
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
    parser.add_argument("--force-threshold", type=float, default=5.0)
    parser.add_argument("--contact-stable-steps", type=int, default=10)
    parser.add_argument("--exhaustive-grasp-search", action="store_true")
    parser.add_argument("--pre-grasp-z", type=float, default=0.16)
    parser.add_argument("--pre-grasp-step-budget", type=int, default=70)
    parser.add_argument("--grasp-step-budget", type=int, default=55)
    parser.add_argument("--centering-iterations", type=int, default=3)
    parser.add_argument("--centering-step-budget", type=int, default=20)
    parser.add_argument("--centering-tolerance", type=float, default=0.003)
    parser.add_argument("--max-grasp-z-error", type=float, default=0.040)
    parser.add_argument("--max-cube-x", type=float, default=0.025)
    parser.add_argument("--grasp-retries", type=int, default=0)
    parser.add_argument("--close-steps", type=int, default=20)
    parser.add_argument("--gripper-step-mm", type=float, default=2.0)
    parser.add_argument("--gripper-preload-mm", type=float, default=2.0)
    parser.add_argument("--retry-down-step", type=float, default=0.004)
    parser.add_argument("--grasp-z", type=float, default=0.006)
    parser.add_argument("--min-grasp-z", type=float, default=0.004)
    parser.add_argument("--slow-lift", action="store_true", default=True)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--max-attempt-steps", type=int, default=650)
    parser.add_argument("--abort-open-steps", type=int, default=20)
    parser.add_argument("--no-contact-tuning", action="store_true")
    parser.add_argument("--static-friction", type=float, default=5.0)
    parser.add_argument("--dynamic-friction", type=float, default=5.0)
    parser.add_argument("--restitution", type=float, default=0.0)
    parser.add_argument("--cube-mass", type=float, default=0.03)
    parser.add_argument("--transport-z", type=float, default=0.16)
    parser.add_argument("--place-z", type=float, default=0.075)
    parser.add_argument("--transfer-segments", type=int, default=10)
    parser.add_argument("--transfer-step-budget", type=int, default=35)
    parser.add_argument("--use-live-carried-offset", action="store_true", default=True)
    parser.add_argument("--fixed-carried-offset", action="store_true")
    parser.add_argument("--drop-from-transport", action="store_true")
    parser.add_argument("--abort-on-force-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grasp-offset-x", type=float, default=0.035)
    parser.add_argument("--grasp-offset-y", type=float, default=0.010)
    parser.add_argument("--grasp-offsets", default=None, help="semicolon-separated x,y offsets, e.g. 0.02,0;0.035,0.01")
    parser.add_argument("--grasp-zs", default=None, help="comma-separated grasp heights, e.g. 0.006,0.014,0.02")
    args = parser.parse_args()

    if args.grasp_offsets or args.grasp_zs:
        args.exhaustive_grasp_search = True
    if args.contact_stable_steps <= 0 or args.centering_iterations <= 0:
        parser.error("contact/centering step counts must be positive")
    if not -0.07 <= args.max_cube_x <= 0.055:
        parser.error("max-cube-x must be within [-0.07, 0.055]")

    if args.display is not None:
        os.environ["DISPLAY"] = args.display
    if args.fixed_carried_offset:
        args.use_live_carried_offset = False

    env = gym.make(
        "PiperPickCubeToBox-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
        num_envs=1,
        max_episode_steps=args.max_episode_steps,
        human_render_camera_configs={"width": 512, "height": 384},
    )

    apply_contact_tuning(env, args)

    delay_ms = max(1, int(1000 / args.fps))
    rng = np.random.default_rng(args.seed)
    try:
        if args.interactive:
            interactive_loop(env, rng, args)
            return

        if args.render:
            cv2.namedWindow("step03_2_force_scripted_piper_pick_place", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("step03_2_force_scripted_piper_pick_place", args.width, args.height)

        def render_cb(label):
            if not args.render:
                return
            frame = draw_interactive_frame(env, args.view, label)
            cv2.imshow("step03_2_force_scripted_piper_pick_place", frame)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (27, ord("q")):
                raise KeyboardInterrupt
            time.sleep(0.001)

        for ep in range(args.episodes):
            print(f"episode {ep}")
            force_scripted_episode(env, rng, args, render_cb=render_cb)
    finally:
        env.close()
        if args.render or args.interactive:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
