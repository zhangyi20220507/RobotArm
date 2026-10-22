"""Piper 机械臂/夹爪操作辅助：进入流模式、初始化、回零。

迁移自 teleop_piper_xr.py 的机械臂交互部分；依赖 numpy，
仅由 XR 运行时在执行期导入。
"""

from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np


def wait_for_flange_pose(robot, timeout: float = 3.0) -> np.ndarray:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        message = robot.get_flange_pose()
        if message is not None:
            pose = np.asarray(message.msg, dtype=float)
            if pose.shape == (6,) and np.all(np.isfinite(pose)):
                return pose
        time.sleep(0.02)
    raise RuntimeError("No valid Piper flange feedback received within 3 seconds")


def wait_for_joint_angles(robot, timeout: float = 3.0) -> np.ndarray:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        message = robot.get_joint_angles()
        if message is not None:
            joints = np.asarray(message.msg, dtype=float)
            if joints.shape == (6,) and np.all(np.isfinite(joints)):
                return joints
        time.sleep(0.02)
    raise RuntimeError("No valid Piper joint feedback received within 3 seconds")


def arm_status_text(robot) -> str:
    message = robot.get_arm_status()
    if message is None:
        return "unavailable"
    status = message.msg
    fields = (
        "ctrl_mode",
        "arm_status",
        "mode_feedback",
        "teach_status",
        "motion_status",
        "err_code",
    )
    return ", ".join(f"{name}={getattr(status, name, 'n/a')}" for name in fields)


def enable_arm(robot, timeout: float = 5.0) -> None:
    """Enable all motors and wait for low-speed driver feedback."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        robot.enable()
        states = robot.get_joints_enable_status_list()
        if len(states) == 6 and all(states):
            return
        time.sleep(0.05)
    raise RuntimeError(f"Piper enable was not acknowledged within {timeout:.1f} seconds")


def enter_joint_stream_mode(robot, joints: np.ndarray, control_mode: str) -> np.ndarray:
    """Enter J/JS while holding the measured pose, following the gamepad runtime's sequence."""
    hold = joints.tolist()
    robot.set_auto_set_motion_mode_enabled(False)
    robot.set_joint_limits_enabled(False)
    robot.move_j(hold)
    robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.J)
    robot.move_j(hold)
    robot.move_j(hold)
    enable_arm(robot)
    time.sleep(0.1)

    current = wait_for_joint_angles(robot)
    robot.move_j(current.tolist())
    if control_mode == "js":
        robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.JS)
        robot.move_js(current.tolist())
    else:
        robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.J)
        robot.move_j(current.tolist())
    return current


def initialize_gripper(robot, max_width: float, force: float):
    gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
    deadline = time.monotonic() + 3.0
    feedback = None
    while time.monotonic() < deadline:
        feedback = gripper.get_gripper_status()
        if feedback is not None and feedback.msg.mode == "width":
            break
        time.sleep(0.02)
    if feedback is None or feedback.msg.mode != "width":
        raise RuntimeError("No Piper gripper width feedback within 3 seconds")
    width = float(feedback.msg.value)
    if not math.isfinite(width) or not -0.002 <= width <= 0.152:
        raise RuntimeError(f"Invalid Piper gripper width feedback: {width!r}")
    command = float(np.clip(width, 0.0, max_width))
    gripper.move_gripper_m(command, force)
    return gripper, command


def gripper_feedback_width(gripper, fallback: float) -> float:
    status = gripper.get_gripper_status()
    if status is not None and status.msg.mode == "width":
        value = float(status.msg.value)
        if math.isfinite(value):
            return value
    return fallback


def update_force_limited_gripper(
    gripper,
    command_width: float,
    force_hold_width: float | None,
    trigger: float,
    max_width: float,
    max_step: float,
    command_force: float,
    force_threshold: float,
) -> tuple[float, float | None, str | None, float | None]:
    """Open with Trigger; otherwise close until contact force, then hold width.

    The driver's commanded force remains the final hardware-side force cap.  A
    positive feedback threshold adds a software latch at the measured contact
    width.  Pressing Trigger far enough to request at least 1 mm more opening
    releases that latch.
    """
    trigger = float(np.clip(trigger, 0.0, 1.0))
    desired_width = trigger * max_width
    feedback_width = None
    feedback_force = None
    status = gripper.get_gripper_status()
    if status is not None and status.msg.mode == "width":
        width = float(status.msg.value)
        force = abs(float(getattr(status.msg, "force", math.nan)))
        if math.isfinite(width) and math.isfinite(force):
            feedback_width = float(np.clip(width, 0.0, max_width))
            feedback_force = force

    event = None
    if force_hold_width is not None:
        # Ignore analog Trigger noise while holding a grasp.  A deliberate
        # opening request releases the force latch.
        if desired_width >= force_hold_width + 0.001:
            force_hold_width = None
            event = "released"
        else:
            desired_width = force_hold_width

    closing_on_object = (
        force_hold_width is None
        and force_threshold > 0.0
        and feedback_width is not None
        and feedback_force is not None
        and feedback_force >= force_threshold
        and desired_width < feedback_width - 0.0005
    )
    if closing_on_object:
        force_hold_width = feedback_width
        command_width = feedback_width
        event = "latched"
    else:
        command_width += float(
            np.clip(desired_width - command_width, -max_step, max_step)
        )
        command_width = float(np.clip(command_width, 0.0, max_width))

    gripper.move_gripper_m(command_width, command_force)
    return command_width, force_hold_width, event, feedback_force


def home_arm_and_gripper(
    robot,
    gripper,
    control_mode: str,
    gripper_width: float,
    gripper_force: float,
    gripper_speed: float,
    stop_check: Callable[[], bool],
    timeout: float = 60.0,
) -> tuple[np.ndarray, float]:
    """Use the gamepad runtime's planned MOVE_J zeroing, then restore streaming mode."""
    target = np.zeros(6, dtype=float)
    robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.J)
    robot.move_j(target.tolist())
    robot.move_j(target.tolist())
    started = time.monotonic()
    last_tick = started
    last_print = 0.0
    while time.monotonic() - started < timeout:
        now = time.monotonic()
        if stop_check():
            robot.electronic_emergency_stop()
            raise RuntimeError("Emergency stop requested while homing")
        dt = max(0.0, now - last_tick)
        last_tick = now
        gripper_width = max(0.0, gripper_width - gripper_speed * dt)
        gripper.move_gripper_m(gripper_width, gripper_force)
        joints = wait_for_joint_angles(robot)
        feedback_width = gripper_feedback_width(gripper, gripper_width)
        status = robot.get_arm_status()
        motion_status = None if status is None else getattr(status.msg.motion_status, "value", status.msg.motion_status)
        motion_done = motion_status is not None and int(motion_status) == 0
        if now - last_print >= 0.25:
            print(
                "HOMING:",
                f"joint_error={math.degrees(float(np.max(np.abs(joints)))):.2f} deg",
                f"gripper={feedback_width * 1000.0:.1f} mm",
            )
            last_print = now
        if (
            now - started >= 0.5
            and motion_done
            and float(np.max(np.abs(joints))) <= math.radians(3.0)
            and feedback_width <= 0.002
        ):
            if control_mode == "js":
                robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.JS)
                robot.move_js(joints.tolist())
            else:
                robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.J)
                robot.move_j(joints.tolist())
            print("Homing complete; joint streaming restored.")
            return joints, 0.0
        time.sleep(0.02)
    robot.electronic_emergency_stop()
    raise TimeoutError(f"Homing did not finish within {timeout:.0f} seconds; E-stop sent")


def hold_arm_on_exit(robot, control_mode: str) -> None:
    """Replace the last target without disabling motors or latching E-stop."""
    joints = wait_for_joint_angles(robot, timeout=0.3)
    # Also restores streaming if cancellation interrupted MOVE_J homing.
    mode = robot.OPTIONS.MOTION_MODE.JS if control_mode == "js" else robot.OPTIONS.MOTION_MODE.J
    robot.set_motion_mode(mode)
    if control_mode == "js":
        robot.move_js(joints.tolist())
    else:
        robot.move_j(joints.tolist())
