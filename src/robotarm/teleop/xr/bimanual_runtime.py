"""PICO 双手柄遥操作两台 AgileX Piper。

左手柄控制 ``left-channel``，右手柄控制 ``right-channel``。默认只预览；
只有显式传入 ``--live`` 后才会使能机械臂和发送运动命令。

按键：默认使能后直接跟随，Trigger 分别控制夹爪；长按 Y/B 分别使能或恢复
左/右臂；长按 X/A 分别回零；按下任一摇杆对两臂发送电子急停。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import signal
import sys
import time

import numpy as np

from ...config import defaults
from .arm_ops import (
    arm_status_text,
    enter_joint_stream_mode,
    hold_arm_on_exit,
    home_arm_and_gripper,
    initialize_gripper,
    update_force_limited_gripper,
    wait_for_flange_pose,
    wait_for_joint_angles,
)
from .control import PoseJumpGuard, Clutch, joint_limit_diagnostics, limit_joint_step
from .geometry import (
    Limits,
    TranslationTracker,
    build_target,
    clamp_rotation_step,
    clamp_vector_step,
    valid_xr_pose,
)
from .piper_ik import PIPER_JOINT_MAX, PIPER_JOINT_MIN, PiperIK, joint_limit_violations
from .runtime import parse_xyz


def make_parser() -> argparse.ArgumentParser:
    from .control import add_input_options
    cfg = defaults("xr", "piper-bimanual-xr")
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_options(parser, cfg.get("engagement-mode", "direct"))
    parser.add_argument(
        "--arms", type=int, choices=(2,), default=2,
        help="two arms; use piper-xr --arms 1 for single-arm control",
    )
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--left-channel", default=cfg["left-channel"])
    parser.add_argument("--right-channel", default=cfg["right-channel"])
    parser.add_argument("--xr-bridge", action="store_true")
    parser.add_argument("--xr-bridge-endpoint", default=cfg["xr-bridge-endpoint"])
    parser.add_argument("--firmware", choices=("auto", "default", "v183", "v188", "v189"), default=cfg["firmware"])
    parser.add_argument("--model", choices=("piper",), default=cfg["model"])
    parser.add_argument("--rate", type=float, default=cfg["rate"])
    parser.add_argument("--scale", type=float, default=cfg["scale"])
    parser.add_argument("--speed-percent", type=int, default=cfg["speed-percent"])
    parser.add_argument("--joint-control-mode", choices=("js", "j"), default=cfg["joint-control-mode"])
    parser.add_argument("--translation-speed", type=float, default=cfg["translation-speed"])
    parser.add_argument("--joint-speed", type=float, default=cfg["joint-speed"])
    parser.add_argument("--max-target-lead", type=float, default=cfg["max-target-lead"])
    parser.add_argument("--orientation", action="store_true")
    parser.add_argument("--orientation-weight", type=float, default=cfg["orientation-weight"])
    parser.add_argument(
        "--posture-anchor", choices=("engage", "seed"), default=cfg.get("posture-anchor", "engage"),
        help="engage: hold the posture captured at engagement; seed: legacy follow-the-seed",
    )
    parser.add_argument("--left-xr-yaw-deg", type=float, default=cfg["left-xr-yaw-deg"])
    parser.add_argument("--right-xr-yaw-deg", type=float, default=cfg["right-xr-yaw-deg"])
    parser.add_argument(
        "--left-xr-mirror", choices=("none", "lr", "fb"), default=cfg.get("left-xr-mirror", "none"),
        help="mirror one horizontal axis when only left-right (lr) or forward-back (fb) is reversed",
    )
    parser.add_argument(
        "--right-xr-mirror", choices=("none", "lr", "fb"), default=cfg.get("right-xr-mirror", "none"),
        help="mirror one horizontal axis when only left-right (lr) or forward-back (fb) is reversed",
    )
    parser.add_argument("--grip-threshold", type=float, default=cfg["grip-threshold"])
    parser.add_argument("--grip-release-threshold", type=float, default=cfg["grip-release-threshold"])
    parser.add_argument("--gripper-max-mm", type=float, default=cfg["gripper-max-mm"])
    parser.add_argument("--gripper-speed-mm-s", type=float, default=cfg["gripper-speed-mm-s"])
    parser.add_argument("--gripper-force", type=float, default=cfg["gripper-force"])
    parser.add_argument(
        "--gripper-force-hold-threshold",
        type=float,
        default=cfg["gripper-force-hold-threshold"],
        help="hold measured width when closing force reaches this value in N; 0 disables",
    )
    parser.add_argument("--button-hold-seconds", type=float, default=cfg["button-hold-seconds"])
    parser.add_argument("--left-xyz-min", type=parse_xyz, default=tuple(cfg["left-xyz-min"]))
    parser.add_argument("--left-xyz-max", type=parse_xyz, default=tuple(cfg["left-xyz-max"]))
    parser.add_argument("--right-xyz-min", type=parse_xyz, default=tuple(cfg["right-xyz-min"]))
    parser.add_argument("--right-xyz-max", type=parse_xyz, default=tuple(cfg["right-xyz-max"]))
    parser.add_argument("--left-base-origin", type=parse_xyz, default=tuple(cfg["left-base-origin"]), help="left base origin in a shared frame, meters")
    parser.add_argument("--right-base-origin", type=parse_xyz, default=tuple(cfg["right-base-origin"]), help="right base origin in a shared frame, meters")
    parser.add_argument("--min-flange-separation", type=float, default=cfg["min-flange-separation"], help="hold both arms below this flange distance; 0 disables the check")
    parser.add_argument("--max-relative-translation", type=float, default=cfg["max-relative-translation"])
    parser.add_argument("--max-step-translation", type=float, default=cfg["max-step-translation"])
    parser.add_argument("--max-step-rotation", type=float, default=math.radians(cfg["max-step-rotation-deg"]))
    parser.add_argument("--max-step-joint", type=float, default=cfg["max-step-joint"])
    parser.add_argument("--max-joint-following-error", type=float, default=cfg["max-joint-following-error"])
    parser.add_argument("--joint-limit-tolerance", type=float, default=math.radians(cfg["joint-limit-tolerance-deg"]))
    parser.add_argument("--stale-timeout", type=float, default=cfg["stale-timeout"])
    parser.add_argument(
        "--homing-stale-timeout",
        type=float,
        default=cfg["homing-stale-timeout"],
        help="XR watchdog timeout during blocking MOVE_J homing",
    )
    parser.add_argument("--xr-wait-timeout", type=float, default=cfg["xr-wait-timeout"])
    return parser


def validate_args(args) -> None:
    from .control import validate_input_options
    validate_input_options(args)
    if args.left_channel == args.right_channel:
        raise ValueError("left-channel and right-channel must be different")
    finite = (
        args.rate, args.scale, args.translation_speed, args.joint_speed,
        args.max_target_lead, args.orientation_weight, args.left_xr_yaw_deg,
        args.right_xr_yaw_deg, args.stale_timeout,
        args.homing_stale_timeout, args.xr_wait_timeout,
    )
    if not all(math.isfinite(value) for value in finite):
        raise ValueError("bimanual XR numeric options must be finite")
    if not 0 < args.rate <= 200 or args.scale <= 0:
        raise ValueError("rate must be in (0, 200] and scale must be positive")
    if not 0 < args.translation_speed <= 0.3 or not 0 < args.joint_speed <= 1.0:
        raise ValueError("translation-speed must be in (0, 0.3] and joint-speed in (0, 1]")
    if not 0 < args.max_target_lead <= 0.05:
        raise ValueError("max-target-lead must be in (0, 0.05]")
    if not 0 <= args.orientation_weight <= 1:
        raise ValueError("orientation-weight must be in [0, 1]")
    for side in ("left", "right"):
        if getattr(args, f"{side}_xr_mirror") not in ("none", "lr", "fb"):
            raise ValueError(f"--{side}-xr-mirror must be one of none, lr, fb")
    if args.posture_anchor not in ("engage", "seed"):
        raise ValueError("--posture-anchor must be engage or seed")
    if not 0 <= args.grip_release_threshold < args.grip_threshold <= 1:
        raise ValueError("grip thresholds must satisfy 0 <= release < engage <= 1")
    if not 0 <= args.speed_percent <= 100:
        raise ValueError("speed-percent must be in [0, 100]")
    if not 0.3 <= args.button_hold_seconds <= 5:
        raise ValueError("button-hold-seconds must be in [0.3, 5]")
    if not 0 < args.gripper_max_mm <= 100 or not 0 < args.gripper_speed_mm_s <= 100:
        raise ValueError("invalid gripper width or speed")
    if not 0 < args.gripper_force <= 5:
        raise ValueError("gripper-force must be in (0, 5]")
    if not 0 <= args.gripper_force_hold_threshold <= args.gripper_force:
        raise ValueError(
            "gripper-force-hold-threshold must be in [0, gripper-force]"
        )
    if min(args.max_relative_translation, args.max_step_translation, args.max_step_rotation,
           args.max_step_joint, args.max_joint_following_error, args.stale_timeout,
           args.homing_stale_timeout, args.xr_wait_timeout) <= 0:
        raise ValueError("motion limits and timeouts must be positive")
    if args.homing_stale_timeout < args.stale_timeout:
        raise ValueError("homing-stale-timeout must not be shorter than stale-timeout")
    if not math.isfinite(args.min_flange_separation) or args.min_flange_separation < 0:
        raise ValueError("min-flange-separation must be finite and non-negative")
    if not 0 <= args.joint_limit_tolerance <= math.radians(5):
        raise ValueError("joint-limit-tolerance must be between 0 and 5 degrees")
    for name in ("left", "right"):
        minimum = np.asarray(getattr(args, f"{name}_xyz_min"), dtype=float)
        maximum = np.asarray(getattr(args, f"{name}_xyz_max"), dtype=float)
        if np.any(minimum >= maximum):
            raise ValueError(f"every {name} workspace minimum must be below its maximum")


@dataclass
class LongPress:
    pressed_at: float | None = None
    fired: bool = False

    def update(self, pressed: bool, now: float, duration: float) -> bool:
        if not pressed:
            self.pressed_at = None
            self.fired = False
            return False
        if self.pressed_at is None:
            self.pressed_at = now
        if not self.fired and now - self.pressed_at >= duration:
            self.fired = True
            return True
        return False


def home_grip_released(state: dict, hand: str, release_threshold: float) -> bool:
    """Return whether the hand controlling the arm being homed is unclutched."""
    if hand not in ("left", "right"):
        raise ValueError(f"unknown XR hand: {hand}")
    return float(state[f"{hand}_grip"]) <= release_threshold


def simultaneous_home_pressed(state: dict) -> bool:
    """Two-arm homing is intentionally rejected without full collision checking."""
    return bool(state["X_button"] and state["A_button"])


def gripper_text(side: "ArmSide") -> str:
    """Commanded opening plus the software force-hold latch, for the 1 Hz status line."""
    hold = (
        "" if side.gripper_force_hold_width is None
        else f" hold={side.gripper_force_hold_width * 1000.0:.1f}mm"
    )
    return f"gripper={side.gripper_command * 1000.0:.1f}mm{hold}"


def ik_diagnostic_text(side: "ArmSide") -> str:
    diagnostics = side.last_ik_diagnostics
    if diagnostics is None:
        return "ik=idle"
    sources = ",".join(side.last_limit_sources) or "NONE"
    return (
        f"ik={diagnostics.solve_ms:.2f}ms "
        f"res={diagnostics.position_error_m * 1000.0:.2f}mm/"
        f"{math.degrees(diagnostics.orientation_error_rad):.2f}deg "
        f"cond_xyz={diagnostics.position_condition:.1f} "
        f"cond_pose={diagnostics.pose_condition:.1f} "
        f"shrink={diagnostics.target_scale:.3f} "
        f"cart_ratio={side.last_cartesian_limit_ratio:.2f} "
        f"joint_ratio={side.last_joint_limit_ratio:.2f} limits={sources}"
    )


@dataclass
class ArmSide:
    name: str
    hand: str
    channel: str
    robot: object
    ik: PiperIK
    limits: Limits
    yaw_degrees: float
    joints: np.ndarray
    clutch: Clutch
    mirror: str = "none"
    gripper: object | None = None
    gripper_command: float = 0.0
    gripper_force_hold_width: float | None = None
    enabled: bool = False
    ready: bool = False
    emergency_stopped: bool = False
    xr_reference: np.ndarray | None = None
    flange_reference: np.ndarray | None = None
    tracker: TranslationTracker | None = None
    last_cartesian_target: np.ndarray | None = None
    last_joint_command: np.ndarray | None = None
    next_gripper_command: float = 0.0
    last_limit_notice: float = 0.0
    last_ik_diagnostics: object | None = None
    last_cartesian_limit_ratio: float = 1.0
    last_joint_limit_ratio: float = 1.0
    last_limit_sources: tuple[str, ...] = ()

    def reset_clutch(self) -> None:
        self.xr_reference = None
        self.flange_reference = None
        self.tracker = None
        self.clutch.blocked = True
        self.ik.set_posture_anchor(None)


def snapshot_xr(xrt, bridged: bool) -> dict:
    if bridged:
        return xrt.snapshot()
    return {
        "timestamp_ns": xrt.get_time_stamp_ns(),
        "left_controller_pose": xrt.get_left_controller_pose(),
        "left_grip": xrt.get_left_grip(),
        "left_trigger": xrt.get_left_trigger(),
        "left_axis_click": xrt.get_left_axis_click(),
        "right_controller_pose": xrt.get_right_controller_pose(),
        "right_grip": xrt.get_right_grip(),
        "right_trigger": xrt.get_right_trigger(),
        "right_axis_click": xrt.get_right_axis_click(),
        "X_button": xrt.get_X_button(),
        "Y_button": xrt.get_Y_button(),
        "A_button": xrt.get_A_button(),
        "B_button": xrt.get_B_button(),
    }


def wait_for_bimanual_xr(xrt, bridged: bool, timeout: float, stop_requested) -> None:
    print("Waiting for fresh left and right controller poses; Y/B enable is not active yet...")
    deadline = time.monotonic() + timeout
    previous = None
    next_notice = 0.0
    detail = "no XR samples"
    while time.monotonic() < deadline:
        if stop_requested():
            raise InterruptedError("stopped while waiting for XR data")
        state = snapshot_xr(xrt, bridged)
        stamp = state.get("timestamp_ns", 0)
        poses = {}
        for hand in ("left", "right"):
            value = state.get(f"{hand}_controller_pose")
            try:
                valid = value is not None and valid_xr_pose(np.asarray(value, dtype=float))
            except (TypeError, ValueError):
                valid = False
            poses[hand] = "missing" if value is None else "valid" if valid else "invalid"
        changing = previous not in (None, 0) and stamp not in (None, 0, previous)
        detail = (
            f"timestamp={stamp} ({'changing' if changing else 'not changing'}), "
            f"left_pose={poses['left']}, right_pose={poses['right']}"
        )
        if changing and poses["left"] == poses["right"] == "valid":
            print("Both XR controllers are ready.")
            return
        if time.monotonic() >= next_notice:
            print("XR waiting: " + detail, flush=True)
            next_notice = time.monotonic() + 2.0
        previous = stamp
        time.sleep(0.05)
    raise RuntimeError(
        f"No valid two-controller XR data within {timeout:.0f} seconds. "
        f"Last XR state: {detail}. Start/restart xr-bridge, enable Send on the headset, "
        "and wake both controllers. Hold Y/B only after both controllers are ready."
    )


def _hold(side: ArmSide, args) -> None:
    if not args.live or not side.enabled or side.emergency_stopped:
        return
    joints = wait_for_joint_angles(side.robot, timeout=0.3)
    if args.joint_control_mode == "js":
        side.robot.move_js(joints.tolist())
    else:
        side.robot.move_j(joints.tolist())


def pause_on_input_jump(sides, state, guards, args):
    jumps = [f"{side.name}: {reason}" for side in sides
             if (reason := guards[side.hand].update(state[f"{side.hand}_controller_pose"]))]
    if not jumps:
        return False
    print("INPUT PAUSED: " + "; ".join(jumps) + "; hold Y/B to resume.")
    for side in sides:
        _hold(side, args)
        side.reset_clutch()
        side.clutch.pause()
    return True


def _estop_all(sides: list[ArmSide]) -> None:
    for side in sides:
        if side.enabled and not side.emergency_stopped:
            try:
                side.robot.electronic_emergency_stop()
            finally:
                side.emergency_stopped = True


def _enable(side: ArmSide, args) -> None:
    if side.ready:
        side.reset_clutch()
        side.clutch.resume()
        print(f"{side.name} control resumed; capturing a new hand reference")
        return
    current = wait_for_joint_angles(side.robot)
    violations = joint_limit_violations(current, args.joint_limit_tolerance)
    if violations:
        raise RuntimeError(f"{side.name} cannot enable: " + "; ".join(violations))
    # Mark before the first hardware write so failure cleanup E-stops this arm.
    side.enabled = True
    side.ready = False
    side.robot.set_speed_percent(args.speed_percent)
    current = enter_joint_stream_mode(side.robot, current, args.joint_control_mode)
    states = side.robot.get_joints_enable_status_list()
    if len(states) != 6 or not all(states):
        raise RuntimeError(f"{side.name} enable failed: {states}")
    side.gripper, side.gripper_command = initialize_gripper(
        side.robot, args.gripper_max_mm / 1000.0, args.gripper_force
    )
    side.gripper_force_hold_width = None
    side.joints = current
    side.last_joint_command = current.copy()
    side.last_cartesian_target = side.ik.flange_pose(current)
    side.ready = True
    side.reset_clutch()
    if args.engagement_mode == "direct":
        side.clutch.resume()
    print(f"{side.name} arm enabled on {side.channel}: {states}")


def _home(side: ArmSide, other: ArmSide, args, xrt, stop_requested) -> None:
    if not side.ready or side.gripper is None:
        print(f"{side.name} arm is not enabled; enable it first")
        return
    # Freeze the other arm and invalidate both clutch references before the
    # blocking homing operation.  A held Grip must not resume from an old
    # controller reference when homing finishes.
    _hold(other, args)
    other.reset_clutch()
    side.reset_clutch()
    last_stamp = None
    last_change = time.monotonic()
    abort_reason = None

    def stop_check() -> bool:
        nonlocal last_stamp, last_change, abort_reason
        state = snapshot_xr(xrt, args.xr_bridge)
        now = time.monotonic()
        stamp = state["timestamp_ns"]
        if stamp not in (None, 0, last_stamp):
            last_stamp = stamp
            last_change = now
        if stop_requested():
            abort_reason = "program stop requested"
        elif state["left_axis_click"] or state["right_axis_click"]:
            abort_reason = "XR stick emergency stop pressed"
        elif stamp in (None, 0):
            abort_reason = "XR timestamp is missing"
        elif now - last_change > args.homing_stale_timeout:
            abort_reason = (
                f"XR feedback stale for {now - last_change:.3f}s "
                f"(homing limit {args.homing_stale_timeout:.3f}s)"
            )
        return abort_reason is not None

    print(f"Homing {side.name} arm; the other arm is holding position...")
    try:
        joints, width = home_arm_and_gripper(
            side.robot, side.gripper, args.joint_control_mode,
            side.gripper_command, args.gripper_force,
            args.gripper_speed_mm_s / 1000.0, stop_check,
        )
    except RuntimeError as exc:
        if abort_reason is not None and str(exc) == "Emergency stop requested while homing":
            raise RuntimeError(f"{side.name} homing aborted: {abort_reason}") from exc
        raise
    side.joints = joints
    side.gripper_command = width
    side.gripper_force_hold_width = None
    side.last_joint_command = joints.copy()
    side.last_cartesian_target = side.ik.flange_pose(joints)


def _compute_arm(side: ArmSide, state: dict, fresh: bool, dt: float, now: float, args):
    pose = np.asarray(state[f"{side.hand}_controller_pose"], dtype=float)
    both_valid = valid_xr_pose(np.asarray(state["left_controller_pose"], dtype=float)) and valid_xr_pose(
        np.asarray(state["right_controller_pose"], dtype=float)
    )
    grip = float(state[f"{side.hand}_grip"])
    trigger = float(np.clip(state[f"{side.hand}_trigger"], 0.0, 1.0))
    active = side.clutch.update(grip, fresh and both_valid)

    if active and args.live and side.ready and side.gripper is not None and now >= side.next_gripper_command:
        maximum = args.gripper_speed_mm_s / 1000.0 * 0.05
        (
            side.gripper_command,
            side.gripper_force_hold_width,
            gripper_event,
            feedback_force,
        ) = update_force_limited_gripper(
            side.gripper,
            side.gripper_command,
            side.gripper_force_hold_width,
            trigger,
            args.gripper_max_mm / 1000.0,
            maximum,
            args.gripper_force,
            args.gripper_force_hold_threshold,
        )
        if gripper_event == "latched":
            print(
                f"{side.name} gripper force hold: "
                f"{feedback_force:.3f} N at {side.gripper_command * 1000.0:.1f} mm"
            )
        elif gripper_event == "released":
            print(f"{side.name} gripper force hold released by Trigger")
        side.next_gripper_command = now + 0.05

    if not active or (args.live and not side.ready):
        if side.xr_reference is not None:
            _hold(side, args)
        side.xr_reference = None
        side.flange_reference = None
        side.tracker = None
        return None

    current = wait_for_joint_angles(side.robot)
    feedback_pose = side.ik.flange_pose(current)
    if side.xr_reference is None:
        side.xr_reference = pose.copy()
        side.flange_reference = feedback_pose.copy()
        side.ik.set_posture_anchor(current if args.posture_anchor == "engage" else None)
        side.tracker = TranslationTracker(
            pose[:3], feedback_pose[:3], args.scale, side.limits,
            args.max_target_lead, side.yaw_degrees, side.mirror,
        )
        side.last_joint_command = current.copy()
        side.last_cartesian_target = feedback_pose.copy()
        outside = not np.all(
            (feedback_pose[:3] >= side.limits.xyz_min)
            & (feedback_pose[:3] <= side.limits.xyz_max)
        )
        print(
            f"{side.name} ENGAGED",
            "xyz=", np.array2string(feedback_pose[:3], precision=4),
            "workspace=", np.array2string(side.limits.xyz_min, precision=3),
            "..", np.array2string(side.limits.xyz_max, precision=3),
            "(outside; inward recovery only)" if outside else "",
        )

    requested = build_target(
        side.xr_reference, pose, side.flange_reference, args.scale,
        args.orientation, side.limits, yaw_degrees=side.yaw_degrees,
        mirror=side.mirror,
    )
    requested[:3], reasons = side.tracker.update(pose[:3], feedback_pose[:3])
    if reasons and now - side.last_limit_notice >= 1.0:
        print(f"{side.name}: " + "; ".join(reasons))
        side.last_limit_notice = now
    cartesian = requested.copy()
    requested_step = float(
        np.linalg.norm(requested[:3] - side.last_cartesian_target[:3])
    )
    cartesian[:3] = clamp_vector_step(
        side.last_cartesian_target[:3], requested[:3],
        min(side.limits.max_step_translation, args.translation_speed * dt),
    )
    cartesian[3:] = clamp_rotation_step(
        side.last_cartesian_target[3:], requested[3:], side.limits.max_step_rotation
    )
    cartesian_step = float(
        np.linalg.norm(cartesian[:3] - side.last_cartesian_target[:3])
    )
    cartesian_ratio = (
        1.0 if requested_step <= 1.0e-12
        else min(1.0, cartesian_step / requested_step)
    )
    seed = side.last_joint_command.copy()
    ik_result = side.ik.solve_adaptive(seed, cartesian)
    solution = ik_result.joints
    command = limit_joint_step(
        seed, solution, current,
        min(args.max_step_joint, args.joint_speed * dt),
        args.max_joint_following_error, PIPER_JOINT_MIN, PIPER_JOINT_MAX,
    )
    joint_ratio, joint_sources = joint_limit_diagnostics(
        seed, solution, current, command,
        min(args.max_step_joint, args.joint_speed * dt),
        args.max_joint_following_error, PIPER_JOINT_MIN, PIPER_JOINT_MAX,
    )
    sources = [reason.split(":", 1)[0] for reason in reasons]
    if cartesian_ratio < 1.0 - 1.0e-6:
        sources.append("CART_STEP")
    if ik_result.diagnostics.target_scale < 1.0:
        sources.append("IK_SHRINK")
    sources.extend(joint_sources)
    side.last_ik_diagnostics = ik_result.diagnostics
    side.last_cartesian_limit_ratio = cartesian_ratio
    side.last_joint_limit_ratio = joint_ratio
    side.last_limit_sources = tuple(dict.fromkeys(sources))
    side.last_joint_command = command
    side.last_cartesian_target = side.ik.flange_pose(command)
    side.joints = current
    return command


def main(argv: list[str] | None = None) -> int:
    from ...core.robot.piper import _load_sdk

    args = make_parser().parse_args(argv)
    validate_args(args)
    create_config, factory, ArmModel, PiperFW = _load_sdk()
    firmware_values = {
        "default": PiperFW.DEFAULT, "v183": PiperFW.V183,
        "v188": PiperFW.V188, "v189": PiperFW.V189,
    }

    def selected_firmware(channel: str):
        if args.firmware != "auto":
            return firmware_values[args.firmware]
        probe_cfg = create_config(
            robot=ArmModel.PIPER, firmeware_version=PiperFW.DEFAULT,
            interface="socketcan", channel=channel,
        )
        probe = factory.create_arm(probe_cfg)
        probe.connect()
        try:
            deadline = time.monotonic() + 5
            info = None
            while time.monotonic() < deadline and info is None:
                info = probe.get_firmware()
                if info is None:
                    time.sleep(0.1)
            if info is None:
                raise RuntimeError(f"could not detect Piper firmware on {channel}")
            version = info["software_version"]
        finally:
            probe.disconnect()
        if version >= "S-V1.8-9":
            result = PiperFW.V189
        elif version >= "S-V1.8-8":
            result = PiperFW.V188
        elif version >= "S-V1.8-3":
            result = PiperFW.V183
        else:
            result = PiperFW.DEFAULT
        print(f"{channel}: detected {version}; using {result}")
        return result

    if args.xr_bridge:
        from .bridge import XrBridgeClient
        xrt = XrBridgeClient(args.xr_bridge_endpoint)
        print(f"XR source: persistent bridge at {args.xr_bridge_endpoint}")
    else:
        import xrobotoolkit_sdk as xrt
        print("XR source: direct SDK connection")

    stop = False

    def request_stop(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    sides: list[ArmSide] = []
    connected_robots: list[object] = []
    xrt.init()
    try:
        # Validate the XR source before probing or connecting either arm.
        wait_for_bimanual_xr(xrt, args.xr_bridge, args.xr_wait_timeout, lambda: stop)
        for name, hand, channel, minimum, maximum, yaw, mirror in (
            ("left", "left", args.left_channel, args.left_xyz_min, args.left_xyz_max,
             args.left_xr_yaw_deg, args.left_xr_mirror),
            ("right", "right", args.right_channel, args.right_xyz_min, args.right_xyz_max,
             args.right_xr_yaw_deg, args.right_xr_mirror),
        ):
            config = create_config(
                robot=ArmModel.PIPER,
                firmeware_version=selected_firmware(channel),
                interface="socketcan", channel=channel,
            )
            robot = factory.create_arm(config)
            robot.connect()
            connected_robots.append(robot)
            wait_for_flange_pose(robot)
            joints = wait_for_joint_angles(robot)
            violations = joint_limit_violations(joints, args.joint_limit_tolerance)
            if violations:
                raise RuntimeError(f"{name} arm outside joint limits: " + "; ".join(violations))
            ik = PiperIK(1.0 / args.rate, args.orientation, args.orientation_weight)
            limits = Limits(
                np.asarray(minimum, dtype=float), np.asarray(maximum, dtype=float),
                args.max_relative_translation, args.max_step_translation,
                args.max_step_rotation,
            )
            side = ArmSide(
                name, hand, channel, robot, ik, limits, yaw, joints,
                Clutch(args.grip_threshold, args.grip_release_threshold, mode=args.engagement_mode, blocked=args.live),
                mirror=mirror,
                last_cartesian_target=ik.flange_pose(joints),
                last_joint_command=joints.copy(),
            )
            sides.append(side)
            print(f"{name} arm {channel} joints:", np.array2string(joints, precision=4))

        mode = "LIVE" if args.live else "DRY-RUN"
        print(f"Mode: {mode}. Left controller -> {args.left_channel}; right controller -> {args.right_channel}")
        print(f"Engagement mode: {args.engagement_mode}; jumps pause BOTH arms; hold Y/B to resume.")
        print("Hold Y/B to enable left/right; X/A to home; either stick click E-stops both.")
        if args.live:
            print("Clear both workspaces and keep the physical E-stop within reach.")
            if args.min_flange_separation <= 0:
                print("WARNING: inter-arm flange-distance guard is disabled; tighten per-arm workspaces before moving.")
        left, right = sides
        presses = {name: LongPress() for name in ("X", "Y", "A", "B")}
        last_stamp = None
        last_change = time.monotonic()
        previous_tick = time.monotonic() - 1.0 / args.rate
        last_print = 0.0
        guards = {side.hand: PoseJumpGuard(args.max_hand_jump_m, args.max_hand_jump_deg) for side in sides}
        period = 1.0 / args.rate

        while not stop:
            started = time.monotonic()
            dt = min(period, max(0.0, started - previous_tick))
            previous_tick = started
            state = snapshot_xr(xrt, args.xr_bridge)
            stamp = state["timestamp_ns"]
            if stamp != last_stamp:
                last_stamp = stamp
                last_change = started
            fresh = stamp not in (None, 0) and started - last_change <= args.stale_timeout

            if state["left_axis_click"] or state["right_axis_click"]:
                print("XR stick emergency stop requested; stopping both arms.")
                if args.live:
                    _estop_all(sides)
                break

            if not fresh:
                for side in sides:
                    if side.xr_reference is not None:
                        _hold(side, args)
                    side.reset_clutch()
                time.sleep(period)
                continue

            if pause_on_input_jump(sides, state, guards, args):
                time.sleep(period)
                continue

            fired = {
                name: presses[name].update(bool(state[f"{name}_button"]), started, args.button_hold_seconds)
                for name in presses
            }
            hands_released = args.engagement_mode == "direct" or (
                float(state["left_grip"]) <= args.grip_release_threshold
                and float(state["right_grip"]) <= args.grip_release_threshold
            )
            if fired["Y"]:
                if not hands_released:
                    print("Release both Grip buttons before enabling the left arm.")
                elif args.live:
                    _enable(left, args)
                else:
                    print("DRY-RUN: left enable skipped")
            if fired["B"]:
                if not hands_released:
                    print("Release both Grip buttons before enabling the right arm.")
                elif args.live:
                    _enable(right, args)
                else:
                    print("DRY-RUN: right enable skipped")
            if simultaneous_home_pressed(state) and (fired["X"] or fired["A"]):
                # Suppress the still-held partner button too.  Both buttons must
                # be released before either LongPress can arm again.
                presses["X"].fired = True
                presses["A"].fired = True
                print(
                    "X+A simultaneous home blocked: home one arm at a time "
                    "because full inter-arm collision checking is not enabled."
                )
            else:
                if fired["X"] and not home_grip_released(state, "left", args.grip_release_threshold):
                    print("Release LEFT GRIP before homing the left arm.")
                elif fired["X"] and args.live:
                    _home(left, right, args, xrt, lambda: stop)
                    # A may have been pressed while the blocking home ran.  Do
                    # not let it immediately start the other arm on return.
                    presses["A"].fired = True
                if fired["A"] and not home_grip_released(state, "right", args.grip_release_threshold):
                    print("Release RIGHT GRIP before homing the right arm.")
                elif fired["A"] and args.live:
                    _home(right, left, args, xrt, lambda: stop)
                    presses["X"].fired = True

            commands = []
            for side in sides:
                try:
                    commands.append(_compute_arm(side, state, fresh, dt, started, args))
                except RuntimeError as exc:
                    print(f"{side.name} target rejected: {exc}")
                    _hold(side, args)
                    side.reset_clutch()
                    commands.append(None)

            if args.min_flange_separation > 0 and any(command is not None for command in commands):
                left_q = commands[0] if commands[0] is not None else wait_for_joint_angles(left.robot)
                right_q = commands[1] if commands[1] is not None else wait_for_joint_angles(right.robot)
                left_point = left.ik.flange_pose(left_q)[:3] + np.asarray(args.left_base_origin)
                right_point = right.ik.flange_pose(right_q)[:3] + np.asarray(args.right_base_origin)
                distance = float(np.linalg.norm(left_point - right_point))
                if distance < args.min_flange_separation:
                    print(
                        f"FLANGE_SEPARATION: {distance:.3f} m below "
                        f"{args.min_flange_separation:.3f} m; holding both arms"
                    )
                    for side in sides:
                        _hold(side, args)
                        side.reset_clutch()
                    commands = [None, None]

            if args.live:
                for side, command in zip(sides, commands):
                    if command is None:
                        continue
                    if args.joint_control_mode == "js":
                        side.robot.move_js(command.tolist())
                    else:
                        side.robot.move_j(command.tolist())

            if started - last_print >= 1.0:
                print(
                    f"fresh={fresh}",
                    f"buttons=X{int(bool(state['X_button']))}/Y{int(bool(state['Y_button']))}/"
                    f"A{int(bool(state['A_button']))}/B{int(bool(state['B_button']))}",
                    f"left(grip={float(state['left_grip']):.2f}, trig={float(state['left_trigger']):.2f}, "
                    f"{gripper_text(left)}, enabled={left.ready}, status={arm_status_text(left.robot)})",
                    f"right(grip={float(state['right_grip']):.2f}, trig={float(state['right_trigger']):.2f}, "
                    f"{gripper_text(right)}, enabled={right.ready}, status={arm_status_text(right.robot)})",
                )
                if left.last_ik_diagnostics is not None:
                    print("left control:", ik_diagnostic_text(left))
                if right.last_ik_diagnostics is not None:
                    print("right control:", ik_diagnostic_text(right))
                last_print = started

            elapsed = time.monotonic() - started
            if elapsed < period:
                time.sleep(period - elapsed)

    except Exception:
        if args.live and any(side.enabled and not side.emergency_stopped for side in sides):
            try:
                _estop_all(sides)
                print("Both-arm emergency stop sent after control-loop failure.", file=sys.stderr)
            except Exception:
                pass
        raise
    finally:
        for side in sides:
            try:
                if args.live and side.enabled and not side.emergency_stopped:
                    hold_arm_on_exit(side.robot, args.joint_control_mode)
                    print(f"{side.name} arm is holding its current position")
            except Exception as exc:
                print(f"Failed to hold {side.name} arm: {exc}", file=sys.stderr)
        for robot in connected_robots:
            try:
                robot.disconnect()
            except Exception:
                pass
        xrt.close()
    print("Bimanual XR control stopped.")
    return 0


def run() -> int:
    try:
        return main()
    except KeyboardInterrupt:
        print("\n[bimanual-xr] interrupted")
        return 130
    except Exception as exc:
        print(f"\n[bimanual-xr] failed: {type(exc).__name__}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
