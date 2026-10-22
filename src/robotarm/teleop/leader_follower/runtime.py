"""双 Piper 软件主从遥操作与 episode 数据采集。

主臂使用零力拖动模式，关节逐对选择最新拖动输入或普通反馈，夹爪读取拖动输入，经关节映射、
限位、单周期限速和跟随误差检查后，通过另一个 CAN 接口向从臂发送 MOVE_JS
目标。训练用 action 是实际下发给从臂的目标；主臂原始状态另存为诊断字段。

程序默认 dry-run。真机运行必须显式指定 --execute；录制还需 --setup-id 和 --task。
--control-only 仅控制主从臂，不连接相机或创建采集目录。
运行时按键：l 进入主臂零力拖动，a 慢速对齐从臂，f 开始/暂停跟随，r 开始/保存 episode，
h 空闲时让双臂和两侧夹爪依次回零，d 丢弃 episode，x 从臂电子急停并退出，q 保持两臂并退出。正常退出时主臂进入受控使能保持；回零期间急停后，
主臂角色可能仍为零力拖动，需单独恢复角色。
"""

from __future__ import annotations

import argparse
import math
import select
import sys
import termios
import time
import traceback
import tty
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .can_diagnostics import home_receive_diagnostics
from ...config import defaults
from ...core.robot import piper as base
from ...hardware import home as arm_home
from ...core.robot.safety import (
    ensure_startup_status,
    safe_joint_bounds,
)
from ...paths import PIPER_RAW_ROOT
from ...record.recorder import RawEpisodeRecorder
from ...record.schema import LEADER_FOLLOWER_SAMPLE_FIELDS


# Encoder tolerance affects measured feedback only, never command limits.
FEEDBACK_LIMIT_TOLERANCE_DEG = 5.0


def validate_joint_feedback(joints: list[float], config: dict[str, Any]) -> None:
    if len(joints) != 6 or not all(math.isfinite(value) for value in joints):
        raise RuntimeError("关节反馈格式异常，拒绝控制。")
    tolerance = math.radians(FEEDBACK_LIMIT_TOLERANCE_DEG)
    for index, value in enumerate(joints, 1):
        low, high = config["joint_limits"][f"joint{index}"]
        if not low - tolerance <= value <= high + tolerance:
            raise RuntimeError(
                f"J{index}反馈 {math.degrees(value):.2f}° 超出标称范围 "
                f"[{math.degrees(low):.2f}, {math.degrees(high):.2f}]° "
                f"及 {FEEDBACK_LIMIT_TOLERANCE_DEG:g}° 反馈容差；停止控制。"
            )


def make_parser() -> argparse.ArgumentParser:
    cfg = defaults("data_acquisition", "collect-leader-follower")
    parser = argparse.ArgumentParser(description=__doc__)
    base.add_execute_argument(parser)
    parser.add_argument("--control-only", action="store_true",
                        help="control without cameras or data recording")
    parser.add_argument(
        "--check-only", action="store_true",
        help="read both arms once without cameras, mode changes, enabling or motion",
    )
    parser.add_argument("--leader-interface", default=cfg["leader-interface"])
    parser.add_argument("--leader-channel", default=cfg["leader-channel"])
    parser.add_argument("--follower-interface", default=cfg["follower-interface"])
    parser.add_argument("--follower-channel", default=cfg["follower-channel"])
    parser.add_argument(
        "--firmware", choices=("default", "v183", "v188", "v189"),
        default=cfg["firmware"],
    )
    parser.add_argument(
        "--feedback-timeout", type=base.positive_float,
        default=cfg["feedback-timeout"],
    )
    parser.add_argument(
        "--leader-init-timeout", type=base.positive_float,
        default=cfg["leader-init-timeout"],
        help="time to wait for an already-zero-gravity leader to emit 155/156/157",
    )
    parser.add_argument(
        "--stale-timeout", type=base.positive_float,
        default=cfg["stale-timeout"],
        help="maximum accepted CAN feedback age in seconds",
    )
    parser.add_argument(
        "--home-timeout", type=base.positive_float, default=cfg["home-timeout"],
        help="maximum native leader / MOVE_J follower / gripper homing time in seconds",
    )
    parser.add_argument(
        "--home-gripper-speed-mm-s", type=base.positive_float,
        default=cfg["home-gripper-speed-mm-s"],
        help="maximum gripper closing speed during h in mm/s",
    )
    parser.add_argument(
        "--control-rate", type=base.positive_float, default=cfg["control-rate"],
    )
    parser.add_argument(
        "--speed", type=base.percentage, default=cfg["speed"],
        help="follower controller speed percentage",
    )
    parser.add_argument(
        "--max-step-joint-deg", type=base.positive_float,
        default=cfg["max-step-joint-deg"],
    )
    parser.add_argument(
        "--max-following-error-deg", type=base.positive_float,
        default=cfg["max-following-error-deg"],
    )
    parser.add_argument(
        "--startup-alignment-deg", type=base.positive_float,
        default=cfg["startup-alignment-deg"],
    )
    parser.add_argument(
        "--joint-limit-margin-deg", type=float,
        default=cfg["joint-limit-margin-deg"],
        help="margin retained inside the configured joint limits (minimum 0.5 deg)",
    )
    parser.add_argument(
        "--joint-signs", type=int, nargs=6,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        default=cfg["joint-signs"],
        help="per-joint leader-to-follower sign, each value must be -1 or 1",
    )
    parser.add_argument(
        "--joint-offsets-deg", type=float, nargs=6,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        default=cfg["joint-offsets-deg"],
    )
    parser.add_argument(
        "--gripper-max-mm", type=base.positive_float,
        default=cfg["gripper-max-mm"],
    )
    parser.add_argument(
        "--gripper-force", type=base.positive_float,
        default=cfg["gripper-force"],
    )
    parser.add_argument(
        "--max-gripper-step-mm", type=base.positive_float,
        default=cfg["max-gripper-step-mm"],
    )
    parser.add_argument(
        "--max-gripper-following-error-mm", type=base.positive_float,
        default=cfg["max-gripper-following-error-mm"],
    )
    parser.add_argument(
        "--allow-enabled-start", action="store_true",
        help="allow the follower to be enabled when this program starts",
    )
    parser.add_argument("--global-camera-serial", default=cfg["global-camera-serial"])
    parser.add_argument("--wrist-camera-serial", default=cfg["wrist-camera-serial"])
    parser.add_argument("--record-width", type=int, default=cfg["record-width"])
    parser.add_argument("--record-height", type=int, default=cfg["record-height"])
    parser.add_argument("--record-fps", type=int, default=cfg["record-fps"])
    parser.add_argument(
        "--encoder-queue-size", type=int, default=cfg["encoder-queue-size"]
    )
    parser.add_argument("--dataset-root", type=Path, default=PIPER_RAW_ROOT)
    parser.add_argument("--setup-id", default=None)
    parser.add_argument("--setup-note", default="")
    parser.add_argument("--task", default=None)
    parser.add_argument("--no-preview", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return make_parser().parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if (
        args.leader_interface == args.follower_interface
        and str(args.leader_channel) == str(args.follower_channel)
    ):
        raise ValueError("leader and follower CAN endpoints must differ")
    if any(value not in (-1, 1) for value in args.joint_signs):
        raise ValueError("--joint-signs values must each be -1 or 1")
    numeric = [
        args.control_rate, args.stale_timeout, args.leader_init_timeout,
        args.home_timeout, args.home_gripper_speed_mm_s,
        args.max_step_joint_deg,
        args.max_following_error_deg, args.startup_alignment_deg,
        args.joint_limit_margin_deg,
        args.gripper_max_mm, args.gripper_force, args.max_gripper_step_mm,
        args.max_gripper_following_error_mm,
        *args.joint_offsets_deg,
    ]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("numeric arguments must be finite")
    if not 10 <= args.control_rate <= 100:
        raise ValueError("--control-rate must be between 10 and 100 Hz")
    if not 0.05 <= args.stale_timeout <= 1.0:
        raise ValueError("--stale-timeout must be between 0.05 and 1.0 seconds")
    if not 3.0 <= args.leader_init_timeout <= 120.0:
        raise ValueError("--leader-init-timeout must be between 3 and 120 seconds")
    if not 5.0 <= args.home_timeout <= 180.0:
        raise ValueError("--home-timeout must be between 5 and 180 seconds")
    if args.home_gripper_speed_mm_s > 50:
        raise ValueError("--home-gripper-speed-mm-s must not exceed 50")
    if args.max_step_joint_deg > 3.0:
        raise ValueError("--max-step-joint-deg must not exceed 3 degrees")
    if args.max_following_error_deg > 20.0:
        raise ValueError("--max-following-error-deg must not exceed 20 degrees")
    if args.startup_alignment_deg > 30.0:
        raise ValueError("--startup-alignment-deg must not exceed 30 degrees")
    if not 0.5 <= args.joint_limit_margin_deg <= 5.0:
        raise ValueError("--joint-limit-margin-deg must be between 0.5 and 5 degrees")
    if args.gripper_max_mm > 100 or args.gripper_force > 5:
        raise ValueError("gripper max/force exceeds the safety limit")
    if args.max_gripper_step_mm > 10:
        raise ValueError("--max-gripper-step-mm must not exceed 10 mm")
    if args.max_gripper_following_error_mm > 50:
        raise ValueError(
            "--max-gripper-following-error-mm must not exceed 50 mm"
        )
    if args.record_width <= 0 or args.record_height <= 0:
        raise ValueError("record dimensions must be positive")
    if not 1 <= args.record_fps <= args.control_rate:
        raise ValueError("--record-fps must be between 1 and --control-rate")
    if not 10 <= args.encoder_queue_size <= 600:
        raise ValueError("--encoder-queue-size must be between 10 and 600")
    if not (args.control_only or args.check_only):
        if not args.global_camera_serial or not args.wrist_camera_serial:
            raise ValueError(
                "recording requires two cameras: --global-camera-serial and "
                "--wrist-camera-serial"
            )
    if args.wrist_camera_serial and args.wrist_camera_serial == args.global_camera_serial:
        raise ValueError("global and wrist camera serial numbers must differ")
    if args.execute and not args.control_only and (not args.setup_id or not args.task):
        raise ValueError("--execute requires non-empty --setup-id and --task")


def map_leader_joints(
    joints: list[float], signs: list[int], offsets_deg: list[float]
) -> list[float]:
    if len(joints) != 6:
        raise RuntimeError("leader joint feedback must contain six values")
    return [
        sign * value + math.radians(offset)
        for value, sign, offset in zip(joints, signs, offsets_deg)
    ]


def validate_target_bounds(
    target: list[float], bounds: list[tuple[float, float]]
) -> None:
    if len(target) != 6 or len(bounds) != 6:
        raise RuntimeError("joint target and bounds must each contain six values")
    for index, (value, (low, high)) in enumerate(zip(target, bounds), start=1):
        if not math.isfinite(value) or not low <= value <= high:
            raise RuntimeError(
                f"mapped leader J{index}={math.degrees(value):.2f} deg is outside "
                f"the safe range [{math.degrees(low):.2f}, {math.degrees(high):.2f}]"
            )


def prepare_leader_target(
    target: list[float], bounds: list[tuple[float, float]], control_only: bool,
) -> list[float]:
    """Saturate goals in both recording and control-only modes."""
    if len(target) != 6 or len(bounds) != 6 or any(not math.isfinite(v) for v in target):
        raise RuntimeError("leader target must contain six finite joints with six bounds")
    applied = [min(max(value, low), high) for value, (low, high) in zip(target, bounds)]
    validate_target_bounds(applied, bounds)
    return applied


def command_joint_bounds(
    config: dict[str, Any], margin_degrees: float,
) -> list[tuple[float, float]]:
    """Keep feedback zero-offset tolerance out of targets sent to the SDK."""
    feedback_bounds = safe_joint_bounds(config, margin_degrees=margin_degrees)
    limits = config.get("joint_limits")
    if limits is None:
        # Lightweight offline mocks may provide bounds directly.
        return feedback_bounds
    bounds = []
    for index, (safe_low, safe_high) in enumerate(feedback_bounds, 1):
        hard_low, hard_high = limits[f"joint{index}"]
        low, high = max(safe_low, hard_low), min(safe_high, hard_high)
        if low > high:
            raise RuntimeError(f"J{index} has no valid command range")
        bounds.append((low, high))
    return bounds


def clip_command_to_bounds(
    joints: list[float], bounds: list[tuple[float, float]],
) -> list[float]:
    if len(joints) != 6 or len(bounds) != 6 or any(not math.isfinite(v) for v in joints):
        raise RuntimeError("joint command must contain six finite values and six bounds")
    return [min(max(value, low), high) for value, (low, high) in zip(joints, bounds)]


def clip_feedback_zero_offsets(
    target: list[float], config: dict[str, Any],
) -> list[float]:
    """Normalize small J2/J3 encoder zero offsets before target validation."""
    limits = config.get("joint_limits")
    if limits is None:
        return list(target)
    clipped = list(target)
    tolerance = math.radians(3.0)
    for index in (1, 2):
        low, high = limits[f"joint{index + 1}"]
        value = clipped[index]
        if low - tolerance <= value < low:
            clipped[index] = low
        elif high < value <= high + tolerance:
            clipped[index] = high
    return clipped


def normalize_hold_roundoff(joints: list[float], config: dict[str, Any]) -> list[float]:
    """Remove sub-microradian feedback/SDK rounding discrepancies at hard limits."""
    hold = list(joints)
    for i, value in enumerate(hold):
        low, high = config["joint_limits"][f"joint{i+1}"]
        if high < value <= high + 1e-6:
            hold[i] = high
        elif low - 1e-6 <= value < low:
            hold[i] = low
    return hold


def limit_joint_target(
    requested: list[float],
    previous: list[float],
    feedback: list[float],
    max_step: float,
    max_following_error: float,
) -> list[float]:
    """Limit a joint-vector step while preserving multi-joint direction ratios."""
    if not (len(requested) == len(previous) == len(feedback) == 6):
        raise RuntimeError("joint target vectors must each contain six values")
    if any(
        abs(command - actual) > max_following_error + 1e-9
        for command, actual in zip(previous, feedback)
    ):
        errors = [math.degrees(c - a) for c, a in zip(previous, feedback)]
        raise RuntimeError(
            "follower exceeded the maximum following error: "
            + ", ".join(f"J{i}={e:+.2f}deg" for i, e in enumerate(errors, 1))
        )
    delta = [target - current for target, current in zip(requested, previous)]
    largest = max(abs(value) for value in delta)
    scale = 1.0 if largest <= max_step else max_step / largest
    candidate = [current + value * scale for current, value in zip(previous, delta)]
    if any(
        abs(command - actual) > max_following_error + 1e-9
        for command, actual in zip(candidate, feedback)
    ):
        room_scale = min(
            max(0.0, (max_following_error - abs(old - actual)) / abs(new - old))
            if abs(new - old) > 0 else 1.0
            for new, old, actual in zip(candidate, previous, feedback)
        )
        candidate = [
            old + (new - old) * room_scale
            for new, old in zip(candidate, previous)
        ]
    return candidate


def feedback_age(message: Any, now_wall: float | None = None) -> float:
    timestamp = float(getattr(message, "timestamp", 0.0))
    now_wall = time.time() if now_wall is None else now_wall
    if timestamp <= 0:
        return math.inf
    return max(0.0, now_wall - timestamp)


def joint_feedback_age(
    robot: Any, message: Any, now_wall: float | None = None
) -> float:
    """Return the oldest age across the three two-joint Piper CAN frames."""
    now_wall = time.time() if now_wall is None else now_wall
    parser = getattr(robot, "_parser", None)
    if parser is None:
        return feedback_age(message, now_wall)
    pair_messages = [
        getattr(parser, name, None) for name in ("joint_12", "joint_34", "joint_56")
    ]
    if any(pair is None for pair in pair_messages):
        return math.inf
    return max(feedback_age(pair, now_wall) for pair in pair_messages)


def read_joint_message(
    robot: Any, config: dict[str, Any], role: str
) -> tuple[list[float], float]:
    message = robot.get_joint_angles()
    if message is None:
        raise RuntimeError(f"{role} joint feedback is unavailable")
    joints = list(message.msg)
    validate_joint_feedback(joints, config)
    return joints, joint_feedback_age(robot, message)


def wait_joint_message(
    robot: Any,
    config: dict[str, Any],
    role: str,
    timeout: float,
) -> tuple[list[float], float]:
    """Wait until all three Piper joint feedback frames have arrived."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            joints, age = read_joint_message(robot, config, role)
        except RuntimeError:
            pass
        else:
            if math.isfinite(age) and age <= timeout:
                return joints, age
        time.sleep(0.01)
    raise TimeoutError(
        f"{role} did not provide a complete six-joint feedback set within "
        f"{timeout:.1f} seconds"
    )


LEADER_JOINT_SOURCES = (
    ("leader_joint_12", "joint_12", "155", "2A5", 0, "joint_1", "joint_2"),
    ("leader_joint_34", "joint_34", "156", "2A6", 2, "joint_3", "joint_4"),
    ("leader_joint_56", "joint_56", "157", "2A7", 4, "joint_5", "joint_6"),
)


def newest_leader_joint_pairs(robot: Any) -> list[tuple[Any, str, int, str, str, float]]:
    """Select the newest received source independently for each joint pair."""
    parser = getattr(robot, "_parser", None)
    if parser is None:
        return []
    now_wall = time.time()
    selected = []
    for drag_attr, feedback_attr, drag_id, feedback_id, offset, first, second in LEADER_JOINT_SOURCES:
        candidates = []
        for attr, frame_id in ((drag_attr, drag_id), (feedback_attr, feedback_id)):
            message = getattr(parser, attr, None)
            if message is None:
                continue
            age = feedback_age(message, now_wall)
            if math.isfinite(age):
                candidates.append((age, message, frame_id))
        if candidates:
            age, message, frame_id = min(candidates, key=lambda item: item[0])
            selected.append((message, frame_id, offset, first, second, age))
    return selected


def update_leader_joint_message(
    robot: Any, previous: list[float], config: dict[str, Any],
) -> tuple[list[float], float]:
    """Merge freshest drag inputs or normal joint feedback, retaining missing pairs."""
    result = previous[:]
    ages = []
    for message, _, offset, first, second, age in newest_leader_joint_pairs(robot):
        result[offset] = float(getattr(message.msg, first))
        result[offset + 1] = float(getattr(message.msg, second))
        ages.append(age)
    validate_joint_feedback(result, config)
    return result, min(ages) if ages else math.inf


def read_complete_leader_joint_message(
    robot: Any, config: dict[str, Any]
) -> tuple[list[float], float]:
    """Read a complete initial 155/156/157 set from an existing drag mode."""
    parser = getattr(robot, "_parser", None)
    if parser is None or any(
        getattr(parser, name, None) is None
        for name in ("leader_joint_12", "leader_joint_34", "leader_joint_56")
    ):
        raise RuntimeError("complete leader joint input is unavailable")
    message = robot.get_leader_joint_angles()
    if message is None:
        raise RuntimeError("leader joint input is unavailable")
    joints = list(message.msg)
    validate_joint_feedback(joints, config)
    ages = [
        feedback_age(getattr(parser, name))
        for name in ("leader_joint_12", "leader_joint_34", "leader_joint_56")
    ]
    return joints, min(ages)


def wait_initial_leader_joints(
    robot: Any,
    config: dict[str, Any],
    timeout: float,
) -> tuple[list[float], str]:
    """Accept either normal 2A feedback or an already-active drag input stream."""
    deadline = time.monotonic() + timeout
    announced = False
    while time.monotonic() < deadline:
        try:
            joints, age = read_joint_message(robot, config, "leader")
        except RuntimeError:
            pass
        else:
            if math.isfinite(age) and age <= timeout:
                return joints, "feedback"
        try:
            joints, _ = read_complete_leader_joint_message(robot, config)
        except RuntimeError:
            pass
        else:
            return joints, "drag_input"
        if not announced and time.monotonic() + 1.0 < deadline:
            print(
                "[leader-follower] waiting for leader input; if it is already in "
                "zero-gravity mode, gently move all joint pairs once."
            )
            announced = True
        time.sleep(0.01)
    raise TimeoutError(
        "leader provided neither complete 2A1/2A2/2A3 feedback nor a complete "
        f"155/156/157 drag-input set within {timeout:.1f} seconds"
    )


def read_gripper_message(gripper: Any, role: str) -> tuple[float, float]:
    message = gripper.get_gripper_status()
    if message is None or message.msg.mode != "width":
        raise RuntimeError(f"{role} gripper feedback is unavailable")
    width = float(message.msg.value)
    if not math.isfinite(width):
        raise RuntimeError(f"{role} gripper feedback is not finite")
    return width, feedback_age(message)


def read_leader_gripper_command(
    gripper: Any, previous: float
) -> tuple[float, float]:
    """Read event-driven 0x159 leader gripper input, retaining the last value."""
    message = gripper.get_gripper_ctrl_states()
    if message is None:
        return previous, math.inf
    value = float(message.msg.value)
    status_code = int(message.msg.status_code)
    if status_code not in (0x00, 0x01, 0x02, 0x03):
        raise RuntimeError("leader 0x159 gripper input is not in width mode")
    if not math.isfinite(value):
        raise RuntimeError("leader 0x159 gripper input is not finite")
    return value, feedback_age(message)


def wait_leader_gripper_command(
    gripper: Any, timeout: float
) -> tuple[float, float]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value, age = read_leader_gripper_command(gripper, 0.0)
        if math.isfinite(age):
            return value, age
        time.sleep(0.01)
    raise TimeoutError(
        f"leader did not provide a 0x159 gripper input within {timeout:.1f} seconds"
    )


def finite_combined_age(*ages: float) -> float:
    finite = [age for age in ages if math.isfinite(age)]
    return max(finite) if finite else -1.0


class KeyReader:
    """Non-blocking single-key terminal reader that restores terminal state."""

    def __init__(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError("interactive terminal required for live leader-follower control")
        self.fd = sys.stdin.fileno()
        self.original = termios.tcgetattr(self.fd)

    def __enter__(self) -> "KeyReader":
        tty.setcbreak(self.fd)
        return self

    def poll(self) -> str | None:
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        return sys.stdin.read(1).lower() if readable else None

    def __exit__(self, *_: object) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.original)


def connection_args(args: argparse.Namespace, role: str) -> SimpleNamespace:
    return SimpleNamespace(
        interface=getattr(args, f"{role}_interface"),
        channel=getattr(args, f"{role}_channel"),
        firmware=args.firmware,
        feedback_timeout=args.feedback_timeout,
    )


def enter_follower_j(
    robot: Any,
    gripper: Any,
    joints: list[float],
    gripper_width: float,
    force: float,
    already_enabled: bool,
) -> None:
    robot.set_auto_set_motion_mode_enabled(False)
    robot.move_j(joints)
    robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.J)
    robot.move_j(joints)
    if not already_enabled:
        base.retry_boolean(robot.enable, 5.0, "Enable follower")
    base.set_gripper_enabled(gripper, True, gripper_width, force, 5.0)
    time.sleep(0.1)
    robot.move_j(joints)


def enter_follower_js(
    robot: Any, gripper: Any, joints: list[float], gripper_width: float,
    force: float, already_enabled: bool,
) -> None:
    enter_follower_j(robot, gripper, joints, gripper_width, force, already_enabled)
    robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.JS)
    robot.move_js(joints)


def exit_leader_drag_mode(
    robot: Any,
    joints: list[float],
    gripper: Any | None,
    gripper_width: float,
    force: float,
    timeout: float = 5.0,
    config: dict[str, Any] | None = None,
) -> None:
    """Leave zero-force leader mode and hold the current pose while enabled.

    Piper exposes leader and follower roles rather than a separate "normal"
    role. Follower mode plus a JS target at the last measured leader pose is
    therefore the controlled, enabled state used after collection.
    """
    if len(joints) != 6 or any(not math.isfinite(value) for value in joints):
        raise RuntimeError("cannot exit leader drag mode without six valid joints")
    if not math.isfinite(gripper_width):
        raise RuntimeError("cannot hold a non-finite leader gripper width")

    hold = list(joints) if config is None else arm_home.bounded_hold(joints, config)
    robot.set_follower_mode()
    # Let the controller leave leader input mode before sending the hold target.
    # Sending the target before enable prevents a jump at the transition.
    time.sleep(0.1)
    robot.set_auto_set_motion_mode_enabled(False)
    robot.move_j(hold)
    robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.J)
    robot.move_j(hold)
    base.retry_boolean(robot.enable, timeout, "Enable leader after drag mode")
    if gripper is not None:
        base.set_gripper_enabled(gripper, True, gripper_width, force, timeout)
    time.sleep(0.1)
    robot.move_j(hold)
    robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.JS)
    robot.move_js(hold)
    states = robot.get_joints_enable_status_list()
    if len(states) != 6 or not all(states):
        raise RuntimeError(
            "leader did not enter the enabled hold state: " + str(states)
        )


ALIGNMENT_SPEED_PERCENT = 10
ALIGNMENT_RATE_RAD_S = math.radians(5.0)
ALIGNMENT_TOLERANCE_RAD = math.radians(2.0)
ALIGNMENT_SETTLE_SECONDS = 0.5


def alignment_complete(
    target: list[float], command: list[float], actual: list[float],
) -> bool:
    # Wait for both the ramp and physical feedback; never hand off early.
    return all(abs(t - c) < 1e-9 for t, c in zip(target, command)) and all(
        abs(t - a) <= ALIGNMENT_TOLERANCE_RAD for t, a in zip(target, actual)
    )


def ensure_receive_running(robot: Any, role: str) -> None:
    """Reject a failed SDK receiver rather than repeatedly using cached input."""
    ctx = getattr(robot, "_ctx", None)
    if ctx is None:
        return
    comm = getattr(ctx, "comm", None)
    error = getattr(comm, "last_error", None) if comm is not None else None
    thread = getattr(ctx, "_read_th", None)
    stopped = thread is not None and not thread.is_alive()
    if error is not None or stopped:
        detail = f"{type(error).__name__}: {error}" if error is not None else "receiver thread stopped"
        raise RuntimeError(
            f"{role} CAN receiver is not healthy ({detail}); "
            "cached joint values cannot be used for following"
        )


def print_arm_diagnostics(
    args: argparse.Namespace, leader: Any, follower: Any,
    leader_joints: list[float], follower_joints: list[float],
) -> None:
    """Read status without changing either arm's mode or enable state."""
    for role, robot, joints in (
        ("leader", leader, leader_joints), ("follower", follower, follower_joints),
    ):
        status = robot.get_arm_status()
        fields = "status unavailable"
        if status is not None:
            fields = ", ".join(
                f"{name}={getattr(status.msg, name, 'unavailable')}"
                for name in ("ctrl_mode", "arm_status", "mode_feedback")
            )
        if role == "leader":
            sources = newest_leader_joint_pairs(robot)
            print("[diagnostic] leader joint sources: " + (
                ", ".join(
                    f"J{offset+1}/{offset+2}=0x{frame_id} age={age*1000:.0f}ms"
                    for _, frame_id, offset, _, _, age in sources
                ) if sources else "no timestamped joint frames"
            ))
        angles = ", ".join(f"{math.degrees(value):+.2f}" for value in joints)
        print(f"\n[diagnostic] {role} {getattr(args, role + '_channel')}: "
              f"{fields}; joints_deg=[{angles}]")


class MotionDiagnostics:
    """Print coherent snapshots on explicit request, with movement since the last snapshot."""

    def __init__(self) -> None:
        self.last_time: float | None = None
        self.previous_follower: list[float] | None = None

    def emit(
        self, now: float, args: argparse.Namespace, leader: Any, follower: Any,
        leader_joints: list[float], requested: list[float],
        last_sent: list[float] | None, follower_joints: list[float],
        leader_age: float, follower_age: float,
        following: bool, aligning: bool, force: bool = False,
        bounded_target: list[float] | None = None,
    ) -> None:
        if not force and self.last_time is not None and now - self.last_time < 0.5:
            return
        elapsed = None if self.last_time is None else now - self.last_time
        mode = "following" if following else "aligning" if aligning else "holding"
        print(
            f"\n[motion] mode={mode}; following={following}; "
            f"leader_age={leader_age*1000:.0f}ms; follower_age={follower_age*1000:.0f}ms"
        )
        print_arm_diagnostics(args, leader, follower, leader_joints, follower_joints)
        degrees = lambda values: "[" + ", ".join(f"{math.degrees(v):+.2f}" for v in values) + "]"
        print("[motion] joint order: J1 J2 J3 J4 J5 J6; units: deg")
        print(f"[motion] mapped_leader={degrees(requested)}")
        if bounded_target is not None:
            clipped = [
                f"J{i}: {math.degrees(raw):+.2f} -> {math.degrees(applied):+.2f}deg"
                for i, (raw, applied) in enumerate(zip(requested, bounded_target), 1)
                if abs(raw - applied) > 1e-9
            ]
            if clipped:
                print("[motion] joint_limit_active: " + "; ".join(clipped))
        if last_sent is None:
            print("[motion] sent_target=none (no joint target sent yet)")
        else:
            print(f"[motion] sent_target={degrees(last_sent)}")
            print("[motion] target_minus_actual=" + degrees([
                target - actual for target, actual in zip(last_sent, follower_joints)
            ]))
        if self.previous_follower is None:
            print("[motion] follower_delta=unavailable (first snapshot)")
        else:
            print(f"[motion] follower_delta over {elapsed:.3f}s=" + degrees([
                actual - previous for actual, previous in zip(follower_joints, self.previous_follower)
            ]))
        self.last_time = now
        self.previous_follower = list(follower_joints)
        print("[motion] end", flush=True)



def status_line(following: bool, aligning: bool, leader_drag: bool,
                recording: bool, saved_episode: bool,
                remaining_deg: float | None = None) -> str:
    if aligning:
        mode = "对齐中"
    elif following:
        mode = "跟随中"
    elif leader_drag:
        mode = "主臂零力拖动 / 从臂保持"
    else:
        mode = "双臂保持"
    record = "录制中" if recording else "已保存，可按 h 回零" if saved_episode else "未录制"
    progress = f" | 剩余 {remaining_deg:.2f}°" if aligning and remaining_deg is not None else ""
    return f"[状态] {mode} | {record}{progress} | s 查看详细信息"


def print_controls() -> None:
    print("\nLeader-follower controls:")
    print("  l: enable leader zero-force drag mode")
    print("  a: slowly align follower to a snapshot of leader joints")
    print("  f: start/pause following; cancel alignment while aligning")
    print("  s: show detailed arm modes, joints and following errors on demand")
    print("  r: start/save episode    d: discard episode")
    print("  h: when idle, native-home leader, then MOVE_J-home follower; close both grippers")
    print("  q: hold both arms and quit")
    print("  x: follower electronic emergency stop, hold leader and quit")
    print("  normal exit: leader holds enabled; after homing E-stop, restore its role separately\n")



def read_native_leader_home_joints(
    robot: Any, previous: list[float], config: dict[str, Any],
    sent_wall_time: float, seen_pairs: set[int],
) -> list[float]:
    """Merge joint pairs received after 0x191; stationary pairs retain their snapshot."""
    joints = list(previous)
    pairs = newest_leader_joint_pairs(robot)
    if not pairs:
        # A test/future SDK without parser access can still provide aggregated
        # feedback, but a cached pre-command message cannot prove movement.
        message = robot.get_joint_angles()
        if message is not None and float(message.timestamp) > sent_wall_time:
            joints = list(message.msg)
            seen_pairs.update((0, 1, 2))
    for message, _, offset, first, second, _ in pairs:
        if float(message.timestamp) <= sent_wall_time:
            continue
        joints[offset] = float(getattr(message.msg, first))
        joints[offset + 1] = float(getattr(message.msg, second))
        seen_pairs.add(offset // 2)
    return arm_home.validate_joint_values(
        joints, config, arm_home.DEFAULT_JOINT_LIMIT_TOLERANCE_DEG
    )


def read_fresh_motor_joint_snapshot(robot: Any, now_wall: float) -> tuple[list[float], float] | None:
    """Read six independent 0x251-0x256 motor positions if all are fresh."""
    get_motor_states = getattr(robot, "get_motor_states", None)
    if not callable(get_motor_states):
        return None
    positions = []
    timestamps = []
    for index in range(1, 7):
        message = get_motor_states(index)
        if message is None:
            return None
        try:
            stamp = float(message.timestamp)
            position = float(message.msg.position)
        except (AttributeError, TypeError, ValueError):
            return None
        if (not math.isfinite(stamp) or not math.isfinite(position)
                or stamp <= 0 or not 0 <= now_wall - stamp <= 0.5):
            return None
        positions.append(position)
        timestamps.append(stamp)
    return positions, min(timestamps)


def home_leader_native(
    robot: Any, config: dict[str, Any], channel: str,
    initial_joints: list[float], timeout: float, stop_requested: Any,
    on_command_start: Any = None,
) -> list[float]:
    """Verify native homing, reporting raw receive evidence on failure."""
    with home_receive_diagnostics(robot, channel):
        return _home_leader_native(
            robot, config, channel, initial_joints, timeout, stop_requested,
            on_command_start,
        )


def _home_leader_native(
    robot: Any, config: dict[str, Any], channel: str,
    initial_joints: list[float], timeout: float, stop_requested: Any,
    on_command_start: Any = None,
) -> list[float]:
    """Use firmware 0x191 mode 1 and verify actual joints before returning."""
    joints = arm_home.validate_joint_values(
        initial_joints, config, arm_home.DEFAULT_JOINT_LIMIT_TOLERANCE_DEG
    )
    base.ensure_arm_status_normal(robot)
    if stop_requested():
        raise InterruptedError("leader homing interrupted before firmware command")
    if max(abs(value) for value in joints) <= math.radians(3.0):
        print(f'{channel}: leader already within 3 deg of home; no 0x191 motion needed.')
        return joints
    move_home = getattr(robot, "move_leader_to_home", None)
    if not callable(move_home):
        raise RuntimeError("installed Piper SDK lacks move_leader_to_home")
    print(
        f"HOMING {channel}: firmware leader-home 0x191 (controller-defined speed); "
        "Ctrl+C/q/x E-stops the active arm"
    )
    # In leader mode the controller can stop publishing 0x155/156/157 and
    # 0x2A5/2A6/2A7. The independent high-speed motor positions are usable as
    # joint angles only when a fresh pre-command snapshot agrees with the
    # measured joint pose; otherwise their coordinate system is unknown.
    motor_initial = read_fresh_motor_joint_snapshot(robot, time.time())
    motor_agrees = (motor_initial is not None and all(
        abs(motor - joint) <= math.radians(1.0)
        for motor, joint in zip(motor_initial[0], joints)
    ))
    if motor_initial is None:
        print(f"{channel}: no complete fresh 0x251-0x256 motor snapshot; waiting for joint frames.")
    elif not motor_agrees:
        largest_disagreement = math.degrees(max(
            abs(motor - joint) for motor, joint in zip(motor_initial[0], joints)
        ))
        print(
            f"{channel}: motor/joint coordinates differ by up to "
            f"{largest_disagreement:.2f} deg (>1 deg); motor fallback disabled."
        )
    sent_wall_time = time.time()
    if on_command_start is not None:
        on_command_start()
    move_home()
    started = time.monotonic()
    last_print = -math.inf
    last_source = None
    best_total_error_deg = sum(math.degrees(abs(value)) for value in joints)
    last_progress_at = started
    settled_since = None
    seen_pairs: set[int] = set()
    required_pairs = {
        i // 2 for i, value in enumerate(joints)
        if abs(value) > math.radians(3.0)
    }
    while time.monotonic() - started < timeout:
        if stop_requested():
            raise InterruptedError("leader firmware homing interrupted")
        now = time.monotonic()
        joints = read_native_leader_home_joints(
            robot, joints, config, sent_wall_time, seen_pairs
        )
        now_wall = time.time()
        pair_stamps = {
            offset // 2: float(message.timestamp)
            for message, _, offset, _, _, _ in newest_leader_joint_pairs(robot)
            if float(message.timestamp) > sent_wall_time
        }
        position_fresh = bool(required_pairs) and all(
            pair in pair_stamps and 0 <= now_wall - pair_stamps[pair] <= 0.5
            for pair in required_pairs
        )
        source = "joint frames" if position_fresh else "stale position snapshot"
        if motor_agrees:
            motor_snapshot = read_fresh_motor_joint_snapshot(robot, now_wall)
            newest_pair_stamp = max(pair_stamps.values(), default=sent_wall_time)
            if (motor_snapshot is not None
                    and motor_snapshot[1] > sent_wall_time
                    and motor_snapshot[1] >= newest_pair_stamp):
                joints = arm_home.validate_joint_values(
                    motor_snapshot[0], config,
                    arm_home.DEFAULT_JOINT_LIMIT_TOLERANCE_DEG,
                )
                seen_pairs.update((0, 1, 2))
                position_fresh = True
                source = "motor 0x251-0x256"
        status = robot.get_arm_status()
        if status is None or base.value_as_int(status.msg.arm_status) != 0:
            raise RuntimeError("leader status missing or abnormal during firmware home")
        error = math.degrees(max(abs(value) for value in joints))
        total_error_deg = sum(math.degrees(abs(value)) for value in joints)
        if position_fresh and total_error_deg <= best_total_error_deg - 1.0:
            best_total_error_deg = total_error_deg
            last_progress_at = now
        if (not position_fresh or error > 3.0) and now - last_progress_at >= 8.0:
            reason = (
                "no fresh position feedback"
                if not position_fresh else "made no meaningful progress"
            )
            raise RuntimeError(
                f"{channel} firmware leader home {reason} "
                f"for 8 s; last_known_max_error={error:.2f} deg; "
                f"updated_joint_pairs={sorted(seen_pairs)}; "
                f"ctrl_mode={getattr(status.msg, 'ctrl_mode', 'unavailable')}, "
                f"motion_status={getattr(status.msg, 'motion_status', 'unavailable')}"
            )
        if now - last_print >= 0.5 or source != last_source:
            angles = ', '.join(f'{math.degrees(value):+.2f}' for value in joints)
            if position_fresh:
                print(f'{channel}: native remaining={error:.2f} deg; joints_deg=[{angles}]; source={source}')
            else:
                print(f'{channel}: waiting for fresh position feedback; last_known_joints_deg=[{angles}]')
            last_print = now
            last_source = source
        if position_fresh and error <= 3.0 and required_pairs <= seen_pairs:
            if settled_since is None:
                settled_since = now
            elif now - settled_since >= 0.5:
                print(f'{channel}: firmware leader home reached zero pose.')
                return joints
        else:
            settled_since = None
        time.sleep(0.02)
    raise TimeoutError(f'{channel} firmware leader home did not finish within {timeout:g} s')


def close_gripper_slowly(
    gripper: Any, force: float, speed_mm_s: float,
    timeout: float, stop_requested: Any, *, robot: Any = None,
) -> None:
    """Close the leader gripper after leaving drag mode, without joint commands."""
    width = arm_home.read_gripper_width(gripper, 0.5)
    command = width
    started = previous = time.monotonic()
    settled_since = None
    best_width = width
    last_progress_at = started
    while time.monotonic() - started < timeout:
        if stop_requested():
            raise InterruptedError('leader gripper homing interrupted')
        if robot is not None:
            status = robot.get_arm_status()
            if status is None or base.value_as_int(status.msg.arm_status) != 0:
                raise RuntimeError('leader status missing or abnormal while closing gripper')
        now = time.monotonic()
        command = max(0.0, command - speed_mm_s / 1000.0 * max(0.0, now - previous))
        previous = now
        gripper.move_gripper_m(command, force)
        width = arm_home.read_gripper_width(gripper, 0.5)
        if width <= best_width - 0.001:
            best_width = width
            last_progress_at = now
        if width > 0.002 and now - last_progress_at >= 8.0:
            raise RuntimeError('leader gripper made no closing progress for 8 s')
        if command <= 0.002 and width <= 0.002:
            if settled_since is None:
                settled_since = now
            elif now - settled_since >= 0.5:
                gripper.move_gripper_m(0.0, force)
                print('leader gripper closed.')
                return
        else:
            settled_since = None
        time.sleep(0.02)
    raise TimeoutError(f'leader gripper did not close within {timeout:g} s')


def home_one_arm(robot: Any, config: dict[str, Any], channel: str,
                 gripper: Any, force: float, gripper_speed_mm_s: float,
                 timeout: float, stop_requested: Any) -> None:
    """MOVE_J to joint zero while rate-limiting the gripper to closed width."""
    options = SimpleNamespace(
        channel=channel, speed_percent=10, timeout=timeout,
        stale_timeout=0.5,
        joint_limit_tolerance_deg=arm_home.DEFAULT_JOINT_LIMIT_TOLERANCE_DEG,
        tolerance_deg=3.0,
    )
    arm_home.run_home(
        robot, config, options, stop_requested, gripper=gripper,
        gripper_force=force, gripper_speed_mm_s=gripper_speed_mm_s,
    )


def home_both_sequentially(
    follower: Any, follower_config: dict[str, Any], follower_channel: str,
    leader: Any, leader_config: dict[str, Any], leader_channel: str,
    follower_gripper: Any, leader_gripper: Any,
    force: float, gripper_speed_mm_s: float,
    timeout: float, stop_requested: Any,
    leader_initial_joints: list[float], leader_initial_gripper_width: float,
    on_estop: Any = None,
) -> list[float]:
    """Home leader with firmware 0x191, hold it, then home the follower."""
    active_arm = leader
    active_channel = leader_channel
    motion_started = False
    def mark_motion_started() -> None:
        nonlocal motion_started
        motion_started = True
    try:
        leader_joints = home_leader_native(
            leader, leader_config, leader_channel,
            leader_initial_joints, timeout, stop_requested,
            on_command_start=mark_motion_started,
        )
        # 0x191 mode 0 restores drag; we instead leave the leader in an
        # enabled hold using the measured zero pose before moving the follower.
        motion_started = True
        exit_leader_drag_mode(
            leader, leader_joints, leader_gripper,
            leader_initial_gripper_width, force, config=leader_config,
        )
        close_gripper_slowly(
            leader_gripper, force, gripper_speed_mm_s,
            timeout, stop_requested, robot=leader,
        )
        if stop_requested():
            raise InterruptedError("dual-arm homing interrupted between arms")
        active_arm = follower
        active_channel = follower_channel
        motion_started = True
        home_one_arm(
            follower, follower_config, follower_channel,
            follower_gripper, force, gripper_speed_mm_s, timeout, stop_requested,
        )
        return leader_joints
    except BaseException as error:
        interrupted = isinstance(error, (KeyboardInterrupt, InterruptedError))
        if motion_started:
            # Suppress exit-time motion commands even if the E-stop send fails.
            if on_estop is not None:
                on_estop()
            try:
                active_arm.electronic_emergency_stop()
                reason = "interrupted by operator" if interrupted else "failed"
                print(
                    f"[leader-follower] {active_channel} homing {reason} "
                    f"({type(error).__name__}: {error}); electronic emergency stop sent "
                    f"to {active_channel} to stop the active motion."
                )
            except Exception as stop_error:
                print(
                    f"[leader-follower] WARNING: failed to send homing E-stop "
                    f"to {active_channel}: {stop_error}",
                    file=sys.stderr,
                )
            if active_arm is leader:
                print(
                    "[leader-follower] WARNING: leader role was not restored after "
                    "E-stop and may persist after power-cycle. Support the arm; "
                    "use restore-piper-role on its CAN channel to select follower "
                    "role before restarting the controller.",
                    file=sys.stderr,
                )
        else:
            print(
                f"[leader-follower] homing preflight stopped before movement "
                f"({type(error).__name__}: {error}); no E-stop sent."
            )
        raise


def run_check(args: argparse.Namespace) -> int:
    """Connect read-only and print one synchronized dual-arm status snapshot."""
    leader = follower = None
    try:
        leader, leader_config = base.connect_robot(connection_args(args, "leader"))
        follower, follower_config = base.connect_robot(connection_args(args, "follower"))
        leader_joints, leader_source = wait_initial_leader_joints(
            leader, leader_config, args.leader_init_timeout
        )
        follower_joints, follower_age = wait_joint_message(
            follower, follower_config, "follower", args.feedback_timeout
        )
        # A leader that is already in zero-gravity mode may have stopped
        # publishing 2A5/2A6/2A7; report that age instead of failing the check.
        leader_age = joint_feedback_age(leader, leader.get_joint_angles())
        mapped = map_leader_joints(
            leader_joints, args.joint_signs, args.joint_offsets_deg
        )
        errors = [target - actual for target, actual in zip(mapped, follower_joints)]
        print(
            "Read-only dual-arm check "
            f"(leader source={leader_source}; no mode changes or control frames):"
        )
        for index, (lead, follow, error) in enumerate(
            zip(mapped, follower_joints, errors), start=1
        ):
            print(
                f"  J{index}: mapped leader={math.degrees(lead):8.3f} deg  "
                f"follower={math.degrees(follow):8.3f} deg  "
                f"delta={math.degrees(error):+8.3f} deg"
            )
        age_text = lambda age: (
            f"{age * 1000:.1f} ms" if math.isfinite(age) else "unavailable"
        )
        print(
            f"  feedback age: leader={age_text(leader_age)}, "
            f"follower={age_text(follower_age)}"
        )
        return 0
    finally:
        base.disconnect_quietly(follower)
        base.disconnect_quietly(leader)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    camera_serials = {} if args.control_only else {"global": args.global_camera_serial}
    if not args.control_only and args.wrist_camera_serial:
        camera_serials["wrist"] = args.wrist_camera_serial
    if args.check_only:
        return run_check(args)
    if not args.execute:
        print(
            "DRY-RUN: software leader-follower "
            f"{args.leader_channel} -> {args.follower_channel}, "
            f"control={args.control_rate:.0f} Hz, record={args.record_fps} FPS, "
            f"cameras={camera_serials}."
        )
        print("Add --execute to connect hardware." if args.control_only else
              "Add --execute, --setup-id and --task to connect hardware.")
        print_controls()
        print("h leader home uses firmware 0x191 at controller-defined speed; follower MOVE_J uses 10%.")
        return 0
    if not sys.stdin.isatty():
        raise RuntimeError("live leader-follower control requires an interactive terminal")

    leader = follower = recorder = None
    follower_gripper = leader_gripper = None
    following = False
    target_limit_active = False
    alignment_target: list[float] | None = None
    alignment_deadline = 0.0
    alignment_settled_since: float | None = None
    alignment_last_progress_at = 0.0
    alignment_best_error = math.inf
    follower_js_mode = False
    follower_enabled = False
    command: list[float] | None = None
    command_gripper = 0.0
    emergency_stop = False
    episode_ready_for_home = False
    leader_drag = False
    leader_joints: list[float] | None = None
    leader_gripper_width = 0.0
    motion_diagnostics = MotionDiagnostics()
    last_status_print = -math.inf
    last_sent_joints: list[float] | None = None
    try:
        print(
            f"[leader-follower] connecting leader {args.leader_channel} and "
            f"follower {args.follower_channel}...",
            flush=True,
        )
        leader, leader_config = base.connect_robot(connection_args(args, "leader"))
        follower, follower_config = base.connect_robot(
            connection_args(args, "follower")
        )
        leader_joints, leader_source = wait_initial_leader_joints(
            leader, leader_config, args.leader_init_timeout
        )
        leader_drag = leader_source == "drag_input"
        follower_joints, _ = wait_joint_message(
            follower, follower_config, "follower", args.feedback_timeout
        )
        validate_joint_feedback(leader_joints, leader_config)
        validate_joint_feedback(follower_joints, follower_config)
        if leader_source == "feedback":
            ensure_startup_status(leader, args.feedback_timeout)
        ensure_startup_status(follower, args.feedback_timeout)

        leader_gripper = leader.init_effector(leader.OPTIONS.EFFECTOR.AGX_GRIPPER)
        follower_gripper = follower.init_effector(
            follower.OPTIONS.EFFECTOR.AGX_GRIPPER
        )
        if leader_source == "feedback":
            leader_gripper_width = base.wait_gripper_width(
                leader_gripper, args.feedback_timeout
            )
        else:
            print(
                "[leader-follower] waiting for leader 0x159 gripper input; "
                "operate the leader gripper once if needed."
            )
            leader_gripper_width, _ = wait_leader_gripper_command(
                leader_gripper, args.leader_init_timeout
            )
        follower_gripper_width = base.wait_gripper_width(
            follower_gripper, args.feedback_timeout
        )
        command_gripper = min(
            max(leader_gripper_width, 0.0), args.gripper_max_mm / 1000.0
        )
        follower_enabled = base.get_all_joints_enabled_state(
            follower, args.feedback_timeout
        )
        if follower_enabled and not args.allow_enabled_start:
            raise RuntimeError(
                "follower is already enabled; confirm and add --allow-enabled-start"
            )
        follower.set_speed_percent(ALIGNMENT_SPEED_PERCENT)

        if not args.control_only:
            recorder = RawEpisodeRecorder(
                args.dataset_root, args.task, args.setup_id, args.setup_note,
                args.record_fps, args.record_width, args.record_height,
                camera_serials,
                {
                    "model": "Piper",
                    "control_mode": "software_leader_follower_move_js",
                    "firmware": args.firmware,
                    "control_rate_hz": args.control_rate,
                    "leader": {
                        "interface": args.leader_interface,
                        "channel": args.leader_channel,
                        "role": "zero_force_drag",
                    },
                    "follower": {
                        "interface": args.follower_interface,
                        "channel": args.follower_channel,
                        "role": "move_js",
                    },
                    "action_source": "rate_limited_mapped_leader_feedback",
                    "observation_source": "follower_feedback",
                    "joint_signs": list(args.joint_signs),
                    "joint_offsets_deg": list(args.joint_offsets_deg),
                    "joint_limit_margin_deg": args.joint_limit_margin_deg,
                    "joint_target_limit_policy": "clip",
                    "feedback_limit_tolerance_deg": FEEDBACK_LIMIT_TOLERANCE_DEG,
                    "joint_frame": "piper_base",
                    "tcp_pose_units": ["m", "m", "m", "rad", "rad", "rad"],
                },
                preview=not args.no_preview,
                encoder_queue_size=args.encoder_queue_size,
                extra_sample_fields=LEADER_FOLLOWER_SAMPLE_FIELDS,
            )
        bounds = command_joint_bounds(
            follower_config, margin_degrees=args.joint_limit_margin_deg
        )
        print(
            "[leader-follower] safe J5 target range: "
            f"[{math.degrees(bounds[4][0]):.2f}, "
            f"{math.degrees(bounds[4][1]):.2f}] deg "
            f"(margin {args.joint_limit_margin_deg:.2f} deg)"
        )
        max_step = math.radians(args.max_step_joint_deg)
        max_error = math.radians(args.max_following_error_deg)
        alignment_limit = min(
            math.radians(args.startup_alignment_deg),
            ALIGNMENT_TOLERANCE_RAD, max_error * 0.5,
        )
        gripper_max = args.gripper_max_mm / 1000.0
        gripper_step = args.max_gripper_step_mm / 1000.0
        max_gripper_error = args.max_gripper_following_error_mm / 1000.0
        period = 1.0 / args.control_rate
        next_gripper_command = 0.0
        if leader_drag:
            print(
                "[leader-follower] leader is already emitting zero-gravity "
                "155/156/157 input; press a to align, then f to follow."
            )
        else:
            print(
                "[leader-follower] connected. Press l to enter leader drag mode, "
                "then press a to align and f to follow."
            )
        print_controls()

        with KeyReader() as keys:
            while True:
                tick = time.monotonic()
                ensure_receive_running(leader, "leader")
                ensure_receive_running(follower, "follower")
                key = keys.poll()
                status_dirty = False
                alignment_remaining_deg = None
                if key == "l":
                    if following or alignment_target is not None:
                        print("\n[leader-follower] press f to pause motion before requesting drag mode.")
                    else:
                        # A sent request is not an acknowledgement. Allow retrying l
                        # even when the local flag was already set by an earlier request.
                        leader.set_leader_mode()
                        leader_drag = True
                        print(
                            "\n[leader-follower] leader zero-force drag mode "
                            f"requested on {args.leader_channel}; "
                            "joints use freshest 155/156/157 or 2A5/2A6/2A7, gripper uses 159."
                        )
                elif key == "a":
                    if not leader_drag:
                        print("\n[leader-follower] press l before alignment.")
                    elif following or alignment_target is not None:
                        print("\n[leader-follower] press f to pause motion before alignment.")
                    else:
                        leader_joints, _ = update_leader_joint_message(
                            leader, leader_joints, leader_config
                        )
                        target = map_leader_joints(
                            leader_joints, args.joint_signs, args.joint_offsets_deg
                        )
                        target = clip_feedback_zero_offsets(target, follower_config)
                        target = prepare_leader_target(target, bounds, args.control_only)
                        follower_joints, joint_age = read_joint_message(
                            follower, follower_config, "follower"
                        )
                        follower_width, gripper_age = read_gripper_message(
                            follower_gripper, "follower"
                        )
                        if max(joint_age, gripper_age) > args.stale_timeout:
                            raise RuntimeError("cannot align with stale follower feedback")
                        command = clip_command_to_bounds(follower_joints, bounds)
                        command_gripper = follower_width
                        follower.set_speed_percent(ALIGNMENT_SPEED_PERCENT)
                        enter_follower_j(
                            follower, follower_gripper, command,
                            follower_width, args.gripper_force, follower_enabled,
                        )
                        follower_enabled = True
                        follower_js_mode = False
                        alignment_settled_since = None
                        alignment_target = target[:]
                        duration = max(abs(t - c) for t, c in zip(target, command))
                        alignment_last_progress_at = time.monotonic()
                        alignment_best_error = max(
                            abs(t - a) for t, a in zip(target, follower_joints)
                        )
                        alignment_deadline = (
                            alignment_last_progress_at + duration / ALIGNMENT_RATE_RAD_S + 30.0
                        )
                        print(
                            "\n[leader-follower] aligning to leader snapshot "
                            "in MOVE_J at 10% speed, joint target rate <= 5 deg/s; "
                            f"initial gap={math.degrees(duration):.2f} deg, "
                            f"target ramp >= {duration / ALIGNMENT_RATE_RAD_S:.1f} s; "
                            "f cancels. Wait for completion before following."
                        )
                elif key == "f":
                    if not leader_drag:
                        print("\n[leader-follower] press l before enabling following.")
                    elif alignment_target is not None:
                        alignment_target = None
                        follower_joints, age = read_joint_message(
                            follower, follower_config, "follower"
                        )
                        if age > args.stale_timeout:
                            raise RuntimeError("cannot hold with stale follower feedback")
                        command = clip_command_to_bounds(follower_joints, bounds)
                        follower.move_j(command)
                        last_sent_joints = list(command)
                        print("\n[leader-follower] alignment cancelled; follower holds position.")
                    elif following:
                        following = False
                        if recorder is not None and recorder.active:
                            recorder.mark_incomplete()
                        print("\n[leader-follower] following paused; follower holds position.")
                    else:
                        leader_joints, _ = update_leader_joint_message(
                            leader, leader_joints, leader_config
                        )
                        follower_joints, follower_joint_age = read_joint_message(
                            follower, follower_config, "follower"
                        )
                        _, follower_gripper_age = read_gripper_message(
                            follower_gripper, "follower"
                        )
                        if (
                            follower_joint_age > args.stale_timeout
                            or follower_gripper_age > args.stale_timeout
                        ):
                            raise RuntimeError(
                                "cannot start with stale follower feedback: "
                                f"joint={follower_joint_age:.3f}s, "
                                f"gripper={follower_gripper_age:.3f}s"
                            )
                        requested = map_leader_joints(
                            leader_joints, args.joint_signs, args.joint_offsets_deg
                        )
                        requested = clip_feedback_zero_offsets(requested, follower_config)
                        requested = prepare_leader_target(requested, bounds, args.control_only)
                        errors = [a - b for a, b in zip(requested, follower_joints)]
                        if max(abs(value) for value in errors) > alignment_limit:
                            text = ", ".join(
                                f"J{i}={math.degrees(value):+.1f}deg"
                                for i, value in enumerate(errors, 1)
                            )
                            print(
                                "\n[leader-follower] startup alignment rejected: "
                                f"{text}; limit={math.degrees(alignment_limit):.1f}deg; press a to align"
                            )
                        else:
                            command = clip_command_to_bounds(follower_joints, bounds)
                            command_gripper = min(
                                max(follower_gripper_width, 0.0), gripper_max
                            )
                            # The SDK speed setter transmits move_mode=0xFF.
                            # Finish initialization with an explicit JS mode frame,
                            # as the gamepad path does; never leave 0xFF last.
                            follower.set_speed_percent(args.speed)
                            enter_follower_js(
                                follower, follower_gripper, command,
                                follower_gripper_width, args.gripper_force,
                                follower_enabled,
                            )
                            follower_enabled = True
                            follower_js_mode = True
                            ensure_receive_running(leader, "leader")
                            ensure_receive_running(follower, "follower")
                            following = True
                            print(
                                f"\n[leader-follower] following active: "
                                f"leader={args.leader_channel} (drag), "
                                f"follower={args.follower_channel} (MOVE_JS). "
                                "Press s for read-only diagnostics."
                            )
                elif key == "r":
                    if recorder is None:
                        print("\n[leader-follower] recording disabled in control-only mode.")
                    elif recorder.active:
                        episode_ready_for_home = recorder.save_episode() is not None
                        if episode_ready_for_home:
                            print("\n[leader-follower] episode saved; press h to home both arms.")
                    elif following:
                        recorder.start_episode()
                        episode_ready_for_home = False
                    else:
                        print("\n[leader-follower] start following before recording.")
                elif key == "d":
                    if recorder is not None and recorder.active:
                        recorder.discard_episode()
                        episode_ready_for_home = False
                    else:
                        print("\n[leader-follower] no active episode to discard.")
                elif key == "h":
                    if recorder is not None and recorder.active:
                        print("\n[leader-follower] save the current episode with r before homing.")
                    elif alignment_target is not None:
                        print("\n[leader-follower] press f to cancel alignment before homing.")
                    else:
                        # Stop streaming before changing the leader's role. Never
                        # let a moving leader drag arm remain free while the
                        # follower is homing.
                        follower_joints, follower_joint_age = read_joint_message(
                            follower, follower_config, "follower"
                        )
                        if follower_joint_age > args.stale_timeout:
                            raise RuntimeError("cannot home with stale follower feedback")
                        base.ensure_arm_status_normal(follower)
                        follower_hold = clip_command_to_bounds(follower_joints, bounds)
                        if follower_enabled:
                            if follower_js_mode:
                                follower.move_js(follower_hold)
                            else:
                                follower.move_j(follower_hold)
                        following = False
                        command = follower_hold
                        last_sent_joints = None
                        # The local drag flag reflects observed input frames, not
                        # necessarily the controller's persisted leader/follower
                        # role. An immediate h must select controlled mode too.
                        if leader_drag:
                            leader_joints, _ = update_leader_joint_message(
                                leader, leader_joints, leader_config
                            )
                        else:
                            leader_joints, leader_joint_age = read_joint_message(
                                leader, leader_config, "leader"
                            )
                            if leader_joint_age > args.stale_timeout:
                                raise RuntimeError(
                                    "cannot home with stale leader feedback"
                                )
                        arm_home.validate_joint_values(
                            leader_joints, leader_config,
                            arm_home.DEFAULT_JOINT_LIMIT_TOLERANCE_DEG,
                        )
                        base.ensure_arm_status_normal(leader)
                        if not math.isfinite(leader_gripper_width):
                            raise RuntimeError("cannot home with invalid leader gripper width")
                        leader_needs_motion = max(abs(value) for value in leader_joints) > math.radians(3.0)
                        if not leader_drag and leader_needs_motion:
                            print(
                                "\n[leader-follower] entering leader zero-force "
                                "mode for native home; support the leader arm."
                            )
                            leader.set_leader_mode()
                            leader_drag = True
                            time.sleep(0.1)
                        print(
                            "\n[leader-follower] following paused; homing leader "
                            "with firmware 0x191 at controller-defined speed, "
                            "then follower with MOVE_J at 10%; both grippers close slowly."
                        )
                        def home_stop_requested() -> bool:
                            return keys.poll() in ("x", "q")
                        def mark_homing_estop() -> None:
                            nonlocal emergency_stop
                            emergency_stop = True
                        try:
                            leader_joints = home_both_sequentially(
                                follower, follower_config, args.follower_channel,
                                leader, leader_config, args.leader_channel,
                                follower_gripper, leader_gripper,
                                args.gripper_force, args.home_gripper_speed_mm_s,
                                args.home_timeout, home_stop_requested,
                                leader_joints, leader_gripper_width,
                                on_estop=mark_homing_estop,
                            )
                        except BaseException:
                            if emergency_stop:
                                # Helper held the leader, or E-stopped the
                                # active arm; do not command it in cleanup.
                                leader_drag = False
                            raise
                        leader_drag = False
                        leader_gripper_width = 0.0
                        follower_enabled = True
                        follower_js_mode = False
                        follower_joints, _ = read_joint_message(
                            follower, follower_config, "follower"
                        )
                        command = follower_joints[:]
                        episode_ready_for_home = False
                        print(
                            "[leader-follower] both arms homed and holding enabled; "
                            "press l, a, f to resume following."
                        )
                elif key == "q":
                    print("\n[leader-follower] clean exit; follower holds position.")
                    break
                elif key == "x":
                    follower.electronic_emergency_stop()
                    emergency_stop = True
                    if recorder is not None and recorder.active:
                        recorder.mark_incomplete()
                    print("\n[leader-follower] follower emergency stop sent.")
                    break

                if leader_drag:
                    leader_joints, leader_joint_age = update_leader_joint_message(
                        leader, leader_joints, leader_config
                    )
                    leader_width, leader_gripper_age = read_leader_gripper_command(
                        leader_gripper, leader_gripper_width
                    )
                else:
                    leader_joints, leader_joint_age = read_joint_message(
                        leader, leader_config, "leader"
                    )
                    leader_width, leader_gripper_age = read_gripper_message(
                        leader_gripper, "leader"
                    )
                follower_joints, follower_joint_age = read_joint_message(
                    follower, follower_config, "follower"
                )
                follower_width, follower_gripper_age = read_gripper_message(
                    follower_gripper, "follower"
                )
                if (
                    follower_joint_age > args.stale_timeout
                    or follower_gripper_age > args.stale_timeout
                ):
                    raise RuntimeError(
                        "stale follower CAN feedback: "
                        f"joint={follower_joint_age:.3f}s, "
                        f"gripper={follower_gripper_age:.3f}s"
                    )
                if not leader_drag and (
                    leader_joint_age > args.stale_timeout
                    or leader_gripper_age > args.stale_timeout
                ):
                    raise RuntimeError(
                        "stale leader feedback before zero-gravity mode: "
                        f"joint={leader_joint_age:.3f}s, "
                        f"gripper={leader_gripper_age:.3f}s"
                    )
                leader_gripper_width = leader_width
                follower_gripper_width = follower_width
                requested = map_leader_joints(
                    leader_joints, args.joint_signs, args.joint_offsets_deg
                )
                requested = clip_feedback_zero_offsets(requested, follower_config)
                bounded_requested = prepare_leader_target(requested, bounds, args.control_only)
                limited = any(abs(a - b) > 1e-9 for a, b in zip(requested, bounded_requested))
                if limited != target_limit_active:
                    target_limit_active = limited
                    status_dirty = True
                    if limited:
                        print("\n[限位] 目标已自动限幅，录制不受影响；将主臂拖回范围内即可恢复正常跟随，s 查看详情。")
                    else:
                        print("\n[限位] 主臂目标已回到允许范围，正常映射已恢复。")

                if following or alignment_target is not None:
                    status = follower.get_arm_status()
                    if (
                        status is not None
                        and base.value_as_int(status.msg.arm_status) != 0
                    ):
                        raise RuntimeError(
                            "follower controller reported an abnormal status"
                        )
                    assert command is not None
                    motion_target = bounded_requested if alignment_target is None else alignment_target
                    motion_step = max_step if alignment_target is None else min(
                        max_step, ALIGNMENT_RATE_RAD_S * period
                    )
                    if alignment_target is not None and tick > alignment_deadline:
                        raise RuntimeError("slow alignment timed out; follower will hold")
                    command = limit_joint_target(
                        motion_target, command, follower_joints, motion_step, max_error
                    )
                    validate_target_bounds(command, bounds)
                    if alignment_target is not None:
                        follower.move_j(command)
                    else:
                        follower.move_js(command)
                    last_sent_joints = list(command)
                    if abs(command_gripper - follower_width) > max_gripper_error:
                        raise RuntimeError(
                            "follower gripper exceeded the maximum following error"
                        )
                    if following and tick >= next_gripper_command:
                        desired_gripper = min(max(leader_width, 0.0), gripper_max)
                        gripper_delta = max(
                            -gripper_step,
                            min(gripper_step, desired_gripper - command_gripper),
                        )
                        command_gripper += gripper_delta
                        if gripper_delta:
                            follower_gripper.move_gripper_m(
                                command_gripper, args.gripper_force
                            )
                        next_gripper_command = tick + 0.05
                    if alignment_target is not None:
                        actual_error = max(
                            abs(t - a) for t, a in zip(alignment_target, follower_joints)
                        )
                        if actual_error <= alignment_best_error - math.radians(0.5):
                            alignment_best_error = actual_error
                            alignment_last_progress_at = tick
                        command_gap = max(
                            abs(c - a) for c, a in zip(command, follower_joints)
                        )
                        if (tick - alignment_last_progress_at >= 5.0
                                and command_gap >= math.radians(3.0)
                                and actual_error > ALIGNMENT_TOLERANCE_RAD):
                            mode = getattr(status.msg, "mode_feedback", "unavailable") if status else "unavailable"
                            enabled = follower.get_joints_enable_status_list()
                            raise RuntimeError(
                                "follower alignment did not move for 5 s; "
                                f"command_minus_actual={math.degrees(command_gap):.2f} deg, "
                                f"mode={mode}, enabled={enabled}; follower will hold"
                            )
                        errors_deg = [
                            math.degrees(t - a)
                            for t, a in zip(alignment_target, follower_joints)
                        ]
                        if alignment_complete(alignment_target, command, follower_joints):
                            if alignment_settled_since is None:
                                alignment_settled_since = tick
                        else:
                            alignment_settled_since = None
                        if (
                            alignment_settled_since is not None
                            and tick - alignment_settled_since >= ALIGNMENT_SETTLE_SECONDS
                        ):
                            alignment_target = None
                            status_dirty = True
                            print(
                                "\n[leader-follower] alignment complete "
                                "(all joints within 2 deg for 0.5s); holding. Press f to follow."
                            )
                        else:
                            alignment_remaining_deg = max(abs(value) for value in errors_deg)
                else:
                    command = clip_command_to_bounds(follower_joints, bounds)
                    command_gripper = follower_width

                following_error = [
                    target - actual for target, actual in zip(command, follower_joints)
                ] + [command_gripper - follower_width]
                if recorder is not None:
                    recorder.capture(
                        time.monotonic(),
                        [*follower_joints, follower_width],
                        [*command, command_gripper],
                        list(follower.fk(follower_joints)),
                        [
                            *(
                                target - actual
                                for target, actual in zip(requested, follower_joints)
                            )
                        ],
                        {
                            "leader_state": [*leader_joints, leader_width],
                            "follower_state": [*follower_joints, follower_width],
                            "following_error": following_error,
                            "leader_feedback_age": finite_combined_age(
                                leader_joint_age, leader_gripper_age
                            ),
                            "follower_feedback_age": finite_combined_age(
                                follower_joint_age, follower_gripper_age
                            ),
                        },
                    )
                if key == "s":
                    motion_diagnostics.emit(
                        tick, args, leader, follower, leader_joints, requested,
                        last_sent_joints, follower_joints,
                        leader_joint_age, follower_joint_age,
                        following, alignment_target is not None, force=True,
                        bounded_target=bounded_requested,
                    )
                if status_dirty or key in ("l", "a", "f", "r", "d", "h", "s") or tick - last_status_print >= 1.0:
                    text = status_line(
                        following, alignment_target is not None, leader_drag,
                        recorder is not None and recorder.active,
                        episode_ready_for_home, alignment_remaining_deg,
                    )
                    print(f"\r{text:<90}", end="", flush=True)
                    last_status_print = tick
                remaining = period - (time.monotonic() - tick)
                if remaining > 0:
                    time.sleep(remaining)
        return 0
    except KeyboardInterrupt:
        if emergency_stop:
            print(
                "\n[leader-follower] interrupted during homing; the active arm "
                "received an electronic emergency stop."
            )
        else:
            print("\n[leader-follower] interrupted; follower will hold position.")
        return 130
    except Exception as exc:
        cleanup = (
            "Homing aborted; electronic emergency-stop handling ran for the "
            "active arm. No following is active after exit."
            if emergency_stop else
            "Starting exit cleanup: follower will hold; leader will leave drag "
            "mode and hold enabled. No following is active after exit."
        )
        print(
            f"\n[leader-follower] control stopped: {type(exc).__name__}: {exc}. "
            f"{cleanup}",
            flush=True,
        )
        if recorder is not None and recorder.active:
            recorder.mark_incomplete()
        raise
    finally:
        if (
            follower is not None
            and command is not None
            and follower_enabled
            and not emergency_stop
        ):
            try:
                latest = follower.get_joint_angles()
                hold = clip_command_to_bounds(
                    list(latest.msg) if latest is not None else command, bounds
                )
                if follower_js_mode:
                    follower.move_js(hold)
                else:
                    follower.move_j(hold)
            except Exception:
                pass
        if leader is not None and leader_drag and leader_joints is not None:
            try:
                exit_leader_drag_mode(
                    leader,
                    leader_joints,
                    leader_gripper,
                    leader_gripper_width,
                    args.gripper_force,
                    config=leader_config,
                )
                print(
                    "[leader-follower] leader left zero-force mode and is "
                    "holding its current position while enabled."
                )
            except Exception as exc:
                print(
                    "[leader-follower] WARNING: failed to leave leader "
                    f"zero-force mode: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
        if recorder is not None:
            recorder.close()
        base.disconnect_quietly(follower)
        base.disconnect_quietly(leader)


def run() -> int:
    try:
        return main()
    except (ValueError, RuntimeError, TimeoutError) as exc:
        print(f"\n[leader-follower] failed: {type(exc).__name__}: {exc}", flush=True)
        return 1
    except Exception as exc:
        print(f"\n[leader-follower] unexpected failure: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
