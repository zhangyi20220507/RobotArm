"""手柄数采运行时：关节 Jog / 末端 DLS 遥操作 + episode 录制。

迁移自 piper_ctrl/control_scripts/09_gamepad_dls_data_collection.py，
行为与参数保持兼容。

用法：
  手柄诊断（不连接机械臂）：
    python3 -m robotarm.teleop.gamepad.runtime --gamepad-monitor
  实际采集：
    python3 -m robotarm.teleop.gamepad.runtime --execute \
        --setup-id table_left_v1 --task "Pick up the block"

映射：
  关节模式：左摇杆左右（反向）/上下 = J1/J2；右摇杆上下（反向）/左右 =
  J3/J5；十字键左右/上下 = J4/J6。末端模式：左摇杆 = X/Y；右摇杆
  上下/左右 = Z/Yaw；十字键左右/上下 = Roll/Pitch。
  A/Cross = 切换关节/末端模式。
  B/Circle 短按并松开 = 开始/保存 episode；录制中长按 B/Circle = 丢弃 episode。
  LT/RT = 夹爪闭合/打开；LB/RB = 降低/提高当前模式 Jog 速度。
  长按 X/Square = 开始/暂停控制；长按 Start/Options = 六关节与夹爪一起使能；
  长按 Back/Share = 六关节与夹爪一起失能；
  长按 Y/Triangle = 关节与夹爪直接回零；L3+R3 = 电子急停并退出。

关节模式采用关节增量 Jog，末端模式使用本地 DLS 逆解；两者均通过 MOVE_JS
控制。数据保存为 MP4、NPZ、JSON，不要求本机安装 LeRobot。
"""

from __future__ import annotations

import argparse
import math
import time
import traceback
from pathlib import Path
from typing import Any

from ...core.camera.realsense import RoleCameras
from ...core.kinematics.dls_ik import IKError
from ...core.robot import piper as base
from ...core.robot.safety import (
    ensure_startup_status,
    read_gripper_feedback,
    safe_joint_bounds,
    validate_joint_feedback,
)
from ...record.recorder import RawEpisodeRecorder
from ...paths import PIPER_RAW_ROOT
from ...config import defaults
from . import device as common
from .inputs import cartesian_inputs, joint_inputs
from .targets import (
    build_joint_targets,
    cartesian_pose,
    interpolate_cartesian_joints,
    jog_step_levels,
    solve_cartesian_target,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    cfg = defaults("data_acquisition", "collect-piper")
    parser = argparse.ArgumentParser(description=__doc__)
    base.add_connection_arguments(parser, cfg)
    base.add_execute_argument(parser)
    parser.add_argument(
        "--step-deg", type=base.positive_float, default=cfg["step-deg"],
        help="full-deflection joint increment per update in degrees",
    )
    parser.add_argument(
        "--max-lead-deg", type=base.positive_float, default=cfg["max-lead-deg"],
        help="maximum command lead over feedback per joint",
    )
    parser.add_argument("--xy-step-mm", type=base.positive_float, default=cfg["xy-step-mm"])
    parser.add_argument("--z-step-mm", type=base.positive_float, default=cfg["z-step-mm"])
    parser.add_argument("--angle-step-deg", type=base.positive_float, default=cfg["angle-step-deg"])
    parser.add_argument(
        "--max-position-lead-mm", type=base.positive_float,
        default=cfg["max-position-lead-mm"],
    )
    parser.add_argument(
        "--max-angle-lead-deg", type=base.positive_float,
        default=cfg["max-angle-lead-deg"],
    )
    parser.add_argument(
        "--max-joint-command-deg", type=base.positive_float,
        default=cfg["max-joint-command-deg"],
    )
    parser.add_argument(
        "--cartesian-smoothing", type=float, default=cfg["cartesian-smoothing"],
        help="Cartesian joint-command low-pass factor, 0..1",
    )
    parser.add_argument(
        "--tcp-offset-mm", type=float, default=cfg["tcp-offset-mm"],
        help="controlled gripper TCP offset along flange local Z in mm",
    )
    parser.add_argument("--ik-backend", choices=("dls", "placo"), default="dls",
                        help="Cartesian IK backend (Placo is opt-in; XR is unchanged)")
    parser.add_argument(
        "--preferred-joints-deg", type=float, nargs=6,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        default=cfg["preferred-joints-deg"],
        help="soft DLS reference posture in degrees",
    )
    parser.add_argument(
        "--speed", type=base.percentage, default=cfg["speed"],
        help="controller speed for MOVE_J homing; Jog uses software steps",
    )
    parser.add_argument(
        "--home-timeout", type=base.positive_float, default=cfg["home-timeout"],
        help="joint and gripper homing timeout in seconds",
    )
    parser.add_argument(
        "--gripper-step-mm", type=base.positive_float, default=cfg["gripper-step-mm"],
        help="gripper increment per update in mm",
    )
    parser.add_argument(
        "--gripper-max-mm", type=base.positive_float, default=cfg["gripper-max-mm"],
        help="maximum opening in mm (use 70 for small gripper)",
    )
    parser.add_argument(
        "--gripper-force", type=base.positive_float, default=cfg["gripper-force"],
        help="gripper force in newtons",
    )
    parser.add_argument(
        "--gripper-home-speed-mm-s", type=base.positive_float,
        default=cfg["gripper-home-speed-mm-s"],
        help="gripper closing speed during homing in mm/s",
    )
    parser.add_argument(
        "--allow-enabled-start", action="store_true",
        help="allow startup when all six joints are already enabled",
    )
    parser.add_argument(
        "--gamepad-monitor", action="store_true",
        help="show raw/parsed controller inputs without connecting to the robot",
    )
    parser.add_argument(
        "--gamepad-connection", choices=("wired", "bluetooth"),
        default=None,
        help="connection type; if omitted, choose interactively at startup",
    )
    parser.add_argument("--global-camera-serial", default=cfg["global-camera-serial"])
    parser.add_argument("--wrist-camera-serial", default=cfg["wrist-camera-serial"])
    parser.add_argument("--record-width", type=int, default=cfg["record-width"])
    parser.add_argument("--record-height", type=int, default=cfg["record-height"])
    parser.add_argument("--record-fps", type=int, default=cfg["record-fps"])
    parser.add_argument("--encoder-queue-size", type=int, default=cfg["encoder-queue-size"])
    parser.add_argument(
        "--dataset-root", type=Path, default=PIPER_RAW_ROOT,
        help="raw session 根目录（默认 $ROBOTARM_DATA_ROOT/raw/piper_raw）",
    )
    parser.add_argument("--setup-id", default=None)
    parser.add_argument("--setup-note", default="")
    parser.add_argument("--task", default=None)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument(
        "--camera-check", action="store_true",
        help="check configured cameras and preview without connecting to the arm",
    )
    common.add_gamepad_arguments(parser, cfg)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.step_deg > 5.0:
        raise SystemExit("--step-deg must not exceed 5 degrees")
    if args.max_lead_deg < args.step_deg:
        raise SystemExit("--max-lead-deg must be >= --step-deg")
    if args.max_lead_deg > 20.0:
        raise SystemExit("--max-lead-deg must not exceed 20 degrees")
    if not 0.0 <= args.deadzone <= 0.5:
        raise SystemExit("--deadzone must be between 0 and 0.5")
    if not 0.0 <= args.trigger_deadzone <= 0.5:
        raise SystemExit("--trigger-deadzone must be between 0 and 0.5")
    if not 5.0 <= args.control_rate <= 100.0:
        raise SystemExit("--control-rate must be between 5 and 100 Hz")
    if not 0.3 <= args.hold_seconds <= 5.0:
        raise SystemExit("--hold-seconds must be between 0.3 and 5 seconds")
    if args.gripper_step_mm > 10 or args.gripper_max_mm > 100:
        raise SystemExit("gripper step/max exceeds safety limit")
    if args.gripper_force > 5 or args.gripper_home_speed_mm_s > 50:
        raise SystemExit("gripper force/home speed exceeds safety limit")
    if args.max_position_lead_mm > 100 or args.max_angle_lead_deg > 20:
        raise SystemExit("Cartesian target lead exceeds safety limit")
    if args.max_joint_command_deg > 20:
        raise SystemExit("--max-joint-command-deg must not exceed 20")
    if not 0.05 <= args.cartesian_smoothing <= 1.0:
        raise SystemExit("--cartesian-smoothing must be between 0.05 and 1")
    if (
        not math.isfinite(args.tcp_offset_mm)
        or not 0.0 <= args.tcp_offset_mm <= 300.0
    ):
        raise SystemExit("--tcp-offset-mm must be finite and between 0 and 300")
    if args.record_width <= 0 or args.record_height <= 0:
        raise SystemExit("record width and height must be positive")
    if not 1 <= args.record_fps <= args.control_rate:
        raise SystemExit("--record-fps must be between 1 and --control-rate")
    if not 10 <= args.encoder_queue_size <= 600:
        raise SystemExit("--encoder-queue-size must be between 10 and 600")
    if args.wrist_camera_serial == args.global_camera_serial:
        raise SystemExit("global and wrist camera serial numbers must differ")
    if args.execute and not (args.camera_check or args.gamepad_monitor) and (not args.setup_id or not args.task):
        raise SystemExit("--execute requires non-empty --setup-id and --task")


def initialize_gripper_target(feedback_width: float, command_max: float) -> float:
    """Validate raw feedback and clamp only the software command accumulator."""
    feedback_max = 0.150  # Piper SDK filters gripper feedback above 150 mm.
    tolerance = 0.002
    if not math.isfinite(feedback_width):
        raise RuntimeError("夹爪反馈不是有效数值。")
    if feedback_width < -tolerance or feedback_width > feedback_max + tolerance:
        raise RuntimeError(
            f"夹爪反馈 {feedback_width * 1000:.2f} mm 超出 SDK 合理范围 "
            f"0..{feedback_max * 1000:.0f} mm。"
        )
    target = min(max(feedback_width, 0.0), command_max)
    if not math.isclose(target, feedback_width, abs_tol=1e-6):
        print(
            f"[gamepad] 夹爪当前反馈 {feedback_width * 1000:.2f} mm，"
            f"软件命令目标已限制到 0..{command_max * 1000:.0f} mm。"
        )
    else:
        print(f"[gamepad] 夹爪当前反馈：{feedback_width * 1000:.2f} mm。")
    return target


def read_and_validate_joints(robot: Any, config: dict[str, Any]) -> list[float]:
    latest = robot.get_joint_angles()
    if latest is None:
        raise RuntimeError("未取得最新关节反馈。")
    joints = list(latest.msg)
    validate_joint_feedback(joints, config)
    return joints


def enter_js_mode(
    robot: Any, config: dict[str, Any], enable_first: bool
) -> list[float]:
    joints = read_and_validate_joints(robot, config)
    robot.move_j(joints)
    robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.J)
    robot.move_j(joints)
    if enable_first:
        base.retry_boolean(robot.enable, 5.0, "Enable")
    time.sleep(0.1)
    joints = read_and_validate_joints(robot, config)
    robot.move_j(joints)
    robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.JS)
    robot.move_js(joints)
    return joints


def home_with_gamepad_estop(
    gamepad: common.GamepadReader,
    robot: Any,
    gripper: Any,
    gripper_force: float,
    config: dict[str, Any],
    timeout: float,
    gripper_home_speed_mm_s: float,
    poll_period: float,
) -> list[float]:
    started = time.monotonic()
    deadline = started + timeout
    joint_tolerance = math.radians(3.0)
    gripper_tolerance = 0.002
    gripper_start = base.wait_gripper_width(gripper, min(timeout, 2.0))
    gripper_command = max(0.0, gripper_start)
    closing_speed = gripper_home_speed_mm_s / 1000.0
    next_gripper_command = started
    target = [0.0] * 6
    robot.move_j(target)
    robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.J)
    robot.move_j(target)
    print(
        f"[gamepad] 正在使用 MOVE_J 回零，夹爪以约 "
        f"{gripper_home_speed_mm_s:.1f} mm/s 闭合；L3+R3 可急停。"
    )
    try:
        while time.monotonic() < deadline:
            tick_started = time.monotonic()
            pad = gamepad.read()
            buttons = int(pad.buttons)
            if (buttons & common.L3) and (buttons & common.R3):
                robot.electronic_emergency_stop()
                raise RuntimeError("回零过程中收到手柄电子急停，程序退出。")

            now = time.monotonic()
            if now >= next_gripper_command:
                next_width = max(
                    0.0, gripper_start - closing_speed * (now - started)
                )
                if next_width < gripper_command:
                    gripper.move_gripper_m(next_width, gripper_force)
                    gripper_command = next_width
                next_gripper_command = now + 0.05

            joints = read_and_validate_joints(robot, config)
            status = robot.get_arm_status()
            gripper_status = gripper.get_gripper_status()
            gripper_width = (
                float(gripper_status.msg.value)
                if gripper_status is not None
                and gripper_status.msg.mode == "width"
                else None
            )
            max_error = max(abs(value) for value in joints)
            gripper_text = (
                f"{gripper_width * 1000:5.1f} mm"
                if gripper_width is not None else "等待反馈"
            )
            print(
                f"\r回零中：关节误差 {math.degrees(max_error):6.2f}°，"
                f"夹爪 {gripper_text}      ", end="", flush=True,
            )
            done = (
                status is not None
                and base.value_as_int(status.msg.motion_status) == 0
                and now - started >= 0.5
                and max_error <= joint_tolerance
                and gripper_width is not None
                and gripper_width <= gripper_tolerance
                and gripper_command <= 0.0
            )
            if done:
                robot.move_j(joints)
                robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.JS)
                robot.move_js(joints)
                print("\n[gamepad] 六关节和夹爪已回零，已恢复 JS 保持模式。")
                return joints
            sleep_remainder(tick_started, poll_period)
    except Exception:
        robot.electronic_emergency_stop()
        raise
    robot.electronic_emergency_stop()
    raise TimeoutError(f"回零在 {timeout:.1f} 秒内未完成，已发送电子急停。")


def print_controls() -> None:
    print()
    print("Joint Jog / Cartesian IK data-collection controls:")
    print("  Joint: left stick J1/J2, right stick J3/J5, D-pad J4/J6")
    print("  Cartesian: left stick X/Y, right stick Yaw/Z, D-pad Roll/Pitch")
    print("  A/Cross: switch Joint / Cartesian")
    print("  Short B/Circle: start/save episode    Hold B/Circle: discard")
    print("  LT: close gripper    RT: open gripper    LB/RB: decrease/increase Jog")
    print("  Hold X/Square: run/pause    Hold Start/Options: enable")
    print("  Hold Back/Share: disable    Hold Y/Triangle: home")
    print("  L3+R3: emergency stop")
    print()


def monitor_gamepad(gamepad: common.GamepadReader, args: argparse.Namespace) -> int:
    print("[gamepad] 手柄诊断模式，不连接机械臂。按 Ctrl+C 退出。")
    print("原始轴与末端解析：X/Y/Z/Roll/Pitch/Yaw")
    period = 1.0 / min(args.control_rate, 30.0)
    try:
        while True:
            started = time.monotonic()
            pad = gamepad.read()
            axes = cartesian_inputs(pad, args.deadzone)
            names = common.button_names(int(pad.buttons))
            print(
                f"\rraw LX={int(pad.left_x):6d} LY={int(pad.left_y):6d} "
                f"RX={int(pad.right_x):6d} RY={int(pad.right_y):6d} "
                f"buttons=0x{int(pad.buttons):04X} keys={names:<38.38} | "
                f"XYZ={axes[0]:+.2f}/{axes[1]:+.2f}/{axes[2]:+.2f} "
                f"RPY={axes[3]:+.0f}/{axes[4]:+.0f}/{axes[5]:+.2f}       ",
                end="", flush=True,
            )
            sleep_remainder(started, period)
    except KeyboardInterrupt:
        print("\n[gamepad] 手柄诊断结束。")
        return 0


def sleep_remainder(started: float, period: float) -> None:
    remaining = period - (time.monotonic() - started)
    if remaining > 0:
        time.sleep(remaining)


def choose_gamepad_connection(requested: str | None) -> str:
    if requested is not None:
        return requested
    print("[gamepad] Select gamepad connection:")
    print("  1: Wired / USB")
    print("  2: Bluetooth")
    while True:
        choice = input("Enter 1 or 2: ").strip().lower()
        if choice in ("1", "wired", "usb"):
            return "wired"
        if choice in ("2", "bluetooth", "bt"):
            return "bluetooth"
        print("Invalid selection; enter 1 for wired or 2 for Bluetooth.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    camera_serials = {"global": args.global_camera_serial}
    if args.wrist_camera_serial:
        camera_serials["wrist"] = args.wrist_camera_serial
    if args.camera_check:
        cameras = RoleCameras(
            camera_serials, args.record_width, args.record_height,
            args.record_fps, not args.no_preview, always_on_top=True,
        )
        try:
            cameras.wait_ready()
            print("[gamepad] Camera check passed; press Ctrl+C to close the preview.")
            while True:
                frames = cameras.latest()
                cameras.show(frames, False)
                time.sleep(0.03)
        except KeyboardInterrupt:
            print("\n[gamepad] Camera check finished.")
            return 0
        finally:
            cameras.close()

    gamepad_connection = choose_gamepad_connection(args.gamepad_connection)
    gamepad = common.GamepadReader(args.gamepad_index, gamepad_connection)
    pad = gamepad.read()
    print(
        f"[gamepad] 已连接 {gamepad_connection} 手柄 {args.gamepad_index}："
        f"buttons=0x{int(pad.buttons):04X}。"
    )
    if args.gamepad_monitor:
        return monitor_gamepad(gamepad, args)
    if not args.execute:
        print(
            f"DRY-RUN: Joint/{args.ik_backend} {args.control_rate:.0f} Hz，recording "
            f"{args.record_fps} FPS，cameras={camera_serials}."
        )
        print("添加 --execute、--setup-id 和 --task 后才会连接机械臂与相机。")
        print_controls()
        return 0

    ik_solver = None
    if args.ik_backend == "placo":
        try:
            from .placo_ik import GamepadPlacoIK
            ik_solver = GamepadPlacoIK(1.0 / args.control_rate, args.tcp_offset_mm / 1000.0)
        except ImportError as exc:
            raise SystemExit("Placo backend requires placo, numpy and scipy in this environment") from exc
    print(f"[gamepad] IK backend={args.ik_backend}; TCP={args.tcp_offset_mm:.1f} mm")
    robot = None
    recorder = None
    try:
        print(
            f"[gamepad] 正在连接机械臂：interface={args.interface}, "
            f"channel={args.channel}, firmware={args.firmware}……",
            flush=True,
        )
        robot, config = base.connect_robot(args)
        joints = list(base.wait_for_feedback(robot, args.feedback_timeout).msg)
        validate_joint_feedback(joints, config)
        ensure_startup_status(robot, args.feedback_timeout)
        gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        gripper_feedback_width = base.wait_gripper_width(
            gripper, args.feedback_timeout
        )
        gripper_max = args.gripper_max_mm / 1000.0
        gripper_width = initialize_gripper_target(
            gripper_feedback_width, gripper_max
        )

        arm_enabled = base.get_all_joints_enabled_state(
            robot, args.feedback_timeout
        )
        if arm_enabled and not args.allow_enabled_start:
            raise RuntimeError(
                "检测到六关节已使能；请确认后添加 --allow-enabled-start。"
            )
        robot.set_auto_set_motion_mode_enabled(False)
        robot.set_joint_limits_enabled(False)
        robot.move_j(joints)
        robot.set_speed_percent(args.speed)

        recorder = RawEpisodeRecorder(
            args.dataset_root, args.task, args.setup_id, args.setup_note,
            args.record_fps, args.record_width, args.record_height,
            camera_serials,
            {
                "model": "Piper",
                "firmware": args.firmware,
                "interface": args.interface,
                "channel": args.channel,
                "control_mode": f"switchable_joint_jog_or_cartesian_{args.ik_backend}_to_move_js",
                "control_rate_hz": args.control_rate,
                "joint_step_deg": args.step_deg,
                "preferred_joints_deg": list(args.preferred_joints_deg),
                "cartesian_tcp_offset_m": args.tcp_offset_mm / 1000.0,
                "ik_backend": args.ik_backend,
                "joint_frame": "piper_base",
                "tcp_pose_units": ["m", "m", "m", "rad", "rad", "rad"],
            },
            preview=not args.no_preview,
            encoder_queue_size=args.encoder_queue_size,
        )

        bounds = safe_joint_bounds(config)
        preferred_joints = [
            math.radians(value) for value in args.preferred_joints_deg
        ]
        for index, (value, (low, high)) in enumerate(
            zip(preferred_joints, bounds), start=1
        ):
            if not low <= value <= high:
                raise ValueError(f"DLS 参考姿态 J{index} 超出安全范围。")
        max_lead = math.radians(args.max_lead_deg)
        targets = joints[:]
        control_active = False
        step_levels_deg, step_index = jog_step_levels(
            args.step_deg, args.max_lead_deg
        )
        current_step_deg = step_levels_deg[step_index]
        cartesian_speed_levels = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
        cartesian_speed_index = 2
        control_mode = "cartesian"
        tcp_offset = args.tcp_offset_mm / 1000.0
        pose_target = cartesian_pose(robot, joints, tcp_offset)
        rate_scale = 20.0 / args.control_rate
        cartesian_smoothing = 1.0 - (1.0 - args.cartesian_smoothing) ** rate_scale
        position_steps = [
            args.xy_step_mm / 1000.0 * rate_scale,
            args.xy_step_mm / 1000.0 * rate_scale,
            args.z_step_mm / 1000.0 * rate_scale,
        ]
        angle_step = math.radians(args.angle_step_deg) * rate_scale
        max_position_lead = args.max_position_lead_mm / 1000.0
        max_angle_lead = math.radians(args.max_angle_lead_deg)
        max_joint_command = math.radians(args.max_joint_command_deg)
        tracker = common.ButtonTracker()
        previous_buttons = 0
        poll_period = 1.0 / args.control_rate
        ik_rejections = 0
        diagnostic_started = time.monotonic()
        diagnostic_cycles = 0
        diagnostic_max_ms = 0.0
        diagnostic_ik_ms = 0.0
        diagnostic_input_ms = diagnostic_record_ms = diagnostic_control_ms = 0.0
        next_preview = next_status = 0.0
        gripper_step = args.gripper_step_mm / 1000.0
        next_gripper_command = 0.0
        b_pressed_at: float | None = None
        b_long_fired = False
        recorder_broken = False

        print(
            f"[gamepad] 机械臂当前{'已使能' if arm_enabled else '未使能'}；"
            f"默认为末端 {args.ik_backend} 模式且控制暂停，相机预览已启动。"
        )
        print_controls()

        while True:
            tick_started = time.monotonic()
            try:
                pad = gamepad.read()
            except ConnectionError:
                control_active = False
                latest = robot.get_joint_angles()
                if latest is not None and arm_enabled:
                    robot.move_js(list(latest.msg))
                if recorder.active:
                    recorder.mark_incomplete()
                raise RuntimeError("手柄已断开，已停止更新关节目标。")

            now = time.monotonic()
            diagnostic_input_ms = max(diagnostic_input_ms, (now - tick_started) * 1000)
            buttons = int(pad.buttons)
            newly_pressed = buttons & ~previous_buttons
            newly_released = previous_buttons & ~buttons
            tracker.update(buttons, now)

            if (buttons & common.L3) and (buttons & common.R3):
                robot.electronic_emergency_stop()
                if recorder.active:
                    recorder.mark_incomplete()
                print("\n[gamepad] 已发送电子急停；当前 episode 标记为 incomplete。")
                return 0

            if newly_pressed & common.A:
                joints = read_and_validate_joints(robot, config)
                targets = joints[:]
                pose_target = cartesian_pose(robot, joints, tcp_offset)
                control_mode = (
                    "joint" if control_mode == "cartesian" else "cartesian"
                )
                print(
                    f"\n[gamepad] 已切换为"
                    f"{'直接关节' if control_mode == 'joint' else '末端 ' + args.ik_backend}模式。"
                )

            if newly_pressed & common.B:
                b_pressed_at = now
                b_long_fired = False
            if (
                buttons & common.B and b_pressed_at is not None
                and not b_long_fired and now - b_pressed_at >= args.hold_seconds
            ):
                b_long_fired = True
                if recorder.active:
                    recorder.discard_episode()
                else:
                    print("\n[gamepad] No active episode to discard.")
            if newly_released & common.B:
                if not b_long_fired and not recorder_broken:
                    if recorder.active:
                        recorder.save_episode()
                    elif not arm_enabled:
                        print("\n[gamepad] 机械臂未使能，不能开始有效 episode。")
                    else:
                        recorder.start_episode()
                b_pressed_at = None
                b_long_fired = False

            if tracker.long_press(
                buttons, common.START, now, args.hold_seconds
            ):
                if arm_enabled:
                    targets = read_and_validate_joints(robot, config)
                else:
                    targets = enter_js_mode(
                        robot, config, enable_first=True
                    )
                    arm_enabled = True
                base.enable_arm_and_gripper(
                    robot, gripper, gripper_width, args.gripper_force
                )
                control_active = False
                pose_target = cartesian_pose(robot, targets, tcp_offset)
                print("\n[gamepad] 六关节和夹爪已使能；长按 X 后开始控制。")

            if tracker.long_press(
                buttons, common.BACK, now, args.hold_seconds
            ):
                control_active = False
                # Always issue both operations.  This also repairs a mismatched
                # startup state where only the gripper or only the arm is live.
                base.disable_arm_and_gripper(
                    robot, gripper, gripper_width, args.gripper_force
                )
                arm_enabled = False
                if recorder.active:
                    recorder.mark_incomplete()
                print(
                    "\n[gamepad] 六关节和夹爪已失能，机械臂可能失去保持力。"
                )

            if tracker.long_press(buttons, common.X, now, args.hold_seconds):
                if not arm_enabled:
                    control_active = False
                    print("\n[gamepad] 机械臂未使能，不能开启控制。")
                else:
                    if not control_active:
                        targets = enter_js_mode(
                            robot, config, enable_first=False
                        )
                    else:
                        targets = read_and_validate_joints(robot, config)
                        robot.move_js(targets)
                    pose_target = cartesian_pose(robot, targets, tcp_offset)
                    control_active = not control_active
                    print(
                        f"\n[gamepad] {'直接关节' if control_mode == 'joint' else '末端 ' + args.ik_backend}控制："
                        f"{'运行' if control_active else '暂停'}。"
                    )

            if tracker.long_press(buttons, common.Y, now, args.hold_seconds):
                control_active = False
                if recorder.active:
                    recorder.mark_incomplete()
                if not arm_enabled:
                    print("\n[gamepad] 机械臂未使能，不能执行回零。")
                else:
                    print("\n[gamepad] 已收到长按 Y 回零命令，立即开始回零。")
                    targets = home_with_gamepad_estop(
                        gamepad, robot, gripper, args.gripper_force,
                        config, args.home_timeout,
                        args.gripper_home_speed_mm_s, poll_period,
                    )
                    gripper_width = 0.0
                    pose_target = cartesian_pose(robot, targets, tcp_offset)
                    print("[gamepad] 回零完成；关节控制保持暂停。")
                    pad = gamepad.read()
                    buttons = int(pad.buttons)
                    previous_buttons = buttons
                    tracker = common.ButtonTracker()
                    tracker.update(buttons, time.monotonic())
                    if buttons & common.Y:
                        tracker.long_fired.add(common.Y)
                    continue

            if newly_pressed & common.LB:
                if control_mode == "joint":
                    step_index = max(0, step_index - 1)
                    current_step_deg = step_levels_deg[step_index]
                    print(
                        f"\n[gamepad] 关节 Jog 降低：每周期 {current_step_deg:.3f}°。"
                    )
                else:
                    cartesian_speed_index = max(0, cartesian_speed_index - 1)
                    print(
                        f"\n[gamepad] 末端 Jog 降低："
                        f"{cartesian_speed_levels[cartesian_speed_index]:.2f}x。"
                    )
            if newly_pressed & common.RB:
                if control_mode == "joint":
                    step_index = min(len(step_levels_deg) - 1, step_index + 1)
                    current_step_deg = step_levels_deg[step_index]
                    print(
                        f"\n[gamepad] 关节 Jog 提高：每周期 {current_step_deg:.3f}°。"
                    )
                else:
                    cartesian_speed_index = min(
                        len(cartesian_speed_levels) - 1,
                        cartesian_speed_index + 1,
                    )
                    print(
                        f"\n[gamepad] 末端 Jog 提高："
                        f"{cartesian_speed_levels[cartesian_speed_index]:.2f}x。"
                    )

            close_value = common.normalize_trigger(
                int(pad.left_trigger), args.trigger_deadzone
            )
            open_value = common.normalize_trigger(
                int(pad.right_trigger), args.trigger_deadzone
            )
            gripper_input = open_value - close_value
            if arm_enabled and gripper_input != 0.0 and now >= next_gripper_command:
                candidate_width = gripper_width + gripper_input * gripper_step
                candidate_width = min(max(candidate_width, 0.0), gripper_max)
                if not math.isclose(candidate_width, gripper_width, abs_tol=1e-9):
                    gripper_width = candidate_width
                    gripper.move_gripper_m(gripper_width, args.gripper_force)
                    print(
                        f"\r夹爪目标：{gripper_width * 1000:6.1f} mm       ",
                        end="", flush=True,
                    )
                next_gripper_command = now + 0.05

            joints = read_and_validate_joints(robot, config)
            if ik_solver is not None and (not control_active or control_mode != "cartesian"):
                ik_solver.reset_anchor()
            control_inputs = [0.0] * 6
            if arm_enabled and control_active:
                status = robot.get_arm_status()
                if status is not None and base.value_as_int(status.msg.arm_status) != 0:
                    control_active = False
                    raise RuntimeError("机械臂状态异常，已暂停控制。")

                if control_mode == "joint":
                    control_inputs = joint_inputs(pad, args.deadzone)
                    if any(control_inputs):
                        targets, active = build_joint_targets(
                            targets, joints, control_inputs,
                            math.radians(current_step_deg) * rate_scale,
                            max_lead, bounds,
                        )
                        robot.move_js(targets)
                        active_text = "+".join(
                            f"J{index + 1}" for index in active
                        )
                        print(
                            f"\r关节 Jog={current_step_deg:5.3f}°/周期  "
                            f"{active_text:<17}       ",
                            end="", flush=True,
                        )
                    else:
                        targets = joints[:]
                else:
                    control_inputs = cartesian_inputs(pad, args.deadzone)
                    if any(control_inputs[:3]):
                        control_inputs[3:] = [0.0, 0.0, 0.0]
                    else:
                        active_orientation = [
                            index for index in range(3, 6)
                            if control_inputs[index]
                        ]
                        if len(active_orientation) > 1:
                            keep = max(
                                active_orientation,
                                key=lambda index: abs(control_inputs[index]),
                            )
                            for index in active_orientation:
                                if index != keep:
                                    control_inputs[index] = 0.0
                    if any(control_inputs):
                        scale = cartesian_speed_levels[cartesian_speed_index]
                        ik_started = time.perf_counter()
                        try:
                            desired_joints, pose_target = solve_cartesian_target(
                                robot, joints, pose_target, control_inputs, bounds,
                                [value * scale for value in position_steps],
                                angle_step * scale, max_position_lead,
                                max_angle_lead, max_joint_command, preferred_joints,
                                tcp_offset,
                                ik_solver=ik_solver,
                            )
                            diagnostic_ik_ms = max(diagnostic_ik_ms, (time.perf_counter() - ik_started) * 1000)
                            targets = interpolate_cartesian_joints(
                                targets, desired_joints, joints,
                                cartesian_smoothing, max_lead, bounds,
                            )
                            robot.move_js(targets)
                            if time.monotonic() >= next_status:
                                next_status = time.monotonic() + 0.2
                                print(
                                    f"\r末端 {args.ik_backend} {scale:.2f}x  "
                                    f"平滑={cartesian_smoothing:.2f}  "
                                    f"XYZ={pose_target[0]*1000:.1f}/"
                                    f"{pose_target[1]*1000:.1f}/"
                                    f"{pose_target[2]*1000:.1f} mm  RPY="
                                    f"{math.degrees(pose_target[3]):.1f}/"
                                    f"{math.degrees(pose_target[4]):.1f}/"
                                    f"{math.degrees(pose_target[5]):.1f}°       ",
                                    end="", flush=True,
                                )
                        except IKError as exc:
                            diagnostic_ik_ms = max(diagnostic_ik_ms, (time.perf_counter() - ik_started) * 1000)
                            ik_rejections += 1
                            targets = joints[:]
                            pose_target = cartesian_pose(robot, joints, tcp_offset)
                            print(f"\rIK 拒绝：{exc}       ", end="", flush=True)
                    else:
                        targets = joints[:]
                        pose_target = cartesian_pose(robot, joints, tcp_offset)
            else:
                targets = joints[:]
                pose_target = cartesian_pose(robot, joints, tcp_offset)

            record_started = time.monotonic()
            diagnostic_control_ms = max(diagnostic_control_ms, (record_started - now) * 1000)
            if not recorder_broken:
                try:
                    state_gripper = read_gripper_feedback(gripper, gripper_width)
                    recorder.capture(
                        time.monotonic(), [*joints, state_gripper],
                        [*targets, gripper_width],
                        cartesian_pose(robot, joints, tcp_offset), control_inputs,
                        show_preview=record_started >= next_preview,
                    )
                    if record_started >= next_preview:
                        next_preview = time.monotonic() + 0.1
                except RuntimeError as exc:
                    control_active = False
                    targets = joints[:]
                    if arm_enabled:
                        robot.move_js(targets)
                    if recorder.active:
                        recorder.mark_incomplete()
                    recorder_broken = True
                    print(f"\n[gamepad] Recording stopped and control paused: {exc}")

            previous_buttons = buttons
            diagnostic_record_ms = max(diagnostic_record_ms, (time.monotonic() - record_started) * 1000)
            diagnostic_cycles += 1
            diagnostic_max_ms = max(diagnostic_max_ms, (time.monotonic() - tick_started) * 1000)
            elapsed = time.monotonic() - diagnostic_started
            if elapsed >= 2.0:
                print(f"\n[gamepad timing] backend={args.ik_backend} rate={diagnostic_cycles / elapsed:.1f}Hz "
                      f"work_max={diagnostic_max_ms:.1f}ms ik_max={diagnostic_ik_ms:.1f}ms "
                      f"input_max={diagnostic_input_ms:.1f}ms control_max={diagnostic_control_ms:.1f}ms "
                      f"record_max={diagnostic_record_ms:.1f}ms "
                      f"rejects_total={ik_rejections}")
                diagnostic_started = time.monotonic()
                diagnostic_cycles = 0
                diagnostic_max_ms = diagnostic_ik_ms = 0.0
                diagnostic_input_ms = diagnostic_record_ms = diagnostic_control_ms = 0.0
            sleep_remainder(tick_started, poll_period)

    except KeyboardInterrupt:
        print("\n[gamepad] Ctrl+C：停止程序；活动 episode 标记为 incomplete。")
        return 130
    finally:
        if recorder is not None:
            recorder.close()
        base.disconnect_quietly(robot)


def run() -> int:
    """Console entry: ``python -m robotarm.teleop.gamepad.runtime``."""
    try:
        exit_code = main()
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else 0
    except Exception as exc:
        print(f"\n[gamepad] 运行失败：{type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
