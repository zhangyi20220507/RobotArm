"""PICO/XR 遥操作：--arms 1 控制单臂，--arms 2 用左右手柄分别控制双臂。

迁移自 XRoboToolkit-Teleop-Sample-Python/scripts/hardware/teleop_piper_xr.py，
行为与参数保持兼容；数据记录复用 robotarm.record（与手柄链路同一格式）。

默认 dry-run；传入 ``--live`` 后，长按 B 使能，之后才允许下发遥操命令。
需要统一 RobotArm 环境中的 placo / zmq / xrobotoolkit_sdk。

按键：握住 RIGHT GRIP 接合离合，松开脱离；RIGHT TRIGGER 控制夹爪；
长按 A 回零（保持使能）；长按 B 使能；RIGHT STICK 急停。
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from pathlib import Path

from ...config import defaults
from ...paths import PIPER_RAW_ROOT
from ...record.recorder import RawEpisodeRecorder


def parse_xyz(value: str) -> tuple[float, float, float]:
    try:
        result = [float(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers") from exc
    if len(result) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers")
    return (result[0], result[1], result[2])


def make_parser() -> argparse.ArgumentParser:
    from .control import add_input_options
    cfg = defaults("xr", "piper-xr")
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_input_options(parser, cfg.get("engagement-mode", "direct"))
    parser.add_argument(
        "--arms", type=int, choices=(1, 2), default=cfg["arms"],
        help="1: right controller and --channel; 2: both controllers and --left/right-channel",
    )
    parser.add_argument("--live", action="store_true", help="enable the arm and send commands")
    parser.add_argument("--channel", default=cfg["channel"], help="SocketCAN channel")
    parser.add_argument(
        "--xr-bridge",
        action="store_true",
        help="read XR state from the persistent local bridge instead of opening the SDK",
    )
    parser.add_argument(
        "--xr-bridge-endpoint",
        default=cfg["xr-bridge-endpoint"],
    )
    parser.add_argument(
        "--model", choices=("piper", "piper_h", "piper_l", "piper_x"),
        default=cfg["model"],
    )
    parser.add_argument(
        "--firmware", choices=("auto", "default", "v183", "v188", "v189"),
        default=cfg["firmware"],
    )
    parser.add_argument("--rate", type=float, default=cfg["rate"], help="control frequency in Hz")
    parser.add_argument("--scale", type=float, default=cfg["scale"], help="XR translation scale")
    parser.add_argument("--speed-percent", type=int, default=cfg["speed-percent"], help="Piper speed percentage")
    parser.add_argument(
        "--joint-control-mode",
        choices=("js", "j"),
        default=cfg["joint-control-mode"],
        help="stream joint targets using MOVE_JS (gamepad-runtime behavior) or MOVE_J",
    )
    parser.add_argument("--max-target-lead", type=float, default=cfg["max-target-lead"], help="maximum Cartesian target lead in meters; excess hand displacement is discarded")
    parser.add_argument("--translation-speed", type=float, default=cfg["translation-speed"], help="Cartesian speed cap in m/s")
    parser.add_argument("--joint-speed", type=float, default=cfg["joint-speed"], help="joint speed cap in rad/s")
    parser.add_argument("--orientation-weight", type=float, default=cfg["orientation-weight"], help="attitude preference weight for translation-only control")
    parser.add_argument(
        "--posture-anchor", choices=("engage", "seed"), default=cfg.get("posture-anchor", "engage"),
        help="engage: hold the posture captured at engagement; seed: legacy follow-the-seed",
    )
    parser.add_argument("--xr-yaw-deg", type=float, default=cfg["xr-yaw-deg"], help="rotate XR mapping about robot vertical, in degrees")
    parser.add_argument(
        "--xr-mirror", choices=("none", "lr", "fb"), default=cfg.get("xr-mirror", "none"),
        help="mirror one horizontal axis when only left-right (lr) or forward-back (fb) is reversed",
    )
    parser.add_argument("--grip-threshold", type=float, default=cfg["grip-threshold"])
    parser.add_argument("--grip-release-threshold", type=float, default=cfg["grip-release-threshold"])
    parser.add_argument(
        "--gripper-max-mm", type=float, default=cfg["gripper-max-mm"],
        help="gripper width at fully pressed trigger in mm",
    )
    parser.add_argument(
        "--gripper-speed-mm-s", type=float, default=cfg["gripper-speed-mm-s"],
        help="software gripper width rate limit in mm/s",
    )
    parser.add_argument("--gripper-force", type=float, default=cfg["gripper-force"])
    parser.add_argument(
        "--gripper-force-hold-threshold",
        type=float,
        default=cfg["gripper-force-hold-threshold"],
        help="hold measured width when closing force reaches this value in N; 0 disables",
    )
    parser.add_argument(
        "--button-hold-seconds", type=float, default=cfg["button-hold-seconds"],
        help="hold duration for A home and B discard",
    )
    parser.add_argument(
        "--orientation", action="store_true", help="also control flange orientation (advanced)"
    )
    parser.add_argument("--xyz-min", type=parse_xyz, default=tuple(cfg["xyz-min"]))
    parser.add_argument("--xyz-max", type=parse_xyz, default=tuple(cfg["xyz-max"]))
    parser.add_argument("--max-relative-translation", type=float, default=cfg["max-relative-translation"])
    parser.add_argument("--max-step-translation", type=float, default=cfg["max-step-translation"])
    parser.add_argument(
        "--max-step-rotation", type=float,
        default=math.radians(cfg["max-step-rotation-deg"]),
    )
    parser.add_argument(
        "--max-step-joint",
        type=float,
        default=cfg["max-step-joint"],
        help="maximum change of each commanded joint per cycle in radians",
    )
    parser.add_argument(
        "--max-joint-following-error",
        type=float,
        default=cfg["max-joint-following-error"],
        help="maximum commanded joint lead over feedback in radians",
    )
    parser.add_argument(
        "--joint-limit-tolerance",
        type=float,
        default=math.radians(cfg["joint-limit-tolerance-deg"]),
        help="startup feedback tolerance beyond nominal limits in radians",
    )
    parser.add_argument("--stale-timeout", type=float, default=cfg["stale-timeout"])
    parser.add_argument(
        "--xr-wait-timeout", type=float, default=cfg["xr-wait-timeout"],
        help="seconds to wait for XR data before enabling",
    )
    parser.add_argument(
        "--record", action="store_true",
        help="enable RealSense episode recording on B (same format as the gamepad runtime)",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=PIPER_RAW_ROOT,
        help="raw session 根目录（默认 $ROBOTARM_DATA_ROOT/raw/piper_raw）",
    )
    parser.add_argument("--task", default=cfg["task"])
    parser.add_argument("--setup-id", default=cfg["setup-id"])
    parser.add_argument("--setup-note", default=cfg["setup-note"])
    parser.add_argument("--global-camera-serial", default=cfg["global-camera-serial"])
    parser.add_argument("--wrist-camera-serial", default=cfg["wrist-camera-serial"])
    parser.add_argument("--record-width", type=int, default=cfg["record-width"])
    parser.add_argument("--record-height", type=int, default=cfg["record-height"])
    parser.add_argument("--record-fps", type=int, default=cfg["record-fps"])
    parser.add_argument("--no-preview", action="store_true")
    return parser


def validate_args(args, np) -> None:
    from .control import validate_input_options
    validate_input_options(args)
    if not math.isfinite(args.max_target_lead) or not 0 < args.max_target_lead <= 0.05:
        raise ValueError("max-target-lead must be in (0, 0.05] meters")
    if not all(math.isfinite(v) for v in (args.translation_speed, args.joint_speed, args.orientation_weight, args.xr_yaw_deg)):
        raise ValueError("XR speed, orientation weight and yaw must be finite")
    if not 0 < args.translation_speed <= 0.3 or not 0 < args.joint_speed <= 1.0:
        raise ValueError("translation-speed must be in (0, 0.3] m/s and joint-speed in (0, 1] rad/s")
    if not 0 <= args.orientation_weight <= 1:
        raise ValueError("orientation-weight must be in [0, 1]")
    if args.xr_mirror not in ("none", "lr", "fb"):
        raise ValueError("--xr-mirror must be one of none, lr, fb")
    if args.posture_anchor not in ("engage", "seed"):
        raise ValueError("--posture-anchor must be engage or seed")
    if not 0 <= args.grip_release_threshold < args.grip_threshold <= 1:
        raise ValueError("grip thresholds must satisfy 0 <= release < engage <= 1")
    if args.max_step_rotation <= 0:
        raise ValueError("--max-step-rotation must be positive")
    if args.rate <= 0 or args.scale <= 0:
        raise ValueError("--rate and --scale must be positive")
    if not 0 <= args.speed_percent <= 100:
        raise ValueError("--speed-percent must be in [0, 100]")
    if np.any(np.asarray(args.xyz_min) >= np.asarray(args.xyz_max)):
        raise ValueError("every --xyz-min component must be smaller than --xyz-max")
    if args.max_relative_translation <= 0 or args.max_step_translation <= 0:
        raise ValueError("translation limits must be positive")
    if args.max_step_joint <= 0:
        raise ValueError("--max-step-joint must be positive")
    if args.max_joint_following_error <= 0:
        raise ValueError("--max-joint-following-error must be positive")
    if args.joint_limit_tolerance < 0:
        raise ValueError("--joint-limit-tolerance must be non-negative")
    if args.joint_limit_tolerance > math.radians(5.0):
        raise ValueError("--joint-limit-tolerance must not exceed 5 degrees")
    if args.stale_timeout <= 0 or args.xr_wait_timeout <= 0:
        raise ValueError("XR timeout values must be positive")
    if not 0 < args.gripper_max_mm <= 100:
        raise ValueError("--gripper-max-mm must be in (0, 100]")
    if not 0 < args.gripper_speed_mm_s <= 100:
        raise ValueError("--gripper-speed-mm-s must be in (0, 100]")
    if not 0 < args.gripper_force <= 5:
        raise ValueError("--gripper-force must be in (0, 5]")
    if not 0 <= args.gripper_force_hold_threshold <= args.gripper_force:
        raise ValueError(
            "--gripper-force-hold-threshold must be in [0, --gripper-force]"
        )
    if not 0.3 <= args.button_hold_seconds <= 5.0:
        raise ValueError("--button-hold-seconds must be in [0.3, 5.0]")
    if args.record_width <= 0 or args.record_height <= 0:
        raise ValueError("record dimensions must be positive")
    if not 1 <= args.record_fps <= args.rate:
        raise ValueError("--record-fps must be between 1 and --rate")
    if args.wrist_camera_serial == args.global_camera_serial:
        raise ValueError("global and wrist camera serial numbers must differ")
    if args.record:
        raise ValueError("PICO recording is disabled: hold B is reserved for enabling the arm")
    if args.model != "piper":
        raise ValueError("local IK currently supports --model piper only")


def wait_for_xr(xrt, timeout: float, stop_requested) -> None:
    """Wait until the SDK provides a valid, timestamped right-controller pose."""
    import numpy as np

    from .geometry import valid_xr_pose

    print("Waiting for XR right-controller data...")
    deadline = time.monotonic() + timeout
    previous_timestamp = None
    while time.monotonic() < deadline:
        if stop_requested():
            raise InterruptedError("Stopped while waiting for XR data")
        pose = np.asarray(xrt.get_right_controller_pose(), dtype=float)
        timestamp = xrt.get_time_stamp_ns()
        if timestamp not in (None, 0, previous_timestamp) and previous_timestamp not in (None, 0) and valid_xr_pose(pose):
            print("XR right-controller data ready.")
            return
        previous_timestamp = timestamp
        time.sleep(0.05)
    raise RuntimeError(
        f"No valid XR right-controller data within {timeout:.0f} seconds; "
        "toggle Send off/on or restart the headset app after starting this script"
    )


def main(argv: list[str] | None = None) -> int:
    # Dispatch before loading XR/CAN dependencies; both runtimes retain their
    # own options and defaults. The existing bimanual entrypoint remains valid.
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument(
        "--arms", type=int, choices=(1, 2),
        default=defaults("xr", "piper-xr")["arms"],
    )
    mode, remaining = selector.parse_known_args(sys.argv[1:] if argv is None else argv)
    if mode.arms == 2:
        from .bimanual_runtime import main as run_bimanual
        return run_bimanual(remaining)
    return _main_single(remaining)


def _main_single(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    # XR 运行时依赖（placo/zmq/scipy/xrobotoolkit_sdk）仅在执行期导入。
    import numpy as np

    from ...core.robot.piper import _load_sdk
    from .control import PoseJumpGuard, Clutch, joint_limit_diagnostics, limit_joint_step
    from .arm_ops import (
        arm_status_text,
        hold_arm_on_exit,
        enter_joint_stream_mode,
        gripper_feedback_width,
        home_arm_and_gripper,
        initialize_gripper,
        update_force_limited_gripper,
        wait_for_flange_pose,
        wait_for_joint_angles,
    )
    from .geometry import TranslationTracker, Limits, build_target, clamp_vector_step, clamp_rotation_step, valid_xr_pose, wrap_angles
    from .piper_ik import (
        PIPER_JOINT_MAX,
        PIPER_JOINT_MIN,
        PiperIK,
        joint_limit_violations,
    )

    create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW = _load_sdk()

    def firmware_value(name: str) -> str:
        return {
            "default": PiperFW.DEFAULT,
            "v183": PiperFW.V183,
            "v188": PiperFW.V188,
            "v189": PiperFW.V189,
        }[name]

    def detect_firmware(model: str, channel: str, timeout: float = 5.0) -> tuple[str, str]:
        """Probe Piper firmware using the default parser, then select its driver."""
        probe_config = create_agx_arm_config(
            robot=arm_model_value(model),
            firmeware_version=PiperFW.DEFAULT,
            interface="socketcan",
            channel=channel,
        )
        probe = AgxArmFactory.create_arm(probe_config)
        probe.connect()
        try:
            deadline = time.monotonic() + timeout
            firmware = None
            while time.monotonic() < deadline:
                firmware = probe.get_firmware()
                if firmware is not None:
                    break
                time.sleep(0.1)
            if firmware is None:
                raise RuntimeError(f"Could not detect Piper firmware on {channel}")
            software_version = firmware["software_version"]
        finally:
            probe.disconnect()

        if software_version >= "S-V1.8-9":
            driver = PiperFW.V189
        elif software_version >= "S-V1.8-8":
            driver = PiperFW.V188
        elif software_version >= "S-V1.8-3":
            driver = PiperFW.V183
        else:
            driver = PiperFW.DEFAULT
        return software_version, driver

    def arm_model_value(name: str) -> str:
        return {
            "piper": ArmModel.PIPER,
            "piper_h": ArmModel.PIPER_H,
            "piper_l": ArmModel.PIPER_L,
            "piper_x": ArmModel.PIPER_X,
        }[name]

    validate_args(args, np)

    if args.xr_bridge:
        from .bridge import XrBridgeClient

        xrt = XrBridgeClient(args.xr_bridge_endpoint)
        print(f"XR source: persistent bridge at {args.xr_bridge_endpoint}")
    else:
        import xrobotoolkit_sdk as xrt_sdk

        xrt = xrt_sdk
        print("XR source: direct SDK connection")

    if args.live:
        print("LIVE MODE will enable and command the physical Piper arm.")
        if args.joint_control_mode == "js":
            print("Joint streaming mode: MOVE_JS (bounded, but no controller trajectory smoothing).")
        print("Clear the workspace, remove payloads, and keep the physical E-stop within reach.")

    limits = Limits(
        xyz_min=np.asarray(args.xyz_min, dtype=float),
        xyz_max=np.asarray(args.xyz_max, dtype=float),
        max_relative_translation=args.max_relative_translation,
        max_step_translation=args.max_step_translation,
        max_step_rotation=args.max_step_rotation,
    )

    if args.firmware == "auto":
        software_version, selected_firmware = detect_firmware(args.model, args.channel)
        print(f"Detected Piper firmware {software_version}; using driver {selected_firmware}.")
    else:
        selected_firmware = firmware_value(args.firmware)
        print(f"Using requested Piper firmware driver {selected_firmware}.")

    config = create_agx_arm_config(
        robot=arm_model_value(args.model),
        firmeware_version=selected_firmware,
        interface="socketcan",
        channel=args.channel,
    )
    robot = AgxArmFactory.create_arm(config)
    recorder = None
    gripper = None
    gripper_width_command = 0.0
    gripper_force_hold_width = None
    stop_requested = False
    emergency_stopped = False
    arm_enabled = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    xrt.init()
    try:
        robot.connect()
        flange_feedback = wait_for_flange_pose(robot)
        joint_feedback = wait_for_joint_angles(robot)
        flange_fk = np.asarray(robot.fk(joint_feedback.tolist()), dtype=float)
        ik = PiperIK(dt=1.0 / args.rate, orientation_enabled=args.orientation, orientation_weight=args.orientation_weight)
        flange_ik = ik.flange_pose(joint_feedback)
        print("Piper joints:", np.array2string(joint_feedback, precision=4))
        print("Piper feedback:", np.array2string(flange_feedback, precision=4))
        print("Piper SDK FK:", np.array2string(flange_fk, precision=4))
        print("Piper URDF FK:", np.array2string(flange_ik, precision=4))
        violations = joint_limit_violations(joint_feedback, args.joint_limit_tolerance)
        if violations:
            raise RuntimeError(
                "Piper is outside the official joint limits: "
                + "; ".join(violations)
                + ". Move it to a non-singular ready posture inside all limits before teleoperation."
            )
        print("Mode:", "LIVE" if args.live else "DRY-RUN")
        print(
            "XR motion:",
            f"scale={args.scale:.3f},",
            f"relative_limit={limits.max_relative_translation:.3f} m,",
            f"cartesian_rate_limit={min(limits.max_step_translation * args.rate, args.translation_speed):.3f} m/s,",
            f"joint_rate_limit={min(args.max_step_joint * args.rate, args.joint_speed):.3f} rad/s",
        )
        print("Hold RIGHT GRIP to engage; release to clutch. Press RIGHT STICK for E-stop.")

        wait_for_xr(xrt, args.xr_wait_timeout, lambda: stop_requested)

        if args.record:
            camera_serials = {"global": args.global_camera_serial}
            if args.wrist_camera_serial:
                camera_serials["wrist"] = args.wrist_camera_serial
            recorder = RawEpisodeRecorder(
                args.dataset_root,
                args.task,
                args.setup_id,
                args.setup_note,
                args.record_fps,
                args.record_width,
                args.record_height,
                camera_serials,
                {
                    "model": "Piper",
                    "firmware": selected_firmware,
                    "interface": "socketcan",
                    "channel": args.channel,
                    "control_mode": f"pico_placo_move_{args.joint_control_mode}",
                    "joint_frame": "piper_base",
                    "tcp_pose_units": ["m", "m", "m", "rad", "rad", "rad"],
                },
                preview=not args.no_preview,
            )

        if args.live:
            if not np.all(
                (flange_ik[:3] >= limits.xyz_min)
                & (flange_ik[:3] <= limits.xyz_max)
            ):
                print(
                    "WARNING: current flange is outside the configured workspace; "
                    "recovery-only bounds will be used until the next clutch."
                )
            print("LIVE ready: hold B for 1 second to enable; direct mode needs no Grip.")
            print("Buttons: hold B=enable, hold A=home (stay enabled), RIGHT STICK=E-stop")
        else:
            print("DRY-RUN: B does not enable hardware; restart with --live for arm control.")

        arm_ready = False

        period = 1.0 / args.rate
        xr_reference = None
        flange_reference = None
        last_cartesian_target = flange_ik.copy()
        last_joint_command = joint_feedback.copy()
        last_xr_timestamp = None
        last_xr_change = time.monotonic()
        last_print = 0.0
        last_wait_print = 0.0
        jump_guard = PoseJumpGuard(args.max_hand_jump_m, args.max_hand_jump_deg)
        print(f"Engagement mode: {args.engagement_mode}; hold B to enable/resume; no Grip required in direct mode.")
        clutch = Clutch(args.grip_threshold, args.grip_release_threshold, mode=args.engagement_mode, blocked=args.live)
        previous_a = False
        previous_b = False
        a_pressed_at = None
        a_long_fired = False
        b_pressed_at = None
        b_long_fired = False
        next_gripper_command = 0.0
        previous_tick = time.monotonic() - period
        translation_tracker = None
        last_limit_notice = 0.0

        def hold_current_position():
            if not args.live:
                return
            held = wait_for_joint_angles(robot)
            if args.joint_control_mode == "js":
                robot.move_js(held.tolist())
            else:
                robot.move_j(held.tolist())

        def home_stop_requested():
            nonlocal last_xr_timestamp, last_xr_change
            now = time.monotonic()
            stamp = xrt.get_time_stamp_ns()
            if stamp not in (None, 0, last_xr_timestamp):
                last_xr_timestamp, last_xr_change = stamp, now
            if stop_requested:
                raise InterruptedError("Normal exit requested during homing")
            return (xrt.get_right_axis_click()
                    or stamp in (None, 0)
                    or now - last_xr_change > args.stale_timeout)

        while not stop_requested:
            loop_start = time.monotonic()
            tick_elapsed = max(0.0, loop_start - previous_tick)
            dt = min(period, tick_elapsed)
            previous_tick = loop_start
            translation_step = min(limits.max_step_translation, args.translation_speed * dt)
            joint_step = min(args.max_step_joint, args.joint_speed * dt)
            teleop_delta = np.zeros(6, dtype=float)
            # Bridge fields must come from the same message, not separate refreshes.
            state = xrt.snapshot() if args.xr_bridge else {
                "timestamp_ns": xrt.get_time_stamp_ns(),
                "right_controller_pose": xrt.get_right_controller_pose(),
                "right_grip": xrt.get_right_grip(),
                "right_trigger": xrt.get_right_trigger(),
                "right_axis_click": xrt.get_right_axis_click(),
                "A_button": xrt.get_A_button(),
                "B_button": xrt.get_B_button(),
            }
            timestamp = state["timestamp_ns"]
            if timestamp != last_xr_timestamp:
                last_xr_timestamp = timestamp
                last_xr_change = loop_start
            fresh = (
                timestamp not in (None, 0)
                and loop_start - last_xr_change <= args.stale_timeout
            )

            if state["right_axis_click"]:
                print("Emergency stop requested by RIGHT STICK.")
                if args.live:
                    robot.electronic_emergency_stop()
                emergency_stopped = True
                break

            grip = float(state["right_grip"])
            pose = np.asarray(state["right_controller_pose"], dtype=float)
            jump = jump_guard.update(pose) if fresh else None
            if jump:
                print(f"INPUT PAUSED: {jump}; hold B to resume.")
                if xr_reference is not None:
                    hold_current_position()
                xr_reference = None
                flange_reference = None
                clutch.pause()
                time.sleep(period)
                continue
            active = clutch.update(grip, fresh and valid_xr_pose(pose))
            engaged = clutch.engaged
            trigger = float(np.clip(state["right_trigger"], 0.0, 1.0))
            a_pressed = fresh and bool(state["A_button"])
            b_pressed = fresh and bool(state["B_button"])

            if a_pressed and not previous_a:
                a_pressed_at = loop_start
                a_long_fired = False
            if (
                a_pressed
                and a_pressed_at is not None
                and not a_long_fired
                and loop_start - a_pressed_at >= args.button_hold_seconds
            ):
                a_long_fired = True
                if grip >= args.grip_release_threshold:
                    print("Release RIGHT GRIP, then hold A again to home.")
                elif args.live and arm_ready and gripper is not None:
                    if recorder is not None and recorder.active:
                        recorder.mark_incomplete()
                    print("A held: homing all six joints and closing the gripper...")
                    xr_reference = None
                    flange_reference = None
                    clutch.blocked = True
                    joint_feedback, gripper_width_command = home_arm_and_gripper(
                        robot,
                        gripper,
                        args.joint_control_mode,
                        gripper_width_command,
                        args.gripper_force,
                        args.gripper_speed_mm_s / 1000.0,
                        home_stop_requested,
                    )
                    gripper_force_hold_width = None
                    last_joint_command = joint_feedback.copy()
                    last_cartesian_target = ik.flange_pose(joint_feedback)
                    previous_a = a_pressed
                    previous_b = b_pressed
                    continue
                elif args.live:
                    print("Arm is not enabled by this session; hold B first.")
            if not a_pressed:
                a_pressed_at = None
                a_long_fired = False

            if b_pressed and not previous_b:
                b_pressed_at = loop_start
                b_long_fired = False
            if (
                b_pressed
                and b_pressed_at is not None
                and not b_long_fired
                and loop_start - b_pressed_at >= args.button_hold_seconds
            ):
                b_long_fired = True
                if not args.live:
                    print("DRY-RUN: enable skipped. Restart with --live, then hold B.")
                elif (args.engagement_mode == "grip" and engaged) or a_pressed:
                    print("Release RIGHT GRIP and A, then hold B again to enable.")
                else:
                    # Mark the attempt before touching hardware so partial
                    # enable failures also execute emergency-stop cleanup.
                    arm_enabled = True
                    arm_ready = False
                    xr_reference = None
                    flange_reference = None
                    clutch.blocked = True
                    print("B held: enabling all six joints at the current position...")
                    current_joints = wait_for_joint_angles(robot)
                    violations = joint_limit_violations(current_joints, args.joint_limit_tolerance)
                    if violations:
                        raise RuntimeError("Cannot enable: " + "; ".join(violations))
                    robot.set_speed_percent(args.speed_percent)
                    joint_feedback = enter_joint_stream_mode(
                        robot, current_joints, args.joint_control_mode
                    )
                    enable_states = robot.get_joints_enable_status_list()
                    if len(enable_states) != 6 or not all(enable_states):
                        raise RuntimeError("Enable failed, joint feedback: " + str(enable_states))
                    if gripper is None:
                        gripper, gripper_width_command = initialize_gripper(
                            robot, args.gripper_max_mm / 1000.0, args.gripper_force
                        )
                        gripper_force_hold_width = None
                    last_joint_command = joint_feedback.copy()
                    last_cartesian_target = ik.flange_pose(joint_feedback)
                    arm_ready = True
                    clutch.resume()
                    print("Piper enabled. Joint motor enables:", enable_states)
                    print("Control ready; direct mode follows without Grip. Status:", arm_status_text(robot))
                    previous_a, previous_b = a_pressed, b_pressed
                    continue
            if not b_pressed:
                b_pressed_at = None
                b_long_fired = False

            if (
                args.live
                and arm_ready
                and fresh
                and active
                and gripper is not None
                and loop_start >= next_gripper_command
            ):
                max_width_step = args.gripper_speed_mm_s / 1000.0 * 0.05
                (
                    gripper_width_command,
                    gripper_force_hold_width,
                    gripper_event,
                    feedback_force,
                ) = update_force_limited_gripper(
                    gripper,
                    gripper_width_command,
                    gripper_force_hold_width,
                    trigger,
                    args.gripper_max_mm / 1000.0,
                    max_width_step,
                    args.gripper_force,
                    args.gripper_force_hold_threshold,
                )
                if gripper_event == "latched":
                    print(
                        "Gripper force hold:",
                        f"{feedback_force:.3f} N at {gripper_width_command * 1000.0:.1f} mm",
                    )
                elif gripper_event == "released":
                    print("Gripper force hold released by Trigger")
                next_gripper_command = loop_start + 0.05

            if active and (not args.live or arm_ready):
                if xr_reference is None:
                    current_joints = wait_for_joint_angles(robot)
                    feedback = ik.flange_pose(current_joints)
                    outside_workspace = not np.all(
                        (feedback[:3] >= limits.xyz_min)
                        & (feedback[:3] <= limits.xyz_max)
                    )
                    xr_reference = pose.copy()
                    flange_reference = feedback.copy()
                    ik.set_posture_anchor(
                        current_joints if args.posture_anchor == "engage" else None
                    )
                    translation_tracker = TranslationTracker(
                        pose[:3], feedback[:3], args.scale, limits,
                        args.max_target_lead, args.xr_yaw_deg, args.xr_mirror,
                    )
                    last_cartesian_target = feedback.copy()
                    last_joint_command = current_joints.copy()
                    if outside_workspace:
                        print(
                            "ENGAGED (outside workspace: recovery-only bounds active)",
                            "xyz=",
                            np.array2string(feedback[:3], precision=4),
                        )
                    else:
                        print("ENGAGED")

                if xr_reference is not None and flange_reference is not None:
                    requested = build_target(
                        xr_reference,
                        pose,
                        flange_reference,
                        args.scale,
                        args.orientation,
                        limits,
                        yaw_degrees=args.xr_yaw_deg,
                        mirror=args.xr_mirror,
                    )
                    current_joints = wait_for_joint_angles(robot)
                    actual_pose = ik.flange_pose(current_joints)
                    requested[:3], limit_reasons = translation_tracker.update(pose[:3], actual_pose[:3])
                    if limit_reasons and loop_start - last_limit_notice >= 1.0:
                        print("; ".join(limit_reasons))
                        last_limit_notice = loop_start
                    cartesian_target = requested.copy()
                    requested_step = float(
                        np.linalg.norm(requested[:3] - last_cartesian_target[:3])
                    )
                    cartesian_target[:3] = clamp_vector_step(
                        last_cartesian_target[:3], requested[:3], translation_step
                    )
                    cartesian_target[3:] = clamp_rotation_step(
                        last_cartesian_target[3:], requested[3:], limits.max_step_rotation
                    )

                    cartesian_step = float(
                        np.linalg.norm(
                            cartesian_target[:3] - last_cartesian_target[:3]
                        )
                    )
                    cartesian_ratio = min(
                        1.0,
                        1.0 if requested_step <= 1.0e-12
                        else cartesian_step / requested_step,
                    )
                    seed_joints = last_joint_command.copy()
                    try:
                        ik_result = ik.solve_adaptive(seed_joints, cartesian_target)
                        ik_solution = ik_result.joints
                    except RuntimeError as exc:
                        print(f"IK rejected target: {exc}")
                        hold_current_position()
                        xr_reference = None
                        flange_reference = None
                        clutch.blocked = True
                        previous_a = a_pressed
                        previous_b = b_pressed
                        continue
                    ik_diagnostics = ik_result.diagnostics
                    ik_ms = ik_diagnostics.solve_ms
                    try:
                        joint_command = limit_joint_step(
                            seed_joints, ik_solution, current_joints,
                            joint_step, args.max_joint_following_error,
                            PIPER_JOINT_MIN, PIPER_JOINT_MAX,
                        )
                    except RuntimeError as exc:
                        print(f"Joint target rejected: {exc}")
                        hold_current_position()
                        xr_reference = None
                        flange_reference = None
                        clutch.blocked = True
                        previous_a, previous_b = a_pressed, b_pressed
                        time.sleep(period)
                        continue

                    limit_ratio, joint_limit_sources = joint_limit_diagnostics(
                        seed_joints, ik_solution, current_joints, joint_command,
                        joint_step, args.max_joint_following_error,
                        PIPER_JOINT_MIN, PIPER_JOINT_MAX,
                    )
                    limit_sources = [
                        reason.split(":", 1)[0] for reason in limit_reasons
                    ]
                    if cartesian_ratio < 1.0 - 1.0e-6:
                        limit_sources.append("CART_STEP")
                    if ik_diagnostics.target_scale < 1.0:
                        limit_sources.append("IK_SHRINK")
                    limit_sources.extend(joint_limit_sources)
                    limit_sources = list(dict.fromkeys(limit_sources))

                    outside_distance = np.maximum(
                        np.maximum(PIPER_JOINT_MIN - current_joints, current_joints - PIPER_JOINT_MAX),
                        0.0,
                    )
                    allowed_lead = args.max_joint_following_error + outside_distance
                    if np.any(np.abs(joint_command - current_joints) > allowed_lead + 1.0e-6):
                        print("IK rejected target: joint command exceeds the following-error limit")
                        hold_current_position()
                        xr_reference = None
                        flange_reference = None
                        clutch.blocked = True
                        previous_a = a_pressed
                        previous_b = b_pressed
                        continue

                    if args.live:
                        if args.joint_control_mode == "js":
                            robot.move_js(joint_command.tolist())
                        else:
                            robot.move_j(joint_command.tolist())
                    # Anchor the next target to the achievable limited command.
                    last_cartesian_target = ik.flange_pose(joint_command)
                    last_joint_command = joint_command
                    teleop_delta = cartesian_target - ik.flange_pose(current_joints)
                    teleop_delta[3:] = wrap_angles(teleop_delta[3:])
                    if (limit_ratio < 0.25 and np.linalg.norm(requested[:3] - actual_pose[:3]) > 0.005
                            and loop_start - last_limit_notice >= 1.0):
                        print("JOINT_LIMIT: 当前姿态或关节限速限制了跟随，请减慢动作或松手调整姿态")
                        last_limit_notice = loop_start
                    if loop_start - last_print >= 0.25:
                        actual_ee = ik.flange_pose(current_joints)
                        status = [
                            f"loop_hz={1.0/max(tick_elapsed,1e-6):.1f}",
                            f"ik_ms={ik_ms:.2f}",
                            f"ik_residual_mm={ik_diagnostics.position_error_m * 1000.0:.2f}",
                            f"ik_residual_deg={math.degrees(ik_diagnostics.orientation_error_rad):.2f}",
                            f"cond_xyz={ik_diagnostics.position_condition:.1f}",
                            f"cond_pose={ik_diagnostics.pose_condition:.1f}",
                            f"ik_shrink={ik_diagnostics.target_scale:.3f}",
                            f"cart_ratio={cartesian_ratio:.2f}",
                            f"limit_ratio={limit_ratio:.2f}",
                            f"limits={','.join(limit_sources) or 'NONE'}",
                            f"tracking_error_mm={np.linalg.norm(requested[:3] - actual_ee[:3])*1000:.1f}",
                            f"joint_lead_rad={np.max(np.abs(joint_command-current_joints)):.3f}",
                            "hand_delta:",
                            np.array2string(pose[:3] - xr_reference[:3], precision=4),
                            "requested_delta:",
                            np.array2string(requested[:3] - flange_reference[:3], precision=4),
                            "ee_target:",
                            np.array2string(cartesian_target, precision=4),
                            "ee_actual:",
                            np.array2string(actual_ee, precision=4),
                            "q_target:",
                            np.array2string(last_joint_command, precision=4),
                            "q_actual:",
                            np.array2string(current_joints, precision=4),
                        ]
                        if args.live:
                            status.extend(["state:", arm_status_text(robot)])
                        status.extend(
                            [
                                "grip:",
                                f"{grip:.2f}",
                                "trigger:",
                                f"{trigger:.2f}",
                                "gripper_mm:",
                                f"{gripper_width_command * 1000.0:.1f}",
                            ]
                        )
                        print(*status)
                        last_print = loop_start
            else:
                if xr_reference is not None:
                    reason = "XR DATA STALE" if not fresh else "DISENGAGED"
                    print(reason)
                    hold_current_position()
                xr_reference = None
                flange_reference = None
                ik.set_posture_anchor(None)
                if loop_start - last_wait_print >= 1.0:
                    print(
                        "waiting:",
                        f"grip={grip:.2f}",
                        f"trigger={trigger:.2f}",
                        f"gripper_mm={gripper_width_command * 1000.0:.1f}",
                        f"enabled={arm_ready}",
                        f"fresh={fresh}",
                        f"pose_valid={valid_xr_pose(pose)}",
                    )
                    last_wait_print = loop_start

            if recorder is not None:
                record_joints = wait_for_joint_angles(robot)
                record_gripper = gripper_feedback_width(
                    gripper, gripper_width_command
                )
                recorder.capture(
                    loop_start,
                    [*record_joints.tolist(), record_gripper],
                    [*last_joint_command.tolist(), gripper_width_command],
                    list(robot.fk(record_joints.tolist())),
                    teleop_delta.tolist(),
                )

            previous_a = a_pressed
            previous_b = b_pressed

            elapsed = time.monotonic() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)

    except Exception as exc:
        normal_cancel = stop_requested and isinstance(exc, InterruptedError)
        if not normal_cancel and args.live and arm_enabled and not emergency_stopped:
            try:
                robot.electronic_emergency_stop()
                emergency_stopped = True
                print("Emergency stop sent because the control loop failed.", file=sys.stderr)
            except Exception:
                pass
        if not normal_cancel:
            raise
    finally:
        # Normal exits hold position; E-stop is reserved for faults/user request.
        if args.live and arm_enabled and not emergency_stopped:
            try:
                hold_arm_on_exit(robot, args.joint_control_mode)
                print("Piper position held; exiting without E-stop or disabling motors.")
            except Exception as exc:
                print(f"Failed to hold Piper on exit: {exc}", file=sys.stderr)
        try:
            if recorder is not None:
                recorder.close()
        finally:
            try:
                robot.disconnect()
            finally:
                xrt.close()

    print("Stopped.")
    return 0


def run() -> int:
    """Console entry: ``python -m robotarm.teleop.xr.runtime``."""
    try:
        return main()
    except KeyboardInterrupt:
        print("\n[xr] Ctrl+C：停止程序。")
        return 130
    except Exception as exc:
        print(f"\n[xr] 运行失败：{type(exc).__name__}: {exc}", flush=True)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
