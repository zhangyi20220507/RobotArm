"""Piper 机械臂硬件访问层：CAN 连接、参数校验、反馈等待和安全检查。

本模块是 pyAgxArm SDK 的唯一接触点；上层（teleop、record）不应直接
import pyAgxArm。所有 SDK 导入都是惰性的，保证未安装硬件依赖时
``--help`` 与静态检查可用。

迁移自 piper_ctrl/control_scripts/src/piper_common.py。
"""

from __future__ import annotations

import argparse
import math
import platform
import sys
import time
from typing import Any, Callable, Iterable

from ...paths import PYAGXARM_ROOT


def add_connection_arguments(
    parser: argparse.ArgumentParser, config: dict | None = None
) -> None:
    """Register CAN connection arguments.

    Default values come from the caller's YAML workflow config (see
    robotarm.config); ``interface`` falls back to platform detection when
    the config leaves it empty.
    """
    config = config or {}
    platform_defaults = connection_defaults()
    interface = config.get("interface") or platform_defaults["interface"]
    channel = config.get("channel") or platform_defaults["channel"]
    parser.add_argument(
        "--interface",
        default=interface,
        help=f"python-can interface (default: {interface})",
    )
    parser.add_argument(
        "--channel",
        default=channel,
        help=f"CAN channel (default: {channel})",
    )
    parser.add_argument(
        "--firmware",
        choices=("default", "v183", "v188", "v189"),
        default=config.get("firmware", "default"),
        help="Piper main-controller firmware API variant",
    )
    parser.add_argument(
        "--feedback-timeout",
        type=positive_float,
        default=config.get("feedback-timeout", 3.0),
        help="seconds to wait for the first CAN feedback frame",
    )


def add_execute_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually send control frames; without this flag only a dry-run is performed",
    )


def connection_defaults() -> dict[str, str]:
    system = platform.system()
    if system == "Windows":
        return {"interface": "agx_cando", "channel": "0"}
    if system == "Darwin":
        return {"interface": "slcan", "channel": "/dev/ttyACM0"}
    return {"interface": "socketcan", "channel": "can_follower1"}


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def percentage(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 100:
        raise argparse.ArgumentTypeError("speed must be in the range 1..100")
    return parsed


def _load_sdk() -> tuple[Any, Any, Any, Any]:
    sdk_path = str(PYAGXARM_ROOT)
    if not PYAGXARM_ROOT.is_dir():
        raise RuntimeError(f"Local pyAgxArm source not found: {PYAGXARM_ROOT}")
    # Conda activation does not necessarily remove ROS/user PYTHONPATH entries.
    # Always put this workspace's SDK first, even if the same path already
    # appears later in sys.path, so an unrelated installed package cannot win.
    while sdk_path in sys.path:
        sys.path.remove(sdk_path)
    sys.path.insert(0, sdk_path)

    try:
        from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import pyAgxArm or one of its dependencies.\n"
            f"Import error: {exc}\n"
            f"Current Python: {sys.executable}\n"
            "Install the missing dependency for the active Python, then retry.\n"
            "Ubuntu example: `python3 -m pip install python-can`, then run:\n"
            "`python3 -m robotarm data-acquisition read-status`"
        ) from exc
    return create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    create_config, _, arm_model, piper_fw = _load_sdk()
    firmware_map = {
        "default": piper_fw.DEFAULT,
        "v183": piper_fw.V183,
        "v188": piper_fw.V188,
        "v189": piper_fw.V189,
    }
    return create_config(
        robot=arm_model.PIPER,
        firmeware_version=firmware_map[args.firmware],
        interface=args.interface,
        channel=args.channel,
    )


def connect_robot(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    _, factory, _, _ = _load_sdk()
    config = build_config(args)
    robot = factory.create_arm(config)
    robot.connect()
    # Keep the SDK's model-specific software joint limits enabled.
    robot.set_joint_limits_enabled(True)
    return robot, config


def wait_for_feedback(robot: Any, timeout: float) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        feedback = robot.get_joint_angles()
        if feedback is not None:
            return feedback
        time.sleep(0.02)
    raise TimeoutError(
        f"No Piper joint feedback received within {timeout:.1f}s. "
        "Check power, CAN wiring, interface/channel and the 1 Mbps bitrate."
    )


def retry_boolean(
    operation: Callable[[], bool], timeout: float, description: str
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if operation():
            return
        time.sleep(0.02)
    raise TimeoutError(f"{description} timed out after {timeout:.1f}s")


def wait_gripper_width(gripper: Any, timeout: float) -> float:
    """Wait for width-mode gripper feedback and return its width in metres."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = gripper.get_gripper_status()
        if status is not None and status.msg.mode == "width":
            return float(status.msg.value)
        time.sleep(0.02)
    raise TimeoutError(f"No gripper width feedback within {timeout:.1f}s")


def gripper_is_enabled(gripper: Any) -> bool | None:
    """Return the gripper drive state, or None before feedback is available."""
    status = gripper.get_gripper_status()
    if status is None:
        return None
    return bool(status.msg.foc_status.driver_enable_status)


def set_gripper_enabled(
    gripper: Any,
    enabled: bool,
    hold_width: float,
    force: float = 1.0,
    timeout: float = 5.0,
) -> None:
    """Set gripper drive state without deliberately changing its width."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = gripper_is_enabled(gripper)
        if state is enabled:
            return
        if enabled:
            # move_gripper_m transmits status_code=1 (enable/width). Keeping
            # the measured width avoids a jump when synchronizing enable.
            gripper.move_gripper_m(hold_width, force)
        else:
            gripper.disable_gripper()
        time.sleep(0.05)
    action = "enable" if enabled else "disable"
    raise TimeoutError(f"Gripper {action} timed out after {timeout:.1f}s")


def enable_arm_and_gripper(
    robot: Any,
    gripper: Any,
    hold_width: float,
    force: float = 1.0,
    timeout: float = 5.0,
) -> None:
    """Enable all six joints and the gripper as one logical operation."""
    retry_boolean(robot.enable, timeout, "Enable all arm joints")
    set_gripper_enabled(gripper, True, hold_width, force, timeout)


def disable_arm_and_gripper(
    robot: Any,
    gripper: Any,
    hold_width: float,
    force: float = 1.0,
    timeout: float = 5.0,
) -> None:
    """Disable the gripper and all six joints as one logical operation."""
    set_gripper_enabled(gripper, False, hold_width, force, timeout)
    retry_boolean(robot.disable, timeout, "Disable all arm joints")


def wait_motion_done(robot: Any, timeout: float) -> bool:
    # Avoid reading a stale "done" frame immediately after sending the target.
    time.sleep(0.5)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = robot.get_arm_status()
        if status is not None and value_as_int(status.msg.motion_status) == 0:
            return True
        time.sleep(0.05)
    return False


def value_as_int(value: Any) -> int:
    raw = getattr(value, "value", value)
    return int(raw)


def ensure_arm_status_normal(robot: Any, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    status = None
    while time.monotonic() < deadline:
        status = robot.get_arm_status()
        if status is not None:
            break
        time.sleep(0.02)
    if status is None:
        raise RuntimeError(
            f"No arm-status feedback was received within {timeout:.1f}s"
        )
    code = value_as_int(status.msg.arm_status)
    if code != 0:
        raise RuntimeError(f"Arm status is not normal (code={code}); motion refused")


def get_all_joints_enabled_state(robot: Any, timeout: float) -> bool:
    """Return the uniform six-joint enable state; reject missing/mixed states."""
    enabled_by_joint: dict[int, bool] = {}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and len(enabled_by_joint) < 6:
        for index in range(1, 7):
            if index in enabled_by_joint:
                continue
            state = robot.get_driver_states(index)
            if state is not None:
                enabled_by_joint[index] = bool(
                    state.msg.foc_status.driver_enable_status
                )
        if len(enabled_by_joint) < 6:
            time.sleep(0.02)

    missing = [
        f"J{index}" for index in range(1, 7) if index not in enabled_by_joint
    ]
    if missing:
        raise RuntimeError(
            f"在 {timeout:.1f} 秒内未收到 {', '.join(missing)} 的驱动器状态。"
        )

    enabled = [index for index, value in enabled_by_joint.items() if value]
    if enabled and len(enabled) != 6:
        enabled_names = ", ".join(f"J{index}" for index in enabled)
        raise RuntimeError(
            f"检测到部分使能状态（已使能：{enabled_names}），拒绝切换控制模式。"
        )
    return len(enabled) == 6


def validate_joint_targets(
    target_rad: Iterable[float], config: dict[str, Any], margin_deg: float = 1.0
) -> None:
    margin = math.radians(margin_deg)
    limits = config["joint_limits"]
    for index, target in enumerate(target_rad, start=1):
        low, high = limits[f"joint{index}"]
        safe_low, safe_high = low + margin, high - margin
        if not safe_low <= target <= safe_high:
            raise ValueError(
                f"J{index} target {math.degrees(target):.2f}° is outside the "
                f"safety-margined range "
                f"[{math.degrees(safe_low):.2f}°, {math.degrees(safe_high):.2f}°]"
            )


def print_joint_table(current_rad: Iterable[float], target_rad: Iterable[float]) -> None:
    print("Joint   current       target        delta")
    for index, (current, target) in enumerate(zip(current_rad, target_rad), start=1):
        print(
            f"J{index:<2}   {math.degrees(current):>8.2f}°   "
            f"{math.degrees(target):>8.2f}°   "
            f"{math.degrees(target - current):>+8.2f}°"
        )


def disconnect_quietly(robot: Any | None) -> None:
    if robot is None:
        return
    try:
        robot.disconnect()
    except Exception:
        pass
