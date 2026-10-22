"""机械臂安全检查：关节反馈校验、求解器边界、启动状态。

迁移自 09/_cartesian_dls_base 中的公共安全函数，供手柄与 XR 采集链路共用。
"""

from __future__ import annotations

import math
import time
from typing import Any

from .piper import ensure_arm_status_normal, value_as_int


def validate_joint_feedback(joints: list[float], config: dict[str, Any]) -> None:
    tolerance = math.radians(3.0)
    if len(joints) != 6 or not all(math.isfinite(value) for value in joints):
        raise RuntimeError("关节反馈格式异常，拒绝控制。")
    for index, value in enumerate(joints, start=1):
        low, high = config["joint_limits"][f"joint{index}"]
        if not low - tolerance <= value <= high + tolerance:
            raise RuntimeError(
                f"J{index}反馈 {math.degrees(value):.2f}° 超出预期范围。"
            )


def safe_joint_bounds(
    config: dict[str, Any], margin_degrees: float = 1.0
) -> list[tuple[float, float]]:
    """求解器使用的关节边界，J2/J3 额外允许编码器零位偏移。"""
    bounds = []
    if not math.isfinite(margin_degrees) or margin_degrees < 0:
        raise ValueError("joint limit margin must be finite and non-negative")
    margin = math.radians(margin_degrees)
    for index in range(6):
        low, high = config["joint_limits"][f"joint{index + 1}"]
        offset_tolerance = math.radians(3.0) if index in (1, 2) else 0.0
        bounds.append(
            (low - offset_tolerance + margin, high + offset_tolerance - margin)
        )
    return bounds


def ensure_startup_status(robot: Any, timeout: float) -> None:
    """要求机械臂状态正常；锁存电子急停时明确拒绝自动复位。"""
    try:
        ensure_arm_status_normal(robot, timeout=timeout)
    except RuntimeError:
        status = robot.get_arm_status()
        code = None if status is None else value_as_int(status.msg.arm_status)
        if code == 1:
            raise RuntimeError(
                "检测到已锁存电子急停；采集程序不会自动复位。请先托住机械臂，"
                "再运行：robotarm data-acquisition enable-control -- "
                "reset --confirm-drop --execute"
            ) from None
        raise


def read_gripper_feedback(gripper: Any, fallback: float) -> float:
    """Return the latest width-mode gripper feedback, or *fallback*."""
    status = gripper.get_gripper_status()
    if status is not None and status.msg.mode == "width":
        value = float(status.msg.value)
        if math.isfinite(value):
            return value
    return fallback


def wait_joint_feedback(robot: Any, config: dict[str, Any], timeout: float) -> list[float]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        latest = robot.get_joint_angles()
        if latest is not None:
            joints = list(latest.msg)
            validate_joint_feedback(joints, config)
            return joints
        time.sleep(0.02)
    raise TimeoutError(f"未在 {timeout:.1f} 秒内取得关节反馈。")
