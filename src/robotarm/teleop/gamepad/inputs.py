"""手柄输入解析：摇杆/十字键 → 归一化的 J1..J6 或 XYZ/RPY 输入。"""

from __future__ import annotations

from typing import Any

from .device import (
    DPAD_DOWN,
    DPAD_LEFT,
    DPAD_RIGHT,
    DPAD_UP,
    normalize_axis,
)


def joint_inputs(pad: Any, deadzone: float) -> list[float]:
    """Return normalized J1..J6 Jog inputs; all axes may move together."""
    buttons = int(pad.buttons)
    return [
        -normalize_axis(int(pad.left_x), deadzone),
        -normalize_axis(int(pad.left_y), deadzone),
        -normalize_axis(int(pad.right_y), deadzone),
        float(bool(buttons & DPAD_RIGHT))
        - float(bool(buttons & DPAD_LEFT)),
        normalize_axis(int(pad.right_x), deadzone),
        float(bool(buttons & DPAD_UP))
        - float(bool(buttons & DPAD_DOWN)),
    ]


def cartesian_inputs(pad: Any, deadzone: float) -> list[float]:
    """Return XYZ/RPY jog input for end-effector (DLS) mode.

    左摇杆：上/下 = X 前后，左/右 = Y 左右。
    右摇杆：上/下 = Z 升降，左/右 = Yaw 旋转。
    十字键：左/右 = Roll，上/下 = Pitch。

    Axes are independent: an orientation command or right-stick movement never
    suppresses left-stick translation.  (Earlier versions zeroed XYZ whenever
    the right stick or the D-pad was active, so a little right-stick drift made
    the left stick look dead.)
    """
    buttons = int(pad.buttons)
    z = normalize_axis(int(pad.right_y), deadzone)
    yaw = normalize_axis(int(pad.right_x), deadzone)
    # Translation wins consistently with the runtime's XYZ priority. Never
    # discard Z in favour of Yaw only to discard Yaw again for left-stick XYZ.
    # Yaw requires returning the vertical axis to its deadzone.
    if z:
        yaw = 0.0
    roll = float(bool(buttons & DPAD_RIGHT)) - float(
        bool(buttons & DPAD_LEFT)
    )
    pitch = float(bool(buttons & DPAD_UP)) - float(
        bool(buttons & DPAD_DOWN)
    )
    return [
        normalize_axis(int(pad.left_y), deadzone),
        -normalize_axis(int(pad.left_x), deadzone),
        z,
        roll,
        pitch,
        yaw,
    ]
