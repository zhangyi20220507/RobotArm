"""手柄遥操作包：设备读取、输入解析、目标求解与数采运行时。

对外保持与原 ``robotarm.teleop.gamepad`` 模块相同的设备层 API。
"""

from .device import (  # noqa: F401
    A,
    B,
    BACK,
    DPAD_DOWN,
    DPAD_LEFT,
    DPAD_RIGHT,
    DPAD_UP,
    L3,
    LB,
    R3,
    RB,
    START,
    X,
    Y,
    ButtonTracker,
    GamepadReader,
    XInputGamepad,
    XInputState,
    add_gamepad_arguments,
    button_names,
    normalize_axis,
    normalize_trigger,
    speed_levels,
)
