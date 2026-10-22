"""跨平台手柄输入设备：Windows XInput 与 Linux joydev。

原 robotarm/teleop/gamepad.py，作为包的设备层保留；包内其余模块提供
输入解析（inputs）、目标求解（targets）与采集运行时（runtime）。
"""

from __future__ import annotations

import ctypes
import glob
import math
import os
import subprocess
import struct
from ctypes import wintypes
from types import SimpleNamespace
from typing import Any


DPAD_UP = 0x0001
DPAD_DOWN = 0x0002
DPAD_LEFT = 0x0004
DPAD_RIGHT = 0x0008
START = 0x0010
BACK = 0x0020
L3 = 0x0040
R3 = 0x0080
LB = 0x0100
RB = 0x0200
A = 0x1000
B = 0x2000
X = 0x4000
Y = 0x8000


class XInputGamepad(ctypes.Structure):
    _fields_ = [
        ("buttons", ctypes.c_ushort),
        ("left_trigger", ctypes.c_ubyte),
        ("right_trigger", ctypes.c_ubyte),
        ("left_x", ctypes.c_short),
        ("left_y", ctypes.c_short),
        ("right_x", ctypes.c_short),
        ("right_y", ctypes.c_short),
    ]


class XInputState(ctypes.Structure):
    _fields_ = [("packet_number", ctypes.c_ulong), ("gamepad", XInputGamepad)]


def _load_xinput() -> Any:
    for name in ("xinput1_4", "xinput1_3", "xinput9_1_0"):
        try:
            library = ctypes.WinDLL(name)
            library.XInputGetState.argtypes = [
                wintypes.DWORD, ctypes.POINTER(XInputState)
            ]
            library.XInputGetState.restype = wintypes.DWORD
            return library
        except OSError:
            continue
    raise RuntimeError("找不到 Windows XInput DLL，无法读取游戏手柄。")


class _WindowsGamepadReader:
    ERROR_DEVICE_NOT_CONNECTED = 1167

    def __init__(self, index: int) -> None:
        self.index = index
        self.library = _load_xinput()

    def read(self) -> XInputGamepad:
        state = XInputState()
        result = self.library.XInputGetState(self.index, ctypes.byref(state))
        if result == self.ERROR_DEVICE_NOT_CONNECTED:
            raise ConnectionError(f"XInput 手柄 {self.index} 已断开。")
        if result != 0:
            raise OSError(result, f"XInputGetState 失败，错误码 {result}")
        return state.gamepad


def _linux_gamepad_connection(path: str) -> str:
    """Classify a Linux joystick as wired USB, Bluetooth, or unknown."""
    try:
        result = subprocess.run(
            ["udevadm", "info", "--query=property", "--name", path],
            check=False, capture_output=True, text=True, timeout=2.0,
        )
        properties = dict(
            line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
        )
        bus = properties.get("ID_BUS", "").lower()
        if bus == "usb":
            return "wired"
        if bus == "bluetooth":
            return "bluetooth"
    except (OSError, subprocess.SubprocessError):
        pass
    name = os.path.basename(path)
    sysfs_path = os.path.realpath(f"/sys/class/input/{name}/device").lower()
    if "bluetooth" in sysfs_path or "uhid" in sysfs_path:
        return "bluetooth"
    if "usb" in sysfs_path:
        return "wired"
    return "unknown"


class _LinuxGamepadReader:
    """Read Linux joydev devices using their kernel semantic button map."""

    EVENT = struct.Struct("IhBB")
    FALLBACK_BUTTON_MAP = {
        0: A, 1: B, 2: X, 3: Y, 4: LB, 5: RB,
        6: BACK, 7: START, 9: L3, 10: R3,
    }
    # Linux input-event semantic button codes. Querying JSIOCGBTNMAP avoids
    # assuming Xbox button indices: a native DualShock 4, for example, puts
    # Share/Options at 8/9 and L3/R3 at 10/11.
    BUTTON_CODE_MAP = {
        0x130: A,      # BTN_SOUTH (Cross / A)
        0x131: B,      # BTN_EAST  (Circle / B)
        0x133: Y,      # BTN_NORTH (Triangle / Y)
        0x134: X,      # BTN_WEST  (Square / X)
        0x136: LB,     # BTN_TL
        0x137: RB,     # BTN_TR
        0x13A: BACK,   # BTN_SELECT (Share / Back)
        0x13B: START,  # BTN_START  (Options / Start)
        0x13D: L3,     # BTN_THUMBL
        0x13E: R3,     # BTN_THUMBR
    }

    def __init__(self, index: int, connection: str | None = None) -> None:
        self.index = index
        if connection is None:
            self.path = f"/dev/input/js{index}"
        else:
            candidates = [
                path for path in sorted(glob.glob("/dev/input/js*"))
                if _linux_gamepad_connection(path) == connection
            ]
            if index >= len(candidates):
                detected = ", ".join(
                    f"{path}({_linux_gamepad_connection(path)})"
                    for path in sorted(glob.glob("/dev/input/js*"))
                ) or "none"
                label = "wired/USB" if connection == "wired" else "Bluetooth"
                raise ConnectionError(
                    f"No {label} gamepad at selection index {index} (detected: {detected})."
                )
            self.path = candidates[index]
        if not os.path.exists(self.path):
            available = ", ".join(glob.glob("/dev/input/js*")) or "无"
            raise ConnectionError(
                f"未找到 Linux 手柄 {self.path}（当前设备：{available}）。"
                "请重新插入手柄，并用 ls /dev/input/js* 检查。"
            )
        try:
            self.fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)
        except PermissionError as exc:
            raise PermissionError(
                f"没有权限读取 {self.path}。请将当前用户加入 input 组："
                "sudo usermod -aG input $USER，然后注销并重新登录。"
            ) from exc
        self.button_map = self._query_button_map()
        self.axes = [0] * 8
        self.buttons_down: set[int] = set()

    def _query_button_map(self) -> dict[int, int]:
        """Map joydev indices through their kernel semantic button codes."""
        try:
            import array
            import fcntl

            codes = array.array("H", [0] * 512)
            fcntl.ioctl(self.fd, 0x84006A34, codes, True)  # JSIOCGBTNMAP
            mapped = {
                index: self.BUTTON_CODE_MAP[code]
                for index, code in enumerate(codes)
                if code in self.BUTTON_CODE_MAP
            }
            if mapped:
                return mapped
        except (ImportError, OSError):
            pass
        return dict(self.FALLBACK_BUTTON_MAP)

    def _drain(self) -> None:
        while True:
            try:
                data = os.read(self.fd, self.EVENT.size)
            except BlockingIOError:
                return
            if not data:
                raise ConnectionError(f"Linux 手柄 {self.path} 已断开。")
            if len(data) != self.EVENT.size:
                continue
            _, value, event_type, number = self.EVENT.unpack(data)
            event_type &= 0x7F
            if event_type == 0x01:
                if value:
                    self.buttons_down.add(number)
                else:
                    self.buttons_down.discard(number)
            elif event_type == 0x02 and number < len(self.axes):
                self.axes[number] = value

    def read(self) -> Any:
        self._drain()
        buttons = 0
        for number in self.buttons_down:
            buttons |= self.button_map.get(number, 0)
        dpad_x = self.axes[6] if len(self.axes) > 6 else 0
        dpad_y = self.axes[7] if len(self.axes) > 7 else 0
        if dpad_x < -16000:
            buttons |= DPAD_LEFT
        elif dpad_x > 16000:
            buttons |= DPAD_RIGHT
        if dpad_y < -16000:
            buttons |= DPAD_UP
        elif dpad_y > 16000:
            buttons |= DPAD_DOWN

        def trigger(axis: int) -> int:
            value = self.axes[axis] if axis < len(self.axes) else -32767
            return max(0, min(255, round((value + 32767) * 255 / 65534)))

        return SimpleNamespace(
            buttons=buttons,
            left_trigger=trigger(2),
            right_trigger=trigger(5),
            left_x=self.axes[0],
            left_y=-self.axes[1],
            right_x=self.axes[3],
            right_y=-self.axes[4],
        )

    def __del__(self) -> None:
        fd = getattr(self, "fd", None)
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


class GamepadReader:
    """Select XInput on Windows and /dev/input/jsN on Linux."""

    def __new__(cls, index: int, connection: str | None = None):
        if os.name == "nt":
            return _WindowsGamepadReader(index)
        if os.name == "posix":
            return _LinuxGamepadReader(index, connection)
        raise RuntimeError(f"不支持的手柄平台：{os.name}")


class ButtonTracker:
    def __init__(self) -> None:
        self.previous = 0
        self.held_since: dict[int, float] = {}
        self.long_fired: set[int] = set()

    def update(self, buttons: int, now: float) -> None:
        changed = self.previous ^ buttons
        for mask in (START, BACK, X, Y):
            if changed & mask:
                if buttons & mask:
                    self.held_since[mask] = now
                else:
                    self.held_since.pop(mask, None)
                    self.long_fired.discard(mask)
        self.previous = buttons

    def long_press(self, buttons: int, mask: int, now: float, seconds: float) -> bool:
        started = self.held_since.get(mask)
        if (
            buttons & mask
            and started is not None
            and mask not in self.long_fired
            and now - started >= seconds
        ):
            self.long_fired.add(mask)
            return True
        return False


def add_gamepad_arguments(parser: Any, config: dict) -> None:
    """Register gamepad input arguments; defaults come from the workflow YAML."""
    parser.add_argument(
        "--gamepad-index", type=int, default=config["gamepad-index"],
        help="controller index: XInput 0..3 or Linux /dev/input/jsN",
    )
    parser.add_argument("--deadzone", type=float, default=config["deadzone"])
    parser.add_argument("--trigger-deadzone", type=float, default=config["trigger-deadzone"])
    parser.add_argument("--control-rate", type=float, default=config["control-rate"])
    parser.add_argument("--hold-seconds", type=float, default=config["hold-seconds"])


def normalize_axis(value: int, deadzone: float) -> float:
    raw = max(-1.0, min(1.0, value / 32767.0))
    magnitude = abs(raw)
    if magnitude <= deadzone:
        return 0.0
    return math.copysign((magnitude - deadzone) / (1.0 - deadzone), raw)


def normalize_trigger(value: int, deadzone: float) -> float:
    raw = max(0.0, min(1.0, value / 255.0))
    if raw <= deadzone:
        return 0.0
    return (raw - deadzone) / (1.0 - deadzone)


def button_names(buttons: int) -> str:
    """Return normalized XInput names for diagnostics and safety checks."""
    names = [
        name for mask, name in (
            (DPAD_UP, "DPad-Up"), (DPAD_DOWN, "DPad-Down"),
            (DPAD_LEFT, "DPad-Left"), (DPAD_RIGHT, "DPad-Right"),
            (START, "Start/Options"), (BACK, "Back/Share"),
            (L3, "L3"), (R3, "R3"), (LB, "LB/L1"), (RB, "RB/R1"),
            (A, "A/Cross"), (B, "B/Circle"),
            (X, "X/Square"), (Y, "Y/Triangle"),
        )
        if buttons & mask
    ]
    return "+".join(names) if names else "-"


def speed_levels(initial: int) -> tuple[list[int], int]:
    levels = [2, 5, 10, 15, 20, 30]
    if initial not in levels:
        levels.append(initial)
        levels.sort()
    return levels, levels.index(initial)
