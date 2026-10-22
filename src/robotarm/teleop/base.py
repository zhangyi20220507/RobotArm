"""遥操作输入源的统一抽象。

手柄（gamepad）与 XR/PICO 都实现该接口，下游控制环、安全限位与录制
模块只面向 TeleopInput 编程，不感知具体输入设备。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TeleopInput(ABC):
    """一个可轮询的遥操作输入源。"""

    @abstractmethod
    def poll(self) -> Any:
        """Return the latest device state in a normalized representation."""

    def close(self) -> None:
        """Release device handles; default no-op."""
