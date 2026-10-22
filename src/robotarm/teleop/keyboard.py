"""Cross-platform immediate keyboard input for control scripts."""
from __future__ import annotations
from collections import deque
import os
import sys
import threading
import time

def _normalise(key: str) -> str:
    if key == "\x03":
        raise KeyboardInterrupt
    if key in ("\r", "\n"):
        return "\r"
    return key.lower()

def read_key() -> str:
    """Read one key immediately without requiring Enter."""
    if os.name == "nt":
        import msvcrt
        key = msvcrt.getwch()
        if key in ("\x00", "\xe0"):
            return msvcrt.getwch()
        return _normalise(key)
    if not sys.stdin.isatty():
        raise RuntimeError("键盘控制需要在交互式终端中运行（stdin 必须是 TTY）。")
    import termios
    import tty
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = os.read(fd, 1).decode(errors="ignore")
        if key == "\x1b":
            import select
            ready, _, _ = select.select([fd], [], [], 0.03)
            if ready:
                suffix = os.read(fd, 2).decode(errors="ignore")
                if suffix == "[A":
                    return "H"
                if suffix == "[B":
                    return "P"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)
    return _normalise(key)

def read_key_if_available() -> str | None:
    """Return one pending key without blocking, or None."""
    if os.name == "nt":
        import msvcrt
        return read_key() if msvcrt.kbhit() else None
    if not sys.stdin.isatty():
        return None
    import select
    import termios
    import tty
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ready, _, _ = select.select([fd], [], [], 0)
        if not ready:
            return None
        key = os.read(fd, 1).decode(errors="ignore")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)
    return _normalise(key)


class TerminalKeyEventReader:
    """Continuously timestamp key events without blocking a control loop."""

    def __init__(self, max_events: int = 256) -> None:
        self._events = deque(maxlen=max_events)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._fd = None
        self._previous = None
        self._error = None

    def start(self) -> None:
        if os.name != "posix":
            raise RuntimeError("后台键盘监听目前只支持 POSIX 终端。")
        if not sys.stdin.isatty():
            raise RuntimeError(
                "键盘控制需要在交互式终端中运行（stdin 必须是 TTY）。"
            )
        import termios
        import tty

        self._fd = sys.stdin.fileno()
        self._previous = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._read_loop, name="terminal-key-reader", daemon=True
        )
        self._thread.start()

    def _read_loop(self) -> None:
        import select

        pending_alt = False
        try:
            while not self._stop.is_set():
                ready, _, _ = select.select([self._fd], [], [], 0.05)
                if not ready:
                    continue
                data = os.read(self._fd, 32)
                observed_at = time.monotonic()
                events = []
                for byte in data:
                    key = chr(byte)
                    if pending_alt:
                        events.append((f"alt+{key.lower()}", observed_at))
                        pending_alt = False
                        continue
                    if key == "\x1b":
                        pending_alt = True
                        continue
                    if key not in ("\x03", "\r", "\n"):
                        key = key.lower()
                    events.append((key, observed_at))
                with self._lock:
                    self._events.extend(events)
        except Exception as exc:  # surfaced in the control thread by poll()
            self._error = exc

    def poll(self) -> list[tuple[str, float]]:
        if self._error is not None:
            raise RuntimeError("后台键盘监听失败") from self._error
        with self._lock:
            events = list(self._events)
            self._events.clear()
        if any(key == "\x03" for key, _ in events):
            raise KeyboardInterrupt
        return events

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.2)
            self._thread = None
        if self._fd is not None and self._previous is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._previous)
        self._fd = None
        self._previous = None
