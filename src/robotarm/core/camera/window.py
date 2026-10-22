"""Request an always-on-top camera preview from the desktop window manager."""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
import subprocess


class _ClientMessage(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("serial", ctypes.c_ulong),
        ("send_event", ctypes.c_int),
        ("display", ctypes.c_void_p),
        ("window", ctypes.c_ulong),
        ("message_type", ctypes.c_ulong),
        ("format", ctypes.c_int),
        ("data", ctypes.c_long * 5),
    ]


class _XEvent(ctypes.Union):
    _fields_ = [("xclient", _ClientMessage), ("pad", ctypes.c_long * 24)]


def _x11_request_topmost(title: str) -> bool:
    """Use EWMH's _NET_WM_STATE_ABOVE; OpenCV/Qt ignores TOPMOST on X11."""
    if not os.environ.get("DISPLAY"):
        return False
    try:
        result = subprocess.run(
            ["xwininfo", "-name", title], capture_output=True,
            text=True, timeout=1, check=False,
        )
        match = re.search(r"Window id:\s*(0x[0-9a-fA-F]+)", result.stdout)
        if result.returncode or match is None:
            return False
        library = ctypes.util.find_library("X11")
        if library is None:
            return False
        x11 = ctypes.CDLL(library)
        x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
        x11.XDefaultRootWindow.restype = ctypes.c_ulong
        x11.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        x11.XInternAtom.restype = ctypes.c_ulong
        x11.XSendEvent.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_long,
            ctypes.POINTER(_XEvent),
        ]
        x11.XSendEvent.restype = ctypes.c_int
        x11.XFlush.argtypes = [ctypes.c_void_p]
        x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        display = x11.XOpenDisplay(None)
        if not display:
            return False
        try:
            state = x11.XInternAtom(display, b"_NET_WM_STATE", 0)
            above = x11.XInternAtom(display, b"_NET_WM_STATE_ABOVE", 0)
            if not state or not above:
                return False
            event = _XEvent()
            event.xclient.type = 33  # ClientMessage
            event.xclient.display = display
            event.xclient.window = int(match.group(1), 16)
            event.xclient.message_type = state
            event.xclient.format = 32
            event.xclient.data[:] = (1, above, 0, 1, 0)  # ADD, application request
            sent = x11.XSendEvent(
                display, x11.XDefaultRootWindow(display), 0,
                (1 << 20) | (1 << 19), ctypes.byref(event),
            )
            x11.XFlush(display)
            return bool(sent)
        finally:
            x11.XCloseDisplay(display)
    except (OSError, subprocess.TimeoutExpired):
        return False


def request_topmost(title: str, cv2) -> bool:
    """Try OpenCV's native property, then X11; never interrupt recording."""
    property_id = getattr(cv2, "WND_PROP_TOPMOST", None)
    if property_id is not None:
        try:
            cv2.setWindowProperty(title, property_id, 1)
            if cv2.getWindowProperty(title, property_id) >= 0.5:
                return True
        except Exception:
            pass
    return _x11_request_topmost(title)
