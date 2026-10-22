from __future__ import annotations

import argparse
import math
import threading
import time
from typing import Any


class LiveCANMonitor:
    """Single read-only Piper CAN subscriber shared by all browser requests."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._payload: dict[str, Any] = {
            "status": "offline", "state": None, "error": None, "sequence": 0,
        }
        self._updated_monotonic: float | None = None

    def start(self, request: dict[str, Any] | None = None) -> dict[str, Any]:
        request = request or {}
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self.snapshot()
            self._stop.clear()
            self._updated_monotonic = None
            self._payload = {
                "status": "connecting", "state": None, "error": None,
                "sequence": 0, "interface": str(request.get("interface", "socketcan")),
                "channel": str(request.get("channel", "can_follower1")),
                "firmware": str(request.get("firmware", "v189")),
            }
            self._thread = threading.Thread(
                target=self._run, args=(dict(self._payload),),
                name="piper-live-can", daemon=True,
            )
            self._thread.start()
        return self.snapshot()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=4.0)
        with self._lock:
            self._thread = None
            self._updated_monotonic = None
            self._payload = {
                "status": "offline", "state": None, "error": None, "sequence": 0,
            }
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            value = dict(self._payload)
            updated = self._updated_monotonic
        age_ms = None if updated is None else (time.monotonic() - updated) * 1000.0
        value["age_ms"] = age_ms
        value["fresh"] = value["status"] == "online" and age_ms is not None and age_ms <= 500
        return value

    def _run(self, settings: dict[str, Any]) -> None:
        robot = None
        try:
            from ...core.robot import piper as base

            args = argparse.Namespace(
                interface=settings["interface"], channel=settings["channel"],
                firmware=settings["firmware"], feedback_timeout=3.0,
            )
            robot, _ = base.connect_robot(args)
            base.wait_for_feedback(robot, args.feedback_timeout)
            gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
            while not self._stop.is_set():
                feedback = robot.get_joint_angles()
                gripper_status = gripper.get_gripper_status()
                if feedback is not None:
                    joints = [float(value) for value in feedback.msg]
                    if len(joints) != 6 or not all(math.isfinite(value) for value in joints):
                        raise RuntimeError("Piper joint feedback is invalid")
                    width = None
                    if gripper_status is not None and gripper_status.msg.mode == "width":
                        width = float(gripper_status.msg.value)
                    if width is not None and math.isfinite(width):
                        with self._lock:
                            sequence = int(self._payload.get("sequence", 0)) + 1
                            self._payload.update({
                                "status": "online", "state": [*joints, width],
                                "error": None, "sequence": sequence,
                                "timestamp": time.time(),
                            })
                            self._updated_monotonic = time.monotonic()
                self._stop.wait(0.05)
        except Exception as exc:
            with self._lock:
                self._payload.update({
                    "status": "error", "state": None,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                self._updated_monotonic = None
        finally:
            if robot is not None:
                try:
                    robot.disconnect()
                except Exception:
                    pass


LIVE_CAN = LiveCANMonitor()
