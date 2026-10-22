"""RealSense RGB 相机访问层：持续保留最新帧及其传感器/主机时间戳。

迁移自 piper_ctrl/control_scripts/src/raw_dataset_recorder.py 的相机部分。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

from .window import request_topmost


def _color_profiles(device: Any, rs: Any) -> list[tuple[int, int, int]]:
    profiles: list[tuple[int, int, int]] = []
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            try:
                video = profile.as_video_stream_profile()
                if (
                    profile.stream_type() == rs.stream.color
                    and profile.format() == rs.format.bgr8
                ):
                    profiles.append((video.width(), video.height(), video.fps()))
            except RuntimeError:
                continue
    return sorted(set(profiles))


def _choose_profile(
    profiles: list[tuple[int, int, int]], width: int, height: int, fps: int
) -> tuple[int, int, int]:
    exact = [profile for profile in profiles if profile[:2] == (width, height)]
    fast = [profile for profile in exact if profile[2] >= fps]
    if fast:
        return min(fast, key=lambda item: item[2])
    if exact:
        return max(exact, key=lambda item: item[2])
    available = ", ".join(f"{w}x{h}@{rate}" for w, h, rate in profiles) or "none"
    raise RuntimeError(
        f"RealSense does not support {width}x{height} BGR8; available: {available}"
    )


class RealSenseWorker:
    """Continuously retain the newest RGB frame and its sensor/host timestamps."""

    def __init__(
        self,
        context: Any,
        rs: Any,
        device: Any,
        role: str,
        serial: str,
        width: int,
        height: int,
        dataset_fps: int,
    ) -> None:
        self.rs = rs
        self.role = role
        self.serial = serial
        self.pipeline = rs.pipeline(context)
        width, height, stream_fps = _choose_profile(
            _color_profiles(device, rs), width, height, dataset_fps
        )
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(
            rs.stream.color, width, height, rs.format.bgr8, stream_fps
        )
        try:
            self.profile = self.pipeline.start(config)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Cannot start {role} camera {serial} at "
                f"{width}x{height}@{stream_fps}: {exc}"
            ) from exc
        device = self.profile.get_device()
        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_profile.get_intrinsics()
        self.metadata = {
            "role": role,
            "model": device.get_info(rs.camera_info.name),
            "serial": serial,
            "usb_type": device.get_info(rs.camera_info.usb_type_descriptor),
            "stream": {
                "width": width,
                "height": height,
                "hardware_fps": stream_fps,
                "dataset_fps": dataset_fps,
                "pixel_format": "BGR8",
            },
            "color_intrinsics": {
                "width": intrinsics.width,
                "height": intrinsics.height,
                "fx": intrinsics.fx,
                "fy": intrinsics.fy,
                "ppx": intrinsics.ppx,
                "ppy": intrinsics.ppy,
                "distortion_model": str(intrinsics.model),
                "coefficients": list(intrinsics.coeffs),
            },
            "extrinsics": None,
            "extrinsics_calibrated": False,
        }
        self._lock = threading.Lock()
        self._latest: tuple[Any, float, float, int] | None = None
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"realsense-{role}-{serial}", daemon=True
        )
        self._thread.start()
        print(
            f"[camera] {role}: {self.metadata['model']} serial={serial}, "
            f"USB {self.metadata['usb_type']}, {width}x{height}@{stream_fps}"
        )

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                frame = frames.get_color_frame()
                if not frame:
                    continue
                import numpy as np

                image = np.asanyarray(frame.get_data()).copy()
                sample = (
                    image,
                    time.monotonic(),
                    float(frame.get_timestamp()) / 1000.0,
                    int(frame.get_frame_number()),
                )
                with self._lock:
                    self._latest = sample
        except BaseException as exc:
            if not self._stop.is_set():
                self._error = exc

    def latest(self) -> tuple[Any, float, float, int]:
        if self._error is not None:
            raise RuntimeError(
                f"Camera {self.role} ({self.serial}) failed: {self._error}"
            ) from self._error
        with self._lock:
            if self._latest is None:
                raise RuntimeError(f"Camera {self.role} has no RGB frame yet")
            return self._latest

    def close(self) -> None:
        self._stop.set()
        try:
            self.pipeline.stop()
        finally:
            self._thread.join(timeout=2.0)


class RoleCameras:
    """Manage a set of RealSense cameras keyed by their recording role."""

    def __init__(
        self,
        serials: dict[str, str],
        width: int,
        height: int,
        fps: int,
        preview: bool = True,
        window_title: str = "RobotArm Data Collection",
        always_on_top: bool = True,
    ) -> None:
        try:
            import cv2
            import numpy as np
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "Camera dependency import failed: "
                f"{exc.name or type(exc).__name__}: {exc}. "
                "Install pyrealsense2, opencv-python and numpy in the active Python."
            ) from exc
        self.cv2 = cv2
        self.np = np
        self.preview = preview
        self.window = window_title
        self.always_on_top = always_on_top
        self._topmost_attempts = 0
        self._topmost_applied = False
        self._fullscreen = False
        self._preview_keys = deque(maxlen=256)
        context = rs.context()
        devices = {
            device.get_info(rs.camera_info.serial_number): device
            for device in context.query_devices()
        }
        missing = [serial for serial in serials.values() if serial not in devices]
        if missing:
            found = {
                serial: device.get_info(rs.camera_info.name)
                for serial, device in devices.items()
            }
            raise RuntimeError(f"Missing RealSense {missing}; detected: {found or 'none'}")
        self.workers: dict[str, RealSenseWorker] = {}
        try:
            for role, serial in serials.items():
                self.workers[role] = RealSenseWorker(
                    context, rs, devices[serial], role, serial, width, height, fps
                )
        except Exception:
            self.close()
            raise
        self.viewer_ok = False
        if preview:
            try:
                cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window, width * len(serials), height)
                cv2.setMouseCallback(self.window, self._mouse_event)
                self.viewer_ok = True
            except Exception as exc:
                print(f"[camera] Preview unavailable ({exc}); recording headlessly.")

    @property
    def metadata(self) -> dict[str, Any]:
        return {role: worker.metadata for role, worker in self.workers.items()}

    def wait_ready(self, timeout: float = 8.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                for worker in self.workers.values():
                    worker.latest()
                return
            except RuntimeError:
                time.sleep(0.02)
        raise RuntimeError("Timed out waiting for all configured RGB cameras")

    def latest(self) -> dict[str, tuple[Any, float, float, int]]:
        return {role: worker.latest() for role, worker in self.workers.items()}

    def show(self, frames: dict[str, tuple[Any, float, float, int]], active: bool) -> None:
        if not self.viewer_ok:
            return
        views = []
        for role, (image, _, _, _) in frames.items():
            view = image.copy()
            label = f"{role}  {'REC' if active else 'PREVIEW'}"
            color = (0, 0, 255) if active else (0, 255, 0)
            self.cv2.putText(
                view, label, (12, 28), self.cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
            )
            views.append(view)
        combined = self.np.hstack(views)
        try:
            self.cv2.imshow(self.window, combined)
            read_key = getattr(self.cv2, "waitKeyEx", self.cv2.waitKey)
            key = read_key(1)
            if isinstance(key, int) and key >= 0:
                if key in (ord("f"), ord("F"), 0xffc8, 16777274, 0x7a0000):
                    self.set_fullscreen(not getattr(self, "_fullscreen", False))
                elif key == 27:
                    self.set_fullscreen(False)
                else:
                    self._preview_keys.append(key & 0xff)
            if (
                self.always_on_top
                and not self._topmost_applied
                and self._topmost_attempts < 10
            ):
                self._topmost_attempts += 1
                self._topmost_applied = request_topmost(self.window, self.cv2)
                if not self._topmost_applied and self._topmost_attempts == 10:
                    print(
                        "[camera] Preview could not be pinned above other windows; "
                        "recording continues."
                    )
        except Exception as exc:
            self.viewer_ok = False
            print(f"\n[camera] Preview stopped ({exc}); recording continues headlessly.")

    def set_fullscreen(self, enabled: bool) -> None:
        try:
            self.cv2.setWindowProperty(self.window, self.cv2.WND_PROP_FULLSCREEN,
                self.cv2.WINDOW_FULLSCREEN if enabled else self.cv2.WINDOW_NORMAL)
            self._fullscreen = enabled
            self._topmost_applied = False
            self._topmost_attempts = 0
        except Exception as exc:
            print(f"[camera] Fullscreen unavailable: {exc}")

    def _mouse_event(self, event, x, y, flags, param) -> None:
        if event == self.cv2.EVENT_LBUTTONDBLCLK:
            self.set_fullscreen(not self._fullscreen)

    def poll_keys(self):
        keys = list(self._preview_keys)
        self._preview_keys.clear()
        return keys

    def close(self) -> None:
        for worker in getattr(self, "workers", {}).values():
            worker.close()
        self.workers = {}
        if getattr(self, "viewer_ok", False):
            try:
                self.cv2.destroyWindow(self.window)
                self.cv2.waitKey(1)
            except Exception:
                pass
