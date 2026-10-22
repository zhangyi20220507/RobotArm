"""Camera preview tests; do not open cameras or move robot arms."""
import unittest
from unittest.mock import Mock, patch

import numpy as np

from robotarm.core.camera.realsense import RoleCameras
from robotarm.core.camera.window import request_topmost


class CameraPreviewTests(unittest.TestCase):
    def test_fullscreen_and_escape_do_not_disable_preview(self):
        cameras = RoleCameras.__new__(RoleCameras)
        cameras.viewer_ok = True
        cameras.cv2 = Mock(FONT_HERSHEY_SIMPLEX=0, WND_PROP_FULLSCREEN=0,
                           WINDOW_FULLSCREEN=1, WINDOW_NORMAL=0)
        cameras.cv2.waitKeyEx.side_effect = [ord('f'), 27, ord('x')]
        cameras.np, cameras.window = np, 'preview'
        cameras.always_on_top = False
        cameras._fullscreen, cameras._preview_keys = False, []
        frame = np.zeros((8,8,3),dtype=np.uint8)
        for expected in [True,False,False]:
            cameras.show({'global':(frame,0,0,0)},False)
            self.assertEqual(cameras._fullscreen,expected)
        self.assertTrue(cameras.viewer_ok)
        self.assertEqual(cameras.poll_keys(),[ord('x')])
        self.assertEqual(cameras.poll_keys(),[])

    def test_native_topmost_falls_back_to_x11_when_unsupported(self):
        cv2 = Mock(WND_PROP_TOPMOST=5)
        cv2.getWindowProperty.return_value = -1
        with patch("robotarm.core.camera.window._x11_request_topmost", return_value=True) as x11:
            self.assertTrue(request_topmost("preview", cv2))
        cv2.setWindowProperty.assert_called_once_with("preview", 5, 1)
        x11.assert_called_once_with("preview")

    def test_preview_is_pinned_once_and_keeps_showing(self):
        cameras = RoleCameras.__new__(RoleCameras)
        cameras.viewer_ok = True
        cameras.cv2 = Mock(FONT_HERSHEY_SIMPLEX=0)
        cameras.np = np
        cameras.window = "preview"
        cameras.always_on_top = True
        cameras._topmost_attempts = 0
        cameras._topmost_applied = False
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frames = {"global": (frame, 0, 0, 0), "wrist": (frame, 0, 0, 0)}
        with patch("robotarm.core.camera.realsense.request_topmost", return_value=True) as pin:
            cameras.show(frames, active=True)
            cameras.show(frames, active=True)
        pin.assert_called_once_with("preview", cameras.cv2)
        self.assertTrue(cameras.viewer_ok)
        self.assertEqual(cameras.cv2.imshow.call_count, 2)

    def test_pin_failure_does_not_interrupt_preview(self):
        cameras = RoleCameras.__new__(RoleCameras)
        cameras.viewer_ok = True
        cameras.cv2 = Mock(FONT_HERSHEY_SIMPLEX=0)
        cameras.np = np
        cameras.window = "preview"
        cameras.always_on_top = True
        cameras._topmost_attempts = 0
        cameras._topmost_applied = False
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frames = {"global": (frame, 0, 0, 0)}
        with patch("robotarm.core.camera.realsense.request_topmost", return_value=False):
            for _ in range(11):
                cameras.show(frames, active=False)
        self.assertTrue(cameras.viewer_ok)
        self.assertEqual(cameras.cv2.imshow.call_count, 11)
        self.assertEqual(cameras._topmost_attempts, 10)


if __name__ == "__main__":
    unittest.main()
