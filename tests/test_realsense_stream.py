"""Preview stream failure handling without accessing cameras or windows."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from robotarm.visualization.realsense_stream import main


class RealSenseStreamTests(unittest.TestCase):
    def run_stream(self, events, keys=()):
        pipeline = Mock()
        pipeline.wait_for_frames.side_effect = events
        device = Mock()
        device.get_info.side_effect = lambda field: {
            "serial": "260322272554", "name": "D405", "usb": "3.2",
        }[field]
        rs = Mock()
        rs.camera_info = SimpleNamespace(
            serial_number="serial", name="name", usb_type_descriptor="usb",
        )
        rs.context.return_value.query_devices.return_value = [device]
        rs.pipeline.return_value = pipeline
        cv2 = Mock()
        cv2.waitKey.side_effect = keys
        with patch.dict("sys.modules", {
            "pyrealsense2": rs, "cv2": cv2, "numpy": Mock(),
        }), patch(
            "robotarm.visualization.realsense_stream.parse_args",
            return_value=SimpleNamespace(
                serials=["260322272554"], color_only=True,
                depth=False, fullscreen=False,
            ),
        ), patch("builtins.print"):
            try:
                result = main()
            finally:
                pipeline.stop.assert_called_once()
                cv2.destroyAllWindows.assert_called_once()
        return result, pipeline

    def test_stopped_pipeline_exits_immediately_with_original_error(self):
        error = RuntimeError("wait_for_frames cannot be called before start()")
        with self.assertRaisesRegex(RuntimeError, "260322272554.*before start") as ctx:
            self.run_stream([error])
        self.assertIs(ctx.exception.__cause__, error)

    def test_genuine_timeouts_stop_after_five_consecutive_waits(self):
        errors = [RuntimeError("Frame didn't arrive within 5000")] * 5
        with self.assertRaisesRegex(RuntimeError, "5 consecutive"):
            self.run_stream(errors, [0] * 4)

    def test_successful_frame_resets_timeout_counter(self):
        timeout = RuntimeError("Frame didn't arrive within 5000")
        frame = Mock()
        frame.get_color_frame.return_value = None
        result, pipeline = self.run_stream(
            [timeout] * 4 + [frame] + [timeout] * 4 + [frame],
            [0] * 9 + [ord("q")],
        )
        self.assertEqual(result, 0)
        self.assertEqual(pipeline.wait_for_frames.call_count, 10)

    def test_ctrl_c_cleans_up_without_traceback(self):
        result, _ = self.run_stream([KeyboardInterrupt()])
        self.assertEqual(result, 130)


if __name__ == "__main__":
    unittest.main()
