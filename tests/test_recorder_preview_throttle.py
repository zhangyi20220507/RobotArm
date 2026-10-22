import unittest
from unittest.mock import Mock

from robotarm.record.recorder import RawEpisodeRecorder


class PreviewThrottleTests(unittest.TestCase):
    def test_preview_can_be_skipped_without_skipping_capture(self):
        recorder = RawEpisodeRecorder.__new__(RawEpisodeRecorder)
        recorder.cameras = Mock()
        recorder.cameras.latest.return_value = {"global": (object(), 1.0, 0., 1)}
        recorder.active = True
        recorder.next_frame_time = 1.0
        recorder.extra_sample_fields = []
        recorder.writers = {"global": Mock()}
        recorder.started_at = 0.
        recorder.frames = recorder.dropped_samples = 0
        recorder.fps = 20
        recorder.samples = {key: [] for key in (
            "timestamp", "observation_state", "action", "observation_tcp_pose",
            "teleop_delta", "camera_host_timestamp", "camera_sensor_timestamp",
            "camera_frame_number", "camera_frame_age", "camera_skew")}
        self.assertTrue(recorder.capture(1., [], [], [], [], show_preview=False))
        recorder.cameras.show.assert_not_called()
        recorder.writers["global"].submit.assert_called_once()
        self.assertEqual(recorder.frames, 1)

    def test_default_preserves_other_callers_preview(self):
        recorder = RawEpisodeRecorder.__new__(RawEpisodeRecorder)
        recorder.cameras = Mock()
        recorder.active = False
        self.assertFalse(recorder.capture(1., [], [], [], []))
        recorder.cameras.show.assert_called_once()
