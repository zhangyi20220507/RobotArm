from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from robotarm.modeling.policy_recording import PolicyRecordingManager
from robotarm.record.recorder import RawEpisodeRecorder


class _Writer:
    def __init__(self, path, width, height, fps, queue_size):
        self.path = Path(path)
        self.frames = []
        self.path.write_bytes(b"video")

    def submit(self, image):
        self.frames.append(np.asarray(image).copy())

    def close(self):
        return None


class PolicyRecordingTests(unittest.TestCase):
    def test_multiple_recordings_save_combined_separate_and_joint_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "recordings"
            manager = PolicyRecordingManager(
                root,
                fps=20,
                width=16,
                height=12,
                camera_roles=["global", "wrist"],
                camera_metadata={"global": {}, "wrist": {}},
                session_metadata={"policy_type": "act"},
                writer_factory=_Writer,
                cv2_module=cv2,
                np_module=np,
            )
            self.assertFalse(root.exists())
            frame = np.zeros((12, 16, 3), dtype=np.uint8)
            for recording_index in range(2):
                manager.start()
                now = time.monotonic()
                frames = {
                    "global": (frame, now, 1.0, recording_index * 2),
                    "wrist": (frame, now, 1.1, recording_index * 2 + 1),
                }
                self.assertTrue(manager.capture(
                    now,
                    frames,
                    np.arange(7, dtype=np.float32),
                    np.arange(7, dtype=np.float32) + 1,
                    np.arange(7, dtype=np.float32) + 2,
                    trial_id=recording_index + 1,
                    motion_enabled=True,
                    command_sent=True,
                    control_event="start",
                ))
                manager.stop()
            manager.close()

            session_dirs = [path for path in root.iterdir() if path.is_dir()]
            self.assertEqual(len(session_dirs), 1)
            recordings = sorted(session_dirs[0].glob("recording_*"))
            self.assertEqual(len(recordings), 2)
            for recording in recordings:
                for name in ("combined.mp4", "global.mp4", "wrist.mp4",
                             "frames.npz", "joints.csv", "recording.json"):
                    self.assertTrue((recording / name).is_file(), name)
                with np.load(recording / "frames.npz") as arrays:
                    self.assertEqual(arrays["observation_state"].shape, (1, 7))
                    self.assertEqual(arrays["camera_frame_number"].shape, (1, 2))
                    self.assertEqual(arrays["control_event"].tolist(), ["start"])
                metadata = json.loads((recording / "recording.json").read_text())
                self.assertEqual(metadata["status"], "complete")

    def test_raw_recorder_does_not_close_borrowed_cameras(self):
        class Cameras:
            metadata = {"global": {}}
            workers = {"global": object()}

            def __init__(self):
                self.closed = False

            def wait_ready(self):
                return None

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as directory, patch(
            "robotarm.record.recorder.shutil.which", return_value="/usr/bin/ffmpeg"
        ):
            cameras = Cameras()
            recorder = RawEpisodeRecorder(
                Path(directory) / "raw",
                "task",
                "setup",
                "",
                20,
                16,
                12,
                {"global": "serial"},
                {},
                cameras=cameras,
            )
            recorder.close()
            self.assertFalse(cameras.closed)


if __name__ == "__main__":
    unittest.main()
