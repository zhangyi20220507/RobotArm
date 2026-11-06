"""Raw recorder creates on-disk sessions only after recording starts."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robotarm.record.recorder import RawEpisodeRecorder


class FakeCameras:
    def __init__(self, *_args, **_kwargs):
        self.metadata = {"global": {"serial": "camera-1"}}
        self.workers = {"global": object()}
        self.closed = False

    def wait_ready(self):
        return None

    def close(self):
        self.closed = True


class FakeWriter:
    def __init__(self, path, *_args, **_kwargs):
        self.path = path

    def close(self):
        return None


def make_recorder(root: Path) -> RawEpisodeRecorder:
    return RawEpisodeRecorder(
        root,
        "pick cup",
        "setup-1",
        "",
        20,
        640,
        480,
        {"global": "camera-1"},
        {"model": "Piper"},
        preview=False,
    )


class LazySessionTests(unittest.TestCase):
    def test_init_and_close_without_recording_leave_no_dataset_folder(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "robotarm.record.recorder.shutil.which", return_value="/usr/bin/ffmpeg"
        ), patch(
            "robotarm.record.recorder.RoleCameras", FakeCameras
        ):
            base = Path(temporary) / "raw"
            recorder = make_recorder(base)
            self.assertIsNone(recorder.root)
            self.assertFalse(base.exists())
            recorder.close()
            self.assertFalse(base.exists())

    def test_first_start_creates_session_metadata_and_episode_directory(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "robotarm.record.recorder.shutil.which", return_value="/usr/bin/ffmpeg"
        ), patch(
            "robotarm.record.recorder.RoleCameras", FakeCameras
        ), patch(
            "robotarm.record.recorder.FFmpegWriter", FakeWriter
        ):
            base = Path(temporary) / "raw"
            recorder = make_recorder(base)
            recorder.start_episode()
            self.assertIsNotNone(recorder.root)
            self.assertTrue((recorder.root / "session.json").is_file())
            self.assertTrue(
                (recorder.root / "episodes" / ".episode_000000.in_progress").is_dir()
            )
            recorder.discard_episode()
            recorder.close()

    def test_failed_first_writer_removes_new_empty_session(self):
        class BrokenWriter:
            def __init__(self, *_args, **_kwargs):
                raise RuntimeError("encoder failed")

        with tempfile.TemporaryDirectory() as temporary, patch(
            "robotarm.record.recorder.shutil.which", return_value="/usr/bin/ffmpeg"
        ), patch(
            "robotarm.record.recorder.RoleCameras", FakeCameras
        ), patch(
            "robotarm.record.recorder.FFmpegWriter", BrokenWriter
        ):
            base = Path(temporary) / "raw"
            recorder = make_recorder(base)
            with self.assertRaisesRegex(RuntimeError, "encoder failed"):
                recorder.start_episode()
            self.assertIsNone(recorder.root)
            self.assertEqual(list(base.iterdir()), [])
            recorder.close()


if __name__ == "__main__":
    unittest.main()
