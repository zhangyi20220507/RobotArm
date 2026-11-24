"""Raw recorder creates on-disk sessions only after recording starts."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

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

    def submit(self, _image):
        return None


def make_recorder(
    root: Path, extra_sample_fields=(), session_prefix: str = ""
) -> RawEpisodeRecorder:
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
        extra_sample_fields=extra_sample_fields,
        session_prefix=session_prefix,
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

    def test_session_prefix_distinguishes_hg_dagger_sessions(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "robotarm.record.recorder.shutil.which", return_value="/usr/bin/ffmpeg"
        ), patch(
            "robotarm.record.recorder.RoleCameras", FakeCameras
        ), patch(
            "robotarm.record.recorder.FFmpegWriter", FakeWriter
        ):
            recorder = make_recorder(
                Path(temporary) / "raw", session_prefix="hg_dagger"
            )
            recorder.prepare_episode()

            self.assertTrue(recorder.root.name.startswith("hg_dagger_"))
            self.assertRegex(
                recorder.root.name,
                r"^hg_dagger_\d{8}_\d{6}_\d{6}$",
            )
            recorder.close()

    def test_prepared_encoder_starts_episode_without_recreating_writer(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "robotarm.record.recorder.shutil.which", return_value="/usr/bin/ffmpeg"
        ), patch(
            "robotarm.record.recorder.RoleCameras", FakeCameras
        ), patch(
            "robotarm.record.recorder.FFmpegWriter", side_effect=FakeWriter
        ) as writer_factory:
            recorder = make_recorder(Path(temporary) / "raw")
            recorder.prepare_episode()
            working_dir = recorder.working_dir

            self.assertTrue(recorder.prepared)
            self.assertFalse(recorder.active)
            self.assertEqual(writer_factory.call_count, 1)

            recorder.start_episode(started_at=123.0)
            self.assertTrue(recorder.active)
            self.assertFalse(recorder.prepared)
            self.assertEqual(recorder.working_dir, working_dir)
            self.assertEqual(recorder.started_at, 123.0)
            self.assertEqual(writer_factory.call_count, 1)
            recorder.discard_episode()
            recorder.close()

    def test_close_discards_prepared_empty_episode(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "robotarm.record.recorder.shutil.which", return_value="/usr/bin/ffmpeg"
        ), patch(
            "robotarm.record.recorder.RoleCameras", FakeCameras
        ), patch(
            "robotarm.record.recorder.FFmpegWriter", FakeWriter
        ):
            base = Path(temporary) / "raw"
            recorder = make_recorder(base)
            recorder.prepare_episode()
            session_root = recorder.root
            recorder.close()

            self.assertIsNotNone(session_root)
            self.assertFalse(session_root.exists())
            self.assertIsNone(recorder.root)

    def test_full_trial_metadata_summarizes_expert_frames(self):
        fields = (
            "control_source",
            "expert_mask",
            "intervention_id",
            "command_sent",
        )
        with tempfile.TemporaryDirectory() as temporary, patch(
            "robotarm.record.recorder.shutil.which", return_value="/usr/bin/ffmpeg"
        ), patch(
            "robotarm.record.recorder.RoleCameras", FakeCameras
        ), patch(
            "robotarm.record.recorder.FFmpegWriter", FakeWriter
        ):
            recorder = make_recorder(Path(temporary) / "raw", fields)
            recorder.start_episode(started_at=10.0)
            image = np.zeros((2, 2, 3), dtype=np.uint8)
            for index, source in enumerate((0, 1, 2, 2)):
                now = 10.0 + index * 0.051
                recorder.capture(
                    now,
                    [0.0] * 7,
                    [0.0] * 7,
                    [0.0] * 6,
                    [0.0] * 6,
                    {
                        "control_source": source,
                        "expert_mask": int(source == 2),
                        "intervention_id": -1 if source == 0 else 0,
                        "command_sent": 1,
                    },
                    show_preview=False,
                    frames={"global": (image, now, now, index)},
                )
            destination = recorder.save_episode()
            metadata = json.loads((destination / "episode.json").read_text())
            self.assertEqual(
                metadata["control_source_frames"],
                {"novice": 1, "armed": 1, "expert": 2},
            )
            self.assertEqual(metadata["interventions"], 1)
            self.assertEqual(metadata["expert_interventions"], 1)
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
