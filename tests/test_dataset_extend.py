"""Metadata/review tests for transactional dataset extension."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from robotarm.modeling.dataset_extend import (
    _base_session,
    _extended_reviews,
    _source_sessions,
    _validate_compatibility,
)
from robotarm.record.convert import write_json


class DatasetExtendTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_source_sessions_are_recovered_through_trim_conversion(self):
        raw = self.root / "raw"
        conversion = {
            "converter": "robotarm.dataset_trim",
            "source_conversion": {
                "source_sessions": [str(raw), str(raw)],
            },
        }
        self.assertEqual(_source_sessions(conversion), [str(raw.resolve())])

    def test_base_session_falls_back_to_source_dataset(self):
        base = self.root / "trimmed"
        source = self.root / "untrimmed"
        expected = {"format_version": "piper_raw_v1", "record_fps": 20}
        write_json(source / "meta" / "piper_raw_session.json", expected)
        write_json(base / "meta" / "trim_provenance.json", {"source_dataset": str(source)})
        actual, origin = _base_session(base, {})
        self.assertEqual(actual, expected)
        self.assertEqual(Path(origin), source / "meta" / "piper_raw_session.json")

    def test_extended_reviews_preserve_old_entries_and_mark_only_new_range(self):
        inherited = {
            "dataset_root": "/old",
            "episodes": {
                "0": {"status": "keep", "labels": {"cup_orientation": "upright"}},
                "1": {"status": "keep", "note": "old note"},
            },
        }
        result = _extended_reviews(
            inherited, self.root / "new", 2, 4,
            status="review", label="sideways", note="check",
        )
        self.assertEqual(result["episodes"]["0"], inherited["episodes"]["0"])
        self.assertEqual(result["episodes"]["1"], inherited["episodes"]["1"])
        self.assertEqual(result["episodes"]["2"]["status"], "review")
        self.assertEqual(
            result["episodes"]["3"]["labels"], {"cup_orientation": "sideways"}
        )
        self.assertEqual(result["derived_from"]["new_episode_range"], [2, 4])

    def test_extension_counts_expert_segments_not_full_hg_trials(self):
        raw = self.root / "hg_raw"
        session = {
            "format_version": "piper_raw_v1",
            "record_fps": 20,
            "state_names": [f"joint_{index}" for index in range(6)] + ["gripper"],
            "state_units": ["rad"] * 6 + ["m"],
            "action_semantics": "commanded MOVE_JS joint targets and gripper target",
            "robot": {"training_export": "expert_mask_only"},
            "cameras": {
                "global": {
                    "serial": "camera",
                    "stream": {"width": 32, "height": 24},
                }
            },
        }
        write_json(raw / "session.json", session)
        episode = raw / "episodes" / "episode_000000"
        write_json(
            episode / "episode.json",
            {
                "episode_index": 0,
                "frames": 6,
                "status": "complete",
                "task": "task",
                "camera_roles": ["global"],
            },
        )
        np.savez_compressed(
            episode / "frames.npz",
            expert_mask=np.asarray([0, 1, 1, 0, 1, 1], dtype=np.int8),
        )
        conversion = {"camera_roles": ["global"], "source_sessions": []}

        episodes, frames, details = _validate_compatibility(
            session, conversion, [raw]
        )

        self.assertEqual((episodes, frames), (2, 4))
        self.assertEqual(details[0][2], 2)


if __name__ == "__main__":
    unittest.main()
