"""Metadata/review tests for transactional dataset extension."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from robotarm.modeling.dataset_extend import (
    _base_session,
    _extended_reviews,
    _source_sessions,
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


if __name__ == "__main__":
    unittest.main()
