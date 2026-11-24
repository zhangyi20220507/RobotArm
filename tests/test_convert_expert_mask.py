from __future__ import annotations

from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path

import numpy as np

from robotarm.record.convert import split_expert_selections, uses_expert_mask


class ExpertMaskConversionTests(unittest.TestCase):
    def test_full_trial_session_uses_expert_export(self):
        self.assertTrue(
            uses_expert_mask(
                {"robot": {"training_export": "expert_mask_only"}}
            )
        )
        self.assertFalse(uses_expert_mask({"robot": {}}))

    def test_reviewed_trial_is_split_into_contiguous_expert_segments(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode_path = Path(temporary) / "episode_000000"
            episode_path.mkdir()
            np.savez_compressed(
                episode_path / "frames.npz",
                expert_mask=np.asarray(
                    [0, 1, 1, 0, 0, 1, 1, 1, 0], dtype=np.int8
                ),
            )
            episode = SimpleNamespace(
                path=episode_path,
                metadata={"frames": 9},
            )
            selection = {
                "episode": episode,
                "start_frame": 0,
                "end_frame": 9,
                "source_index": 0,
            }

            segments = split_expert_selections([selection])

            self.assertEqual(
                [
                    (item["start_frame"], item["end_frame"])
                    for item in segments
                ],
                [(1, 3), (5, 8)],
            )
            self.assertEqual(
                [item["expert_segment_index"] for item in segments], [0, 1]
            )

    def test_review_trim_is_intersected_with_expert_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode_path = Path(temporary) / "episode_000000"
            episode_path.mkdir()
            np.savez_compressed(
                episode_path / "frames.npz",
                expert_mask=np.asarray([0, 1, 1, 1, 1, 0], dtype=np.int8),
            )
            episode = SimpleNamespace(
                path=episode_path,
                metadata={"frames": 6},
            )
            segments = split_expert_selections(
                [{
                    "episode": episode,
                    "start_frame": 2,
                    "end_frame": 4,
                }]
            )
            self.assertEqual(len(segments), 1)
            self.assertEqual(
                (segments[0]["start_frame"], segments[0]["end_frame"]),
                (2, 4),
            )


if __name__ == "__main__":
    unittest.main()
