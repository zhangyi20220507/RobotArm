"""Reviewed BC demonstrations expose rewards without touching dataset files."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from robotarm.modeling.reviewed_offline_rl import (
    ReviewedOfflineRLDataset,
    reviewed_episodes,
)


class _Frames:
    def __init__(self):
        self.rows = [
            {"episode_index": torch.tensor(3), "frame_index": torch.tensor(i),
             "action": torch.zeros(7)} for i in range(3)
        ] + [
            {"episode_index": torch.tensor(4), "frame_index": torch.tensor(i),
             "action": torch.zeros(7)} for i in range(2)
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class ReviewedOfflineRLTests(unittest.TestCase):
    def test_sparse_rewards_and_episode_boundaries(self):
        dataset = ReviewedOfflineRLDataset(
            _Frames(), {3: {"frames": 3, "outcome": "success"},
                        4: {"frames": 2, "outcome": "failure"}}
        )
        self.assertEqual(
            [float(dataset[i]["next.reward"].item()) for i in range(len(dataset))],
            [0, 0, 1, 0, 0],
        )
        self.assertEqual(
            [bool(dataset[i]["next.done"].item()) for i in range(len(dataset))],
            [False, False, True, False, True],
        )

    def test_requires_manual_labels_and_full_human_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            meta = root / "meta"
            meta.mkdir()
            manifest = {
                "repo_id": "local/test",
                "episodes": [
                    {"lerobot_episode_index": 0, "frames": 3,
                     "review": {"status": "keep", "review_origin": "manual",
                                "labels": {"task_outcome": "success"}}},
                    {"lerobot_episode_index": 1, "frames": 2,
                     "review": {"status": "keep", "review_origin": "default",
                                "labels": {"task_outcome": "failure"}}},
                ],
            }
            (meta / "piper_conversion.json").write_text(json.dumps(manifest))
            self.assertEqual(
                reviewed_episodes(root)[1],
                {0: {"frames": 3, "outcome": "success"}},
            )
            manifest["includes_expert_mask_exports"] = True
            (meta / "piper_conversion.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "full-human"):
                reviewed_episodes(root)
