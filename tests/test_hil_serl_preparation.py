"""Feature preparation and pipeline checks without CAN or an ACT download."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from robotarm.modeling.hil_serl.config import Config
from robotarm.modeling.hil_serl.prepare import prepare, inspect_dataset, encoded_samples
from robotarm.modeling.hil_serl.pipeline import commands, select_act_checkpoint
from robotarm.modeling.hil_serl.replay import read_episode
from robotarm.modeling.hil_serl.safety import CANIdentityGuard, ObservationDeadline, require_fresh_feedback


class HILPreparationTests(unittest.TestCase):
    def test_deadline_retries_isolated_slow_frames_but_stops_on_third_miss(self):
        guard = ObservationDeadline(100)
        self.assertFalse(guard.retry(40, True))
        self.assertTrue(guard.retry(267.5, True))
        self.assertEqual(guard.misses, 1)
        self.assertFalse(guard.retry(45, True))
        self.assertEqual(guard.misses, 0)
        self.assertTrue(guard.retry(270.4, True))
        self.assertTrue(guard.retry(260, True))
        self.assertFalse(guard.retry(250, True))
        self.assertEqual(guard.misses, 3)
        self.assertFalse(guard.retry(250, True))

    def test_deadline_never_retries_other_faults_or_invalid_ages(self):
        guard = ObservationDeadline(100)
        self.assertTrue(guard.retry(267, True))
        self.assertFalse(guard.retry(267, False))  # camera/mode/joint/inference fault
        self.assertEqual(guard.misses, 0)
        for age in (None, float("nan"), float("inf"), 100, 40):
            self.assertFalse(guard.retry(age, True))
            self.assertEqual(guard.misses, 0)

    def test_batched_encoding_preserves_order_and_final_partial_batch(self):
        samples = [{"frame_index": i, "observation.state": torch.full((7,), float(i)),
                    "observation.images.global": torch.zeros(3, 8, 8)} for i in range(5)]
        prior = Mock()
        prior.roles = ["global"]
        prior.encode_batch.side_effect = lambda batch: (batch["observation.state"].numpy(), None)
        encoded = list(encoded_samples(samples, prior, 2))
        self.assertEqual([sample["frame_index"] for sample, _ in encoded], list(range(5)))
        self.assertEqual([float(obs[0]) for _, obs in encoded], list(range(5)))
        self.assertEqual(prior.encode_batch.call_count, 3)

    def test_preparation_commits_complete_episode_cache_and_binds_prior(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            (dataset / "meta").mkdir(parents=True)
            manifest = {"representation": {"space": "joint", "frame": "absolute"},
                        "repo_id": "local/test", "episodes": [{"task": "pick"}]}
            (dataset / "meta/hil_serl_conversion.json").write_text(json.dumps(manifest))
            fields = ["observation.state", "action", "next.reward", "next.done",
                      "complementary_info.is_intervention", "complementary_info.command_sent"]
            (dataset / "meta/info.json").write_text(json.dumps({
                "features": dict.fromkeys(fields, {}), "total_episodes": 1, "total_frames": 3}))
            prior = Mock()
            prior.roles = ["global"]
            prior.encode.return_value = (np.zeros(30, np.float32), np.zeros(7, np.float32))
            samples = [{"episode_index": 0, "frame_index": i,
                        "observation.state": torch.zeros(7), "observation.images.global": torch.zeros(3, 8, 8),
                        "action": torch.zeros(7), "next.reward": float(i == 2), "next.done": i == 2,
                        "complementary_info.is_intervention": i == 1,
                        "complementary_info.command_sent": True} for i in range(3)]
            args = SimpleNamespace(dataset=dataset, act_checkpoint=root / "act", run_dir=root / "run",
                                   execute=True, device="cpu", gripper_max_mm=100,
                                   max_joint_delta_deg=1, max_gripper_delta_mm=2, seed=1000)
            with patch("lerobot.datasets.LeRobotDataset", return_value=samples), patch(
                "robotarm.modeling.hil_serl.prepare.ACTPrior.load", return_value=prior
            ), patch("robotarm.modeling.hil_serl.prepare.checkpoint_hash", return_value="hash"):
                prepare(args)
            cfg = Config.load(args.run_dir / "config.json")
            data, metadata = read_episode(args.run_dir / "seed/episode_000000.npz", cfg)
            self.assertEqual(cfg.task, "pick")
            self.assertEqual(cfg.act_sha256, "hash")
            np.testing.assert_array_equal(data["expert"].flatten(), [0, 1, 0])
            np.testing.assert_array_equal(data["reward"].flatten(), [0, 0, 1])
            self.assertEqual(metadata["source_episode"], 0)
            prior.close.assert_called_once()

    def test_incomplete_conversion_refused_before_loading_models(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "CONVERSION_INCOMPLETE.txt").touch()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                inspect_dataset(root)

    def test_stale_feedback_and_usb_interface_recreation_are_rejected(self):
        with patch("robotarm.modeling.hil_serl.safety.time.time", return_value=10):
            require_fresh_feedback(SimpleNamespace(timestamp=9.9), "joints", 0.5)
            with self.assertRaisesRegex(RuntimeError, "stale"):
                require_fresh_feedback(SimpleNamespace(timestamp=8), "joints", 0.5)
        with patch("robotarm.modeling.hil_serl.safety.socket.if_nametoindex", side_effect=[7, 8]):
            guard = CANIdentityGuard(["can_leader1"])
            with self.assertRaisesRegex(RuntimeError, "reconnected"):
                guard.check()

    def test_pipeline_binds_rl_to_new_hg_checkpoint_and_never_launches_actor(self):
        args = SimpleNamespace(run_dir=Path("/tmp/run"), act_dataset=Path("/tmp/bc_hg"),
                               hil_dataset=Path("/tmp/replay"), act_checkpoint=Path("/tmp/old_act"),
                               device="cpu", act_steps=20, hil_steps=30, offline_steps=10)
        plan, run = commands(args)
        self.assertEqual(len(plan), 3)
        self.assertIn("--policy.path=/tmp/old_act", plan[0])
        self.assertIn("/tmp/run/act/checkpoints/last/pretrained_model", plan[1])
        self.assertIn("learner", plan[2])
        self.assertNotIn("actor", plan[2])
        self.assertEqual(run, Path("/tmp/run/hil"))

    def test_pipeline_selects_best_evaluated_saved_act_instead_of_last(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for step in (10000, 40000):
                config = root / f"checkpoints/{step:06d}/pretrained_model/config.json"
                config.parent.mkdir(parents=True)
                config.write_text("{}")
            history = [{"_step": 10000, "eval/eval_loss": 0.0703},
                       {"_step": 11000, "eval/eval_loss": 0.069},
                       {"_step": 40000, "eval/eval_loss": 0.0727}]
            with patch("robotarm.modeling.training_report.read_run_history", return_value=(history, [])):
                selected, summary = select_act_checkpoint(root)
            self.assertEqual(selected, root / "checkpoints/010000/pretrained_model")
            self.assertEqual(summary["best_eval_step"], 11000)
            self.assertEqual(summary["recommended_checkpoint_step"], 10000)
