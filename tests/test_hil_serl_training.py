"""CPU-only residual SAC, replay boundaries and live actor lifecycle checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from robotarm.modeling.hil_serl.config import Config
from robotarm.modeling.hil_serl.replay import Replay, records_to_episode, write_episode, read_episode
from robotarm.modeling.hil_serl.runtime import HILRuntime
from robotarm.modeling.hil_serl.sac import SAC, load_actor
from robotarm.modeling.hil_serl.smoke import smoke
from types import SimpleNamespace


class HILTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = Config("fake", "test-hash", 30, ["global", "wrist"],
                          hidden_dim=32, batch_size=8, replay_capacity=64)
        self.cfg.save(self.root / "config.json")

    def records(self):
        return [{"observation": np.full(30, i / 100, np.float32),
                 "action": np.full(7, i / 100, np.float32),
                 "expert": i == 1, "command_sent": i != 2} for i in range(4)]

    def test_terminal_reward_and_executed_actions_with_gaps(self):
        records = self.records()
        result = records_to_episode(records, "success", self.cfg)
        self.assertEqual(result["state"].shape, (3, 30))
        np.testing.assert_allclose(result["next_state"][1], records[2]["observation"])
        np.testing.assert_array_equal(result["reward"].flatten(), [0, 0, 1])
        np.testing.assert_array_equal(result["done"].flatten(), [0, 0, 1])
        failed = records_to_episode(records, "failure", self.cfg)
        self.assertEqual(failed["reward"].sum(), 0)

    def test_human_pool_balances_and_survives_online_eviction(self):
        data = records_to_episode(self.records(), "success", self.cfg)
        write_episode(self.root / "seed/episode_0.npz", data, self.cfg, metadata={})
        replay = Replay(self.cfg)
        replay.ingest(self.root / "seed")
        self.assertGreaterEqual(replay.sample()["expert"].sum(), self.cfg.batch_size // 2)
        self.assertEqual(replay.ingest(self.root / "seed"), 0)

    def test_sac_update_resume_and_exact_next_rng(self):
        algorithm = SAC(self.cfg)
        data = records_to_episode(self.records(), "success", self.cfg)
        stats = algorithm.update(data)
        self.assertTrue(all(np.isfinite(v) for v in stats.values()))
        algorithm.save(self.root / "training.pt", training=True)
        expected_random = torch.rand(3)
        restored = SAC(self.cfg)
        restored.restore(self.root / "training.pt")
        torch.testing.assert_close(torch.rand(3), expected_random)
        self.assertEqual(restored.step, 1)
        self.assertEqual(algorithm.actor_opt.state_dict()["state"].keys(),
                         restored.actor_opt.state_dict()["state"].keys())

    def test_actor_lifecycle_sync_only_between_trials_and_label_before_commit(self):
        algorithm = SAC(self.cfg)
        algorithm.step = 100
        algorithm.save(self.root / "policy.pt")
        prior = Mock()
        prior.roles = self.cfg.camera_roles
        observation = np.zeros(30, np.float32)
        prior.encode.return_value = (observation, np.zeros(7, np.float32))
        with patch("robotarm.modeling.hil_serl.runtime.ACTPrior", return_value=prior), patch(
            "robotarm.modeling.hil_serl.runtime.checkpoint_hash", return_value="test-hash"
        ):
            actor = HILRuntime(self.root, "fake", Mock(), Mock(), Mock())
        actor.start_trial()
        proposed = actor.predict({})
        np.testing.assert_array_less(np.abs(proposed), np.array(self.cfg.residual_scale) * self.cfg.action_scale + 1e-6)
        actual = np.ones(7, np.float32) / 100
        actor.capture(actual, expert=True, command_sent=True)
        # A slow feature frame is rejected without ending the human trial or
        # committing an unexecuted action. Fresh features can resume capture.
        actor.predict({}, autonomous=False)
        actor.discard_observation()
        self.assertEqual(len(actor.records), 1)
        self.assertIsNone(actor.base_action)
        with self.assertRaisesRegex(RuntimeError, "missing observation"):
            actor.capture(actual, expert=True, command_sent=True)
        fresh = observation.copy()
        fresh[0] = 0.25
        prior.encode.return_value = (fresh, np.zeros(7, np.float32))
        actor.predict({}, autonomous=False)
        actor.capture(actual, expert=True, command_sent=True)
        self.assertEqual(len(actor.records), 2)
        self.assertEqual(actor.records[-1]["observation"][0], 0.25)
        with self.assertRaisesRegex(RuntimeError, "between trials"):
            actor.reload()
        episode = self.root / "raw/episode_0"
        actor.finish_trial(episode)
        self.assertFalse((self.root / "online").exists())
        with self.assertRaisesRegex(RuntimeError, "label"):
            actor.start_trial()
        actor.label(episode, "failure")
        files = list((self.root / "online").glob("*.npz"))
        self.assertEqual(len(files), 1)
        data, meta = read_episode(files[0], self.cfg)
        np.testing.assert_allclose(data["action"][0], actual / self.cfg.action_scale)
        self.assertEqual(data["reward"].sum(), 0)
        self.assertEqual(meta["policy_version"], 100)
        actor.start_trial()
        actor.predict({})
        actor.capture(actual, expert=True, command_sent=True)
        actor.discard()
        self.assertEqual(len(list((self.root / "online").glob("*.npz"))), 1)
        actor.close()

    def test_wrong_prior_and_untrained_snapshot_are_rejected(self):
        algorithm = SAC(self.cfg)
        algorithm.save(self.root / "policy.pt")
        with self.assertRaisesRegex(ValueError, "not completed"):
            load_actor(self.root / "policy.pt", self.cfg)
        algorithm.step = 10
        algorithm.save(self.root / "policy.pt")
        self.cfg.act_sha256 = "other"
        with self.assertRaisesRegex(ValueError, "incompatible"):
            load_actor(self.root / "policy.pt", self.cfg)

    def test_trial_start_reuses_unchanged_snapshot_and_loads_new_version(self):
        algorithm = SAC(self.cfg)
        algorithm.step = 100
        algorithm.save(self.root / "policy.pt")
        prior = Mock()
        prior.roles = self.cfg.camera_roles
        with patch("robotarm.modeling.hil_serl.runtime.ACTPrior", return_value=prior), patch(
            "robotarm.modeling.hil_serl.runtime.checkpoint_hash", return_value="test-hash"
        ), patch("robotarm.modeling.hil_serl.runtime.load_actor", wraps=load_actor) as load:
            actor = HILRuntime(self.root, "fake", Mock(), Mock(), Mock())
            actor.start_trial()
            self.assertEqual(load.call_count, 1)
            actor.discard()
            algorithm.step = 101
            algorithm.save(self.root / "policy.pt")
            actor.start_trial()
            self.assertEqual(load.call_count, 2)
            self.assertEqual(actor.version, 101)
            actor.close()

    def test_gripper_actor_stays_in_physical_width_range(self):
        actor = SAC(self.cfg).actor
        observation = torch.zeros(16, self.cfg.observation_dim)
        observation[:, -1] = 0.0
        for _ in range(5):
            action, _ = actor(observation)
            self.assertTrue((action[:, -1] >= 0).all())
            self.assertTrue((action[:, -1] <= 1).all())

    def test_emergency_stop_cleanup_does_not_restore_zero_force_arm(self):
        from robotarm.modeling.hg_dagger import LeaderExpert
        leader = LeaderExpert.__new__(LeaderExpert)
        leader.robot = Mock()
        leader.drag_active = True
        leader.joints = [0.0] * 6
        leader.robot.electronic_emergency_stop.side_effect = OSError("USB disconnected")
        with self.assertRaises(OSError):
            leader.emergency_stop()
        self.assertFalse(leader.drag_active)
        with patch("robotarm.modeling.hg_dagger.exit_leader_drag_mode") as restore:
            leader.close()
        restore.assert_not_called()

    def test_end_to_end_cpu_learner_and_resume(self):
        smoke(SimpleNamespace(output=self.root / "smoke", steps=3))
        result = json.loads((self.root / "smoke/smoke_result.json").read_text())
        self.assertTrue(result["passed"])
        self.assertFalse(result["hardware_connected"])
        self.assertEqual(result["updates"], 4)
