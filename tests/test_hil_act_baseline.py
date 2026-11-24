"""ACT parity, explicit exploration and cache-version boundaries; no hardware."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from robotarm.modeling.hil_serl.cli import main
from robotarm.modeling.hil_serl.config import Config
from robotarm.modeling.hil_serl.prepare import encoded_samples
from robotarm.modeling.hil_serl.prior import ACTPrior


class BaselineCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.cfg = Config("/tmp/fake-act", "fake", 26, ["global"])
        self.cfg.save(self.run / "config.json")

    def test_default_baseline_uses_original_act_without_loading_sac(self):
        # There is deliberately no policy.pt or seed cache in this directory.
        with patch("robotarm.modeling.online_policy.main", return_value=0) as online:
            main(["actor", "--run-dir", str(self.run), "--execute"])
        args = online.call_args.args[0]
        self.assertNotIn("--hil-run", args)
        self.assertEqual(args[args.index("--temporal-ensemble-coeff") + 1], "0.01")
        self.assertEqual(args[args.index("--n-action-steps") + 1], "1")
        self.assertIn("--hg-dagger", args)

    def test_old_residual_weights_cannot_be_silently_reused(self):
        with patch("robotarm.modeling.online_policy.main") as online:
            with self.assertRaisesRegex(ValueError, "different ACT reference"):
                main(["actor", "--run-dir", str(self.run), "--mode", "residual"])
        online.assert_not_called()
        with self.assertRaisesRegex(ValueError, "new v2 run"):
            main(["learner", "--run-dir", str(self.run), "--resume"])

    def test_residual_is_deterministic_and_exploration_is_explicit(self):
        self.cfg.format_version = "piper_hil_residual_sac_v2"
        self.cfg.temporal_ensemble_coeff = 0.01
        self.cfg.save(self.run / "config.json")
        for mode in ("residual", "explore"):
            with patch("robotarm.modeling.online_policy.main", return_value=0) as online:
                main(["actor", "--run-dir", str(self.run), "--mode", mode])
            args = online.call_args.args[0]
            self.assertIn("--hil-run", args)
            self.assertEqual("--hil-deterministic" in args, mode == "residual")
            self.assertEqual("--hil-explore" in args, mode == "explore")

    def test_legacy_signature_stays_identical_and_v2_is_distinct(self):
        original = json.loads((self.run / "config.json").read_text())
        original.pop("temporal_ensemble_coeff")
        digest = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
        self.assertEqual(self.cfg.signature, digest)
        self.cfg.format_version = "piper_hil_residual_sac_v2"
        self.cfg.temporal_ensemble_coeff = 0.01
        self.assertNotEqual(self.cfg.signature, digest)

    def test_temporal_cache_is_sequential_and_resets_on_episode_and_takeover(self):
        prior = Mock()
        prior.temporal_ensemble_coeff = 0.01
        prior.roles = ["global"]
        prior.encode.return_value = (np.zeros(26), np.zeros(7))
        samples = [{"episode_index": ep, "complementary_info.control_source": source,
                    "observation.state": torch.zeros(7),
                    "observation.images.global": torch.zeros(3, 2, 2)}
                   for ep, source in ((0, 0), (0, 1), (0, 2), (0, 0), (1, 0))]
        result = list(encoded_samples(samples, prior, 8))
        self.assertEqual(len(result), 5)
        self.assertEqual(prior.reset.call_count, 4)
        self.assertEqual(prior.encode.call_count, 5)
        prior.encode_batch.assert_not_called()


class _Backbone(torch.nn.Module):
    def forward(self, image):
        return {"feature_map": image}


class _Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
        self.config = SimpleNamespace(n_action_steps=1, temporal_ensemble_coeff=0.01,
                                      image_features={"observation.images.global": None})
        self.model = SimpleNamespace(backbone=_Backbone())
        self.temporal_ensembler = ACTTemporalEnsembler(0.01, 3)

    def predict_action_chunk(self, batch):
        self.model.backbone(batch["observation.images.global"])
        return batch["observation.state"][:, None] + torch.arange(3)[None, :, None] * 0.01

    def select_action(self, batch):
        return self.temporal_ensembler.update(self.predict_action_chunk(batch))

    def reset(self):
        self.temporal_ensembler.reset()


class ACTParityTests(unittest.TestCase):
    def test_reference_matches_original_ensemble_across_hold_reject_and_reset(self):
        original = _Policy()
        adapted = _Policy()
        scale = torch.tensor([np.pi] * 6 + [0.1])
        prior = ACTPrior(adapted, lambda b: b, lambda x: x * scale,
                         temporal_ensemble_coeff=0.01)
        self.addCleanup(prior.close)

        def batch(value):
            return {"observation.state": torch.full((1, 7), value),
                    "observation.images.global": torch.zeros(1, 3, 2, 2)}

        def compare(value):
            _, actual = prior.encode(batch(value))
            expected = (original.select_action(batch(value)) * scale)[0].numpy()
            np.testing.assert_array_equal(actual, expected)

        compare(0.01)
        compare(0.02)
        hold = np.ones(7, np.float32) * 0.015
        _, held = prior.encode(batch(0.03), reference_action=hold)
        np.testing.assert_array_equal(held, hold)
        compare(0.04)  # A safety hold must not advance the temporal clock.
        prior.encode(batch(0.05))
        prior.discard_observation()
        compare(0.06)  # A rejected late frame must not remain in the ensemble.
        prior.reset()
        original.reset()
        compare(0.07)
