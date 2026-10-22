"""Safety regression tests for the base Pi0.5 shadow-only path."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from safetensors.torch import save_file

from robotarm.modeling import approve, online_policy, pi05_offline, pi05_shadow


class Pi05ShadowTests(unittest.TestCase):
    def test_offline_dataset_mapping(self):
        sample = {
            "observation.state": torch.arange(7, dtype=torch.float32),
            "observation.images.global": torch.full((3, 8, 10), 0.5),
            "observation.images.wrist": torch.full((3, 8, 10), 0.25),
            "task": "Pick up the cup",
        }
        batch = pi05_offline.make_dataset_batch(sample)
        self.assertEqual(tuple(batch["observation.state"].shape), (1, 32))
        self.assertTrue(torch.equal(batch["observation.state"][0, :7], sample["observation.state"]))
        self.assertTrue(torch.equal(batch["observation.state"][0, 7:], torch.zeros(25)))
        self.assertEqual(tuple(batch["observation.images.base_0_rgb"].shape), (1, 3, 8, 10))
        self.assertEqual(tuple(batch["observation.images.right_wrist_0_rgb"].shape), (1, 3, 8, 10))
        self.assertNotIn("observation.images.left_wrist_0_rgb", batch)
        self.assertEqual(batch["task"], ["Pick up the cup"])

    def test_offline_policy_type_must_match_checkpoint(self):
        args = pi05_offline.make_parser().parse_args(["--policy-type", "pi0"])
        self.assertEqual(args.policy_type, "pi0")

    def test_base_observation_mapping(self):
        bgr = np.array([[[10, 20, 30]]], dtype=np.uint8)
        batch = pi05_shadow.make_batch([0.0] * 6, 0.04, bgr, "Pick up the cup")
        self.assertEqual(tuple(batch["observation.state"].shape), (1, 32))
        self.assertEqual(float(batch["observation.state"][0, 6]), float(np.float32(0.04)))
        self.assertTrue(torch.equal(batch["observation.state"][0, 7:], torch.zeros(25)))
        self.assertEqual(batch["task"], ["Pick up the cup"])
        np.testing.assert_allclose(
            batch["observation.images.base_0_rgb"][0, :, 0, 0].numpy(),
            [30 / 255, 20 / 255, 10 / 255],
        )

    def test_base_schema_only(self):
        names = (
            "observation.state", "observation.images.base_0_rgb",
            "observation.images.left_wrist_0_rgb",
            "observation.images.right_wrist_0_rgb",
        )
        cfg = SimpleNamespace(
            input_features={key: SimpleNamespace(shape=(32,) if key.endswith("state") else (3, 224, 224)) for key in names},
            output_features={"action": SimpleNamespace(shape=(32,))},
        )
        pi05_shadow.validate_base_schema(cfg)
        cfg.output_features["action"].shape = (7,)
        with self.assertRaisesRegex(RuntimeError, "base schema"):
            pi05_shadow.validate_base_schema(cfg)

    def test_weight_probe_rejects_random_model(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            weights = {key: torch.ones(2) for key in (
                "action_in_proj.bias", "action_out_proj.bias", "time_mlp_in.bias"
            )}
            save_file(weights, checkpoint / "model.safetensors")
            state = {f"model.{key}": value.clone() for key, value in weights.items()}
            policy = SimpleNamespace(state_dict=lambda: state)
            pi05_shadow.verify_weights_loaded(policy, checkpoint)
            state["model.action_out_proj.bias"] = torch.zeros(2)
            with self.assertRaisesRegex(RuntimeError, "not loaded correctly"):
                pi05_shadow.verify_weights_loaded(policy, checkpoint)

    def test_execute_rejected_before_shadow_or_can_connection(self):
        from lerobot.configs.policies import PreTrainedConfig

        with (
            mock.patch.object(online_policy, "checkpoint_digest", return_value="0" * 64),
            mock.patch.object(PreTrainedConfig, "from_pretrained", return_value=SimpleNamespace(type="pi05")),
            mock.patch.object(pi05_shadow, "run") as shadow_run,
            mock.patch.object(online_policy.base, "connect_robot") as connect_robot,
        ):
            with self.assertRaisesRegex(RuntimeError, "CAN execution is locked"):
                online_policy.main([
                    "--checkpoint", "/tmp/pi05_test_checkpoint", "--device", "cpu",
                    "--task", "Pick up the cup", "--execute",
                ])
            shadow_run.assert_not_called()
            connect_robot.assert_not_called()

    def test_base_shadow_cannot_be_approved(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "shadow.jsonl"
            report.write_text(json.dumps({
                "mode": "shadow", "command_sent": False, "healthy": True,
                "checkpoint_sha256": "0" * 64, "inference_ms": 100.0,
                "camera_age_ms": 10.0, "policy_type": "pi05",
            }) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "cannot receive a CAN execution approval"):
                approve.build_approval(report, Path(directory) / "approval.json", 1, 0.05)


if __name__ == "__main__":
    unittest.main()
