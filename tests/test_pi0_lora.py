"""Cloud launch contracts: no model allocation, downloads or hardware."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from robotarm.modeling.pi0_lora import build_plan, main, parse_args
from robotarm.modeling.pi0_lora_driver import load_verified_base, verify_base_weights


class Pi0LoraTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = self.root / "dataset"
        (self.dataset / "meta").mkdir(parents=True)
        features = {"observation.state": {"shape": [7]}, "action": {"shape": [7]},
                    "observation.images.global": {"shape": [480, 640, 3]},
                    "observation.images.wrist": {"shape": [480, 640, 3]}}
        (self.dataset / "meta/info.json").write_text(json.dumps(
            {"total_episodes": 10, "total_frames": 100, "fps": 20, "features": features}))
        (self.dataset / "meta/piper_conversion.json").write_text(json.dumps(
            {"representation": {"space": "joint", "frame": "absolute"}}))
        self.base = self.root / "base"
        self.base.mkdir()
        (self.base / "config.json").write_text('{"type":"pi0"}')
        for name in ["model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"]:
            (self.base / name).touch()
        self.argv = ["--dataset-root", str(self.dataset), "--base-checkpoint", str(self.base),
                     "--output-dir", str(self.root / "run")]

    def test_plan_uses_base_then_lora_and_native_seven_dim_features(self):
        plan = build_plan(parse_args(self.argv))
        f = plan["flags"]
        self.assertEqual(f["policy.path"], str(self.base))
        self.assertEqual(f["policy.input_features"], "null")
        self.assertEqual(f["policy.output_features"], "null")
        self.assertEqual(f["policy.use_peft"], "false")
        self.assertEqual(f["peft.method_type"], "LORA")
        self.assertEqual(f["policy.empty_cameras"], 1)
        self.assertEqual(f["policy.use_relative_actions"], "false")
        self.assertEqual(f["batch_size"], 1)
        self.assertEqual(len(set(plan["selected_episodes"])), 10)
        self.assertEqual(plan["selected_episodes"], build_plan(parse_args(self.argv))["selected_episodes"])

    def test_smoke_still_saves_and_evaluates(self):
        flags = build_plan(parse_args(self.argv + ["--smoke"]))["flags"]
        self.assertEqual((flags["steps"], flags["save_freq"], flags["eval_steps"]), (20, 20, 20))
        self.assertEqual(flags["num_workers"], 0)

    def test_dry_run_never_checks_gpu_or_launches(self):
        with patch("robotarm.modeling.pi0_lora.check_dependencies", side_effect=AssertionError), \
             patch("robotarm.modeling.pi0_lora.subprocess.call", side_effect=AssertionError), \
             patch("builtins.print"):
            self.assertEqual(main(self.argv), 0)
        self.assertFalse((self.root / "run").exists())

    def test_existing_run_and_wrong_semantics_rejected(self):
        (self.root / "run").mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            build_plan(parse_args(self.argv))
        (self.dataset / "meta/piper_conversion.json").write_text(json.dumps(
            {"representation": {"space": "joint", "frame": "relative"}}))
        with self.assertRaisesRegex(ValueError, "absolute"):
            build_plan(parse_args(self.argv[:-1] + [str(self.root / "other")]))

    def test_missing_execute_dependency_cannot_start_training(self):
        with patch("robotarm.modeling.pi0_lora.check_dependencies", side_effect=RuntimeError("missing peft")), \
             patch("robotarm.modeling.pi0_lora.subprocess.call") as launch, patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, "missing peft"):
                main(self.argv + ["--execute"])
            launch.assert_not_called()

    def test_silent_pretrained_failure_becomes_error(self):
        def failed_loader():
            print("Warning: Could not load state dict: missing weights")
            return object()
        from contextlib import redirect_stdout
        import io
        with redirect_stdout(io.StringIO()), patch("robotarm.modeling.pi0_lora_driver.verify_base_weights") as verify:
            with self.assertRaisesRegex(RuntimeError, "reported a failure"):
                load_verified_base(failed_loader, self.base)
            verify.assert_not_called()

    def test_projection_guard_detects_random_or_different_weights(self):
        import torch
        from safetensors.torch import save_file
        tensors = {f"{name}.weight": torch.ones(2, 2) for name in
                   ("state_proj", "action_in_proj", "action_out_proj")}
        save_file(tensors, str(self.base / "model.safetensors"))
        class Model:
            def state_dict(inner):
                return {"model." + key: value for key, value in tensors.items()}
        verify_base_weights(Model(), self.base)
        class Wrong(Model):
            def state_dict(inner):
                return {"model." + key: value * 0 for key, value in tensors.items()}
        with self.assertRaisesRegex(RuntimeError, "not loaded correctly"):
            verify_base_weights(Wrong(), self.base)
