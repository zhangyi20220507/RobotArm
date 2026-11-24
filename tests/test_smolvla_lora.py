"""SmolVLA cloud launch contracts without model allocation or downloads."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from robotarm.modeling.smolvla_lora import build_plan, main, parse_args
from robotarm.modeling.smolvla_lora_driver import verify_base_weights


class SmolVLALoraTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = self.root / "dataset"
        (self.dataset / "meta").mkdir(parents=True)
        features = {
            "observation.state": {"shape": [7]},
            "action": {"shape": [7]},
            "observation.images.global": {"shape": [480, 640, 3]},
            "observation.images.wrist": {"shape": [480, 640, 3]},
        }
        (self.dataset / "meta/info.json").write_text(
            json.dumps({"total_episodes": 10, "total_frames": 100, "fps": 20, "features": features})
        )
        (self.dataset / "meta/piper_conversion.json").write_text(
            json.dumps({"representation": {"space": "joint", "frame": "absolute"}})
        )
        import pandas as pd
        pd.DataFrame({"task_index": [0]}, index=pd.Index(["Pick up the component."], name="task")).to_parquet(
            self.dataset / "meta/tasks.parquet"
        )
        self.base = self.root / "base"
        self.base.mkdir()
        (self.base / "config.json").write_text('{"type":"smolvla"}')
        for name in ("model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
            (self.base / name).touch()
        self.argv = [
            "--dataset-root",
            str(self.dataset),
            "--base-checkpoint",
            str(self.base),
            "--output-dir",
            str(self.root / "run"),
        ]

    def test_default_plan_loads_pretrained_base_and_trains_expert(self):
        plan = build_plan(parse_args(self.argv))
        flags = plan["flags"]
        self.assertEqual(flags["policy.path"], str(self.base))
        features = json.loads(flags["policy.input_features"])
        self.assertEqual(features["observation.state"]["shape"], [7])
        self.assertEqual(features["observation.images.global"]["shape"], [3, 480, 640])
        self.assertEqual(features["observation.images.wrist"]["shape"], [3, 480, 640])
        self.assertNotIn("rename_map", flags)
        self.assertEqual(flags["policy.output_features"], "null")
        self.assertEqual(flags["policy.load_vlm_weights"], "true")
        self.assertEqual(flags["policy.use_peft"], "false")
        self.assertNotIn("peft.method_type", flags)
        self.assertEqual(plan["training_mode"], "expert")
        self.assertEqual(flags["batch_size"], 4)
        self.assertEqual(len(set(plan["selected_episodes"])), 10)

    def test_lora_is_an_explicit_lower_memory_mode(self):
        plan = build_plan(parse_args(self.argv + ["--training-mode", "lora"]))
        self.assertEqual(plan["training_mode"], "lora")
        self.assertEqual(plan["flags"]["peft.method_type"], "LORA")
        self.assertEqual(plan["flags"]["peft.r"], 16)

    def test_smoke_saves_and_evaluates(self):
        flags = build_plan(parse_args(self.argv + ["--smoke"]))["flags"]
        self.assertEqual((flags["steps"], flags["save_freq"], flags["eval_steps"]), (20, 20, 20))
        self.assertEqual(flags["num_workers"], 0)
        self.assertEqual(flags["policy.scheduler_warmup_steps"], 2)
        self.assertEqual(flags["policy.scheduler_decay_steps"], 20)

    def test_scheduler_tracks_actual_training_length(self):
        flags = build_plan(parse_args(self.argv + ["--steps", "5000", "--warmup-steps", "100"]))["flags"]
        self.assertEqual(flags["policy.scheduler_decay_steps"], 5000)
        self.assertEqual(flags["policy.scheduler_warmup_steps"], 100)
        self.assertEqual(flags["policy.scheduler_decay_lr"], 2.5e-6)

    def test_language_task_index_and_missing_tasks_rejected(self):
        import pandas as pd
        path = self.dataset / "meta/tasks.parquet"
        pd.DataFrame({"task": ["Pick"], "task_index": [0]}).to_parquet(path)
        with self.assertRaisesRegex(ValueError, "pandas index"):
            build_plan(parse_args(self.argv))
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing language"):
            build_plan(parse_args(self.argv))

    def test_empty_task_rejected(self):
        import pandas as pd
        pd.DataFrame({"task_index": [0]}, index=pd.Index([" "], name="task")).to_parquet(
            self.dataset / "meta/tasks.parquet"
        )
        with self.assertRaisesRegex(ValueError, "non-empty"):
            build_plan(parse_args(self.argv))

    def test_lerobot_parses_plan_and_uses_piper_features(self):
        from lerobot.configs import parser
        from lerobot.configs.train import TrainPipelineConfig
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

        SmolVLAConfig().save_pretrained(self.base)
        for extra in ([], ["--training-mode", "lora"]):
            plan = build_plan(parse_args(self.argv + extra))

            @parser.wrap()
            def inspect(cfg: TrainPipelineConfig):
                cfg.validate()
                return cfg

            with patch("sys.argv", ["train"] + plan["command"][3:]):
                cfg = inspect()
            self.assertEqual(cfg.policy.input_features["observation.state"].shape, (7,))
            self.assertIn("observation.images.wrist", cfg.policy.input_features)
            self.assertEqual(cfg.scheduler.num_decay_steps, 30000)
            self.assertEqual(cfg.policy.train_expert_only, True)
            self.assertEqual(cfg.peft is not None, bool(extra))

    def test_dry_run_does_not_check_gpu_or_launch(self):
        with patch("robotarm.modeling.smolvla_lora.check_dependencies", side_effect=AssertionError), patch(
            "robotarm.modeling.smolvla_lora.subprocess.call", side_effect=AssertionError
        ), patch("builtins.print"):
            self.assertEqual(main(self.argv), 0)

    def test_projection_guard(self):
        import torch
        from safetensors.torch import save_file

        tensors = {
            name: torch.ones(2, 2)
            for name in ("model.state_proj.weight", "model.action_in_proj.weight", "model.action_out_proj.weight")
        }
        save_file(tensors, str(self.base / "model.safetensors"))

        class Model:
            def state_dict(inner):
                return tensors

        verify_base_weights(Model(), self.base)
        save_file({"model.state_proj.weight": tensors["model.state_proj.weight"]}, str(self.base / "partial.safetensors"))
        (self.base / "partial.safetensors").replace(self.base / "model.safetensors")
        with self.assertRaisesRegex(RuntimeError, "Missing SmolVLA base projection"):
            verify_base_weights(Model(), self.base)
        save_file(tensors, str(self.base / "model.safetensors"))
        tensors["model.state_proj.weight"] = torch.zeros(2, 2)
        with self.assertRaisesRegex(RuntimeError, "not loaded correctly"):
            verify_base_weights(Model(), self.base)


if __name__ == "__main__":
    unittest.main()
