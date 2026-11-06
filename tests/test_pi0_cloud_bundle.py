import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts.build_pi0_cloud_bundle import build, compress_bundle


class CloudBundleTests(unittest.TestCase):
    def test_portable_independent_bundle_excludes_credentials(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "source"
            def write(name, content="example"):
                path = workspace / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
                return path
            for name in ("pyproject.toml", "docs/README.md", "docs/BestPractice_and_LessonLearn.md",
                         "scripts/train_monitor.py", "scripts/cloud/cloud_env.sh",
                         "tests/test_pi0_lora.py", "docs/best_practice_assets/example.svg",
                         "src/robotarm/modeling/pi0_lora.py"):
                write(name)
            for name in ("run_smoke.sh", "run_train.sh", "check_smoke.sh"):
                write("scripts/cloud/" + name,
                      "data/lerobot/piper_pick_cup_dual_camera_lf_180eps\npi0_lora_lf180\n"
                      + ("--batch-size 1\n" if name != "check_smoke.sh" else ""))
            write("src/robotarm/__pycache__/cached.pyc")
            write(".aws/credentials", "secret")
            write("vendor/native_binary", "not cloud training")
            data_rel = "data/lerobot/piper_pick_cup_dual_camera_lf_180eps"
            write(data_rel + "/meta/info.json", json.dumps({"total_episodes": 2, "total_frames": 10, "fps": 20}))
            write(data_rel + "/data/chunk.parquet")
            review_rel = "artifacts/reports/dataset_quality/piper_pick_cup_dual_camera_lf_180eps.json"
            original_review = json.dumps({"dataset_root": str(workspace / data_rel), "episodes": {}})
            write(review_rel, original_review)
            original_weight = write("artifacts/pretrained/pi0_base/model.safetensors", "weights")
            tokenizer = root / "cache/models--google--paligemma-3b-pt-224"
            revision = "a" * 40
            (tokenizer / "refs").mkdir(parents=True)
            (tokenizer / "refs/main").write_text(revision)
            snapshot = tokenizer / "snapshots" / revision
            snapshot.mkdir(parents=True)
            for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json", "added_tokens.json"):
                (snapshot / name).write_text('{}')
            (tokenizer.parent / "token").write_text("secret")
            bundle = root / "export/bundle"
            with patch("builtins.print"):
                archive = build(workspace, bundle, tokenizer)
            self.assertTrue(archive.is_file())
            self.assertIn("docs/README.md#85-云端包上传校验与启动",
                          (bundle / "CLOUD_README.md").read_text(encoding="utf-8"))
            self.assertTrue((bundle / "docs/best_practice_assets/example.svg").is_file())
            self.assertTrue((bundle / "docs/README.md").is_file())
            self.assertTrue((bundle / "docs/BestPractice_and_LessonLearn.md").is_file())
            self.assertEqual(json.loads((bundle / review_rel).read_text())["dataset_root"], data_rel)
            self.assertEqual((workspace / review_rel).read_text(), original_review)
            self.assertNotEqual(original_weight.stat().st_ino,
                                (bundle / "artifacts/pretrained/pi0_base/model.safetensors").stat().st_ino)
            manifest = json.loads((bundle / "CLOUD_MANIFEST.json").read_text())
            paths = [entry["path"] for entry in manifest["files"]]
            self.assertFalse(any("__pycache__" in p or p.startswith((".aws/", "vendor/")) for p in paths))
            self.assertFalse((bundle / ".cache/huggingface/token").exists())
            for entry in manifest["files"]:
                self.assertEqual(hashlib.sha256((bundle / entry["path"]).read_bytes()).hexdigest(), entry["sha256"])
            with tarfile.open(archive) as compressed:
                self.assertIn("bundle/CLOUD_MANIFEST.json", compressed.getnames())
            (bundle / ".cache/huggingface/token").write_text("runtime secret")
            (bundle / "src/robotarm/__pycache__").mkdir()
            (bundle / "src/robotarm/__pycache__/runtime.pyc").write_bytes(b"cache")
            filtered = compress_bundle(bundle, root / "export/filtered.tar.gz")
            with tarfile.open(filtered) as compressed:
                self.assertFalse(any(name.endswith("/token") or "__pycache__" in name for name in compressed.getnames()))
            with self.assertRaisesRegex(ValueError, "already exists"):
                build(workspace, bundle, tokenizer)

            trimmed_rel = "data/lerobot/piper_pick_cup_20261019_direct_122eps_trimmed_20261022"
            write(trimmed_rel + "/meta/info.json",
                  json.dumps({"total_episodes": 2, "total_frames": 8, "fps": 20}))
            write(trimmed_rel + "/meta/trim_provenance.json", json.dumps({
                "source_dataset": "/old/computer/piper_pick_cup_20261019_direct_122eps",
                "reviews_snapshot": {"0": {}, "1": {}, "2": {"status": "reject"}},
            }))
            write(trimmed_rel + "/data/chunk.parquet")
            selected_bundle = root / "export/cup122"
            with patch("builtins.print"):
                build(workspace, selected_bundle, tokenizer, trimmed_rel, "pi0_lora_cup122", 4)
            selected_manifest = json.loads((selected_bundle / "CLOUD_MANIFEST.json").read_text())
            self.assertEqual(selected_manifest["dataset"]["path"], trimmed_rel)
            self.assertEqual(selected_manifest["dataset"]["origin"]["source_episodes"], 3)
            self.assertEqual(selected_manifest["run_prefix"], "pi0_lora_cup122")
            self.assertEqual(selected_manifest["batch_size"], 4)
            selected_readme = (selected_bundle / "CLOUD_README.md").read_text()
            self.assertIn("2 条", selected_readme)
            self.assertIn("原始 3 条", selected_readme)
            self.assertIn("cup122.tar.gz.sha256", selected_readme)
            self.assertIn("batch size 均为 4", selected_readme)
            for name in ("run_smoke.sh", "run_train.sh", "check_smoke.sh"):
                content = (selected_bundle / "scripts/cloud" / name).read_text()
                self.assertIn(trimmed_rel, content)
                self.assertIn("pi0_lora_cup122", content)
                self.assertNotIn("lf_180eps", content)
                if name != "check_smoke.sh":
                    self.assertIn("--batch-size 4", content)
                    self.assertNotIn("--batch-size 1", content)
            with self.assertRaisesRegex(ValueError, "inside workspace"):
                build(workspace, root / "export/unsafe", tokenizer, "../outside")
            with self.assertRaisesRegex(ValueError, "positive integer"):
                build(workspace, root / "export/invalid", tokenizer, batch_size=0)
