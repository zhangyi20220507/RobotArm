from pathlib import Path
import tarfile
import tempfile
import unittest

from scripts.build_vla_cloud_bundles import build, build_directories, render_job_env


class VLACloudBundleTests(unittest.TestCase):
    def test_two_code_only_bundles_share_environment_and_differ_by_job(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "workspace"
            for relative in (
                "pyproject.toml",
                "requirements/unified-constraints.txt",
                "scripts/train_monitor.py",
                "docs/CLOUD_VLA_TRAINING.md",
                "src/robotarm/__init__.py",
                "scripts/cloud_vla/setup_env.sh",
            ):
                path = workspace / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative, encoding="utf-8")
            (workspace / "src/robotarm/__pycache__").mkdir()
            (workspace / "src/robotarm/__pycache__/ignored.pyc").write_bytes(b"cache")
            results = build(workspace, root / "out", "training")
            self.assertEqual(len(results), 2)
            for archive, checksum in results:
                self.assertTrue(archive.is_file())
                self.assertTrue(checksum.is_file())
                with tarfile.open(archive) as compressed:
                    names = compressed.getnames()
                    self.assertTrue(any(name.endswith("/scripts/cloud_vla/setup_env.sh") for name in names))
                    self.assertFalse(any("__pycache__" in name for name in names))
                    self.assertFalse(any("data/lerobot/task_dataset/" in name for name in names))
            self.assertIn("POLICY=pi0", render_job_env("pi0"))
            self.assertIn("TRAINING_MODE=lora", render_job_env("pi0"))
            self.assertIn("BATCH_SIZE=1", render_job_env("pi0"))
            self.assertIn("POLICY=smolvla", render_job_env("smolvla"))
            self.assertIn("TRAINING_MODE=expert", render_job_env("smolvla"))
            self.assertIn("BATCH_SIZE=4", render_job_env("smolvla"))

    def test_directory_mode_keeps_editable_packages_uncompressed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "workspace"
            for relative in (
                "pyproject.toml",
                "requirements/unified-constraints.txt",
                "scripts/train_monitor.py",
                "docs/CLOUD_VLA_TRAINING.md",
                "src/robotarm/__init__.py",
                "scripts/cloud_vla/service.sh",
            ):
                path = workspace / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative, encoding="utf-8")
            packages = build_directories(workspace, root / "out", "training")
            self.assertEqual([package.name for package in packages], ["training_pi0", "training_smolvla"])
            for package in packages:
                self.assertTrue((package / "README.md").is_file())
                self.assertTrue((package / "job.env").is_file())
                self.assertFalse(package.with_suffix(".tar.gz").exists())
            with self.assertRaises(FileExistsError):
                build_directories(workspace, root / "out", "training")


if __name__ == "__main__":
    unittest.main()
