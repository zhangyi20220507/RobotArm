"""Dataset relocation tests without CAN, cameras, GPU or Hub access."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from robotarm.modeling.online_policy import load_dataset_metadata


class DatasetMetadataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkpoint = self.root / "checkpoint"
        self.checkpoint.mkdir()
        self.local = self.root / "local"
        self.recorded = self.root / "cloud" / "piper_pick_cup_test"
        self.configure(self.recorded)
        self.loader = Mock(return_value=object())
        self.enterContext(patch.dict("sys.modules", {
            "lerobot.datasets.lerobot_dataset": SimpleNamespace(
                LeRobotDatasetMetadata=self.loader),
        }))
        self.enterContext(patch("robotarm.paths.LEROBOT_DATA_ROOT", self.local))

    def configure(self, recorded, repo_id="local/piper_pick_cup_test"):
        (self.checkpoint / "train_config.json").write_text(json.dumps({
            "dataset": {"root": str(recorded), "repo_id": repo_id},
        }))

    def metadata(self, root):
        (root / "meta").mkdir(parents=True)
        (root / "meta" / "info.json").write_text("{}")

    def deny_cloud_probe(self):
        original = Path.is_file

        def is_file(path):
            if path == self.recorded / "meta" / "info.json":
                raise PermissionError(13, "Permission denied", str(path))
            return original(path)

        return patch.object(Path, "is_file", is_file)

    def test_permission_denied_cloud_root_falls_back_without_changing_checkpoint(self):
        local = self.local / self.recorded.name
        self.metadata(local)
        before = (self.checkpoint / "train_config.json").read_bytes()
        with self.deny_cloud_probe():
            self.assertIs(load_dataset_metadata(self.checkpoint), self.loader.return_value)
        self.loader.assert_called_once_with(repo_id="local/piper_pick_cup_test", root=local)
        self.assertEqual(before, (self.checkpoint / "train_config.json").read_bytes())

    def test_permission_denied_during_resolve_also_falls_back(self):
        local = self.local / self.recorded.name
        self.metadata(local)
        original = Path.resolve

        def resolve(path, *args, **kwargs):
            if path == self.recorded:
                raise PermissionError(13, "Permission denied", str(path))
            return original(path, *args, **kwargs)

        with patch.object(Path, "resolve", resolve), self.deny_cloud_probe():
            load_dataset_metadata(self.checkpoint)
        self.loader.assert_called_once_with(repo_id="local/piper_pick_cup_test", root=local)

    def test_accessible_recorded_root_keeps_priority(self):
        self.metadata(self.recorded)
        self.metadata(self.local / self.recorded.name)
        load_dataset_metadata(self.checkpoint)
        self.loader.assert_called_once_with(
            repo_id="local/piper_pick_cup_test", root=self.recorded)

    def test_missing_cloud_root_keeps_legacy_rename_fallback(self):
        old = self.root / "cloud" / "piper_pick_up_test"
        self.configure(old, "local/piper_pick_up_test")
        local = self.local / "piper_pick_cup_test"
        self.metadata(local)
        load_dataset_metadata(self.checkpoint)
        self.loader.assert_called_once_with(repo_id="local/piper_pick_cup_test", root=local)

    def test_no_local_metadata_reports_tried_and_denied_paths(self):
        with self.deny_cloud_probe(), self.assertRaises(RuntimeError) as caught:
            load_dataset_metadata(self.checkpoint)
        self.assertIn(str(self.local / self.recorded.name), str(caught.exception))
        self.assertIn("permission denied", str(caught.exception))
        self.loader.assert_not_called()

    def test_corrupt_metadata_is_not_silently_replaced(self):
        self.metadata(self.recorded)
        self.metadata(self.local / self.recorded.name)
        self.loader.side_effect = ValueError("invalid metadata")
        with self.assertRaisesRegex(ValueError, "invalid metadata"):
            load_dataset_metadata(self.checkpoint)
        self.loader.assert_called_once()


if __name__ == "__main__":
    unittest.main()
