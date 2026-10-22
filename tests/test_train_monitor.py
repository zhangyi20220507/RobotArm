"""scripts/train_monitor.py: must never touch the run directory it watches.

LeRobot refuses to train into an existing output directory, so a monitor that
creates --run-dir would block the run it exists to watch.
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_monitor.py"
_spec = importlib.util.spec_from_file_location("train_monitor", SCRIPT)
monitor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(monitor)


class MonitorSideEffectTests(unittest.TestCase):
    def test_main_does_not_create_the_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "not_yet"
            with mock.patch.object(monitor, "ThreadingHTTPServer"):
                self.assertEqual(
                    monitor.main(["--run-dir", str(run_dir), "--port", "0"]), 0
                )
            self.assertFalse(
                run_dir.exists(),
                "monitor created --run-dir; lerobot-train would then refuse to start",
            )

    def test_collect_tolerates_a_missing_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "not_yet"
            snapshot = monitor.collect(run_dir)
            self.assertFalse(run_dir.exists())
            self.assertIsNone(snapshot["run_file"])
            self.assertEqual(snapshot["train"], [])
            self.assertEqual(snapshot["eval"], [])
            self.assertEqual(snapshot["checkpoints"], [])
            self.assertIsNone(snapshot["best"])

    def test_link_best_refuses_a_missing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "checkpoints").mkdir()
            with self.assertRaises(FileNotFoundError):
                monitor.link_best(run_dir, 4000)
            self.assertFalse((run_dir / "checkpoints" / "best").exists())


if __name__ == "__main__":
    unittest.main()
