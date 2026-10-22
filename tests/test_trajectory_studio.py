from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robotarm import modeling
from robotarm.modeling.trajectory_studio.adapters.act import ACTAdapter


class OpenLoopAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.checkpoint = root / "checkpoint"
        self.dataset = root / "dataset"
        self.output = root / "output"
        self.checkpoint.mkdir()
        self.dataset.mkdir()
        self.output.mkdir()
        (self.checkpoint / "config.json").write_text(
            json.dumps({"type": "act", "chunk_size": 100}), encoding="utf-8"
        )

    def request(self, **overrides) -> dict:
        request = {
            "checkpoint": str(self.checkpoint),
            "dataset_root": str(self.dataset),
            "episode": 71,
            "device": "cpu",
            "output_step": 0,
            "inference_mode": "replan_each_frame",
        }
        request.update(overrides)
        return request

    def test_describes_both_inference_modes(self) -> None:
        description = ACTAdapter().describe(self.checkpoint)

        self.assertEqual(description["parameters"]["output_step"]["max"], 99)
        self.assertEqual(
            description["parameters"]["inference_mode"]["values"],
            ["replan_each_frame", "chunked"],
        )

    def test_chunked_mode_is_forwarded_to_internal_worker(self) -> None:
        (self.output / "summary.json").write_text(
            json.dumps({"mae": 0.25}), encoding="utf-8"
        )
        completed = subprocess.CompletedProcess([], 0, stdout="ok", stderr="")

        with patch(
            "robotarm.modeling.trajectory_studio.adapters.act.subprocess.run",
            return_value=completed,
        ) as run:
            summary = ACTAdapter().evaluate(
                self.request(inference_mode="chunked"), self.output
            )

        self.assertIn("--chunked", run.call_args.args[0])
        self.assertEqual(summary["mae"], 0.25)

    def test_chunked_mode_rejects_nonzero_output_step(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires output_step=0"):
            ACTAdapter().evaluate(
                self.request(inference_mode="chunked", output_step=1), self.output
            )

    def test_internal_worker_is_not_a_public_workflow(self) -> None:
        self.assertNotIn("replay-compare", modeling.WORKFLOWS)
        self.assertIn("trajectory-studio", modeling.WORKFLOWS)


if __name__ == "__main__":
    unittest.main()
