from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .base import ModelAdapter


class ACTAdapter(ModelAdapter):
    name = "act"
    label = "LeRobot ACT"

    def describe(self, checkpoint: Path | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "label": self.label,
            "parameters": {
                "inference_mode": {
                    "type": "enum",
                    "values": ["replan_each_frame", "chunked"],
                    "default": "replan_each_frame",
                },
                "output_step": {"type": "integer", "min": 0, "default": 0},
            },
        }
        if checkpoint is not None:
            config_path = checkpoint / "config.json"
            if config_path.is_file():
                config = json.loads(config_path.read_text(encoding="utf-8"))
                size = int(config.get("chunk_size", 1))
                result["parameters"]["output_step"]["max"] = size - 1
                result["chunk_size"] = size
        return result

    def evaluate(self, request: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        checkpoint = Path(request["checkpoint"]).expanduser().resolve()
        dataset = Path(request["dataset_root"]).expanduser().resolve()
        episode = int(request["episode"])
        output_step = int(request.get("output_step", 0))
        inference_mode = str(request.get("inference_mode", "replan_each_frame"))
        device = str(request.get("device", "cuda"))
        description = self.describe(checkpoint)
        maximum = description["parameters"]["output_step"].get("max", 0)
        if not 0 <= output_step <= maximum:
            raise ValueError(f"output_step must be in 0..{maximum}")
        if inference_mode not in {"replan_each_frame", "chunked"}:
            raise ValueError(f"unsupported inference mode: {inference_mode}")
        if inference_mode == "chunked" and output_step:
            raise ValueError("chunked mode requires output_step=0")
        command = [
            sys.executable, "-m", "robotarm.modeling.trajectory_studio.worker",
            "--checkpoint", str(checkpoint),
            "--dataset-root", str(dataset),
            "--episode", str(episode),
            "--device", device,
            "--output-step", str(output_step),
            "--output-dir", str(output_dir),
        ]
        if inference_mode == "chunked":
            command.append("--chunked")
        completed = subprocess.run(
            command,
            text=True, capture_output=True, timeout=1800,
        )
        (output_dir / "inference.log").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        if completed.returncode:
            raise RuntimeError(completed.stdout.strip() or completed.stderr.strip())
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        summary["adapter"] = self.name
        summary["capabilities"] = description
        return summary


class LeRobotPolicyAdapter(ACTAdapter):
    """Open-loop adapter for a specific LeRobot policy type.

    The studio worker loads policies through LeRobot's common factory, so
    Diffusion Policy and SmolVLA share the same offline replay contract as ACT.
    Keeping separate adapters makes the model choice explicit in the UI and
    prevents accidentally evaluating a checkpoint under the wrong label.
    """

    def __init__(self, name: str, label: str, policy_type: str):
        self.name = name
        self.label = label
        self.policy_type = policy_type

    def describe(self, checkpoint: Path | None = None) -> dict[str, Any]:
        result = super().describe(checkpoint)
        result["policy_type"] = self.policy_type
        if checkpoint is not None:
            config_path = checkpoint / "config.json"
            if config_path.is_file():
                actual = json.loads(config_path.read_text(encoding="utf-8")).get("type")
                result["checkpoint_policy_type"] = actual
                result["checkpoint_matches_adapter"] = actual == self.policy_type
        return result

    def evaluate(self, request: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        checkpoint = Path(request["checkpoint"]).expanduser().resolve()
        config_path = checkpoint / "config.json"
        if not config_path.is_file():
            raise ValueError(f"checkpoint is missing {config_path.name}")
        actual = json.loads(config_path.read_text(encoding="utf-8")).get("type")
        if actual != self.policy_type:
            raise ValueError(
                f"adapter {self.name!r} requires a {self.policy_type!r} "
                f"checkpoint, got {actual!r}"
            )
        return super().evaluate(request, output_dir)
