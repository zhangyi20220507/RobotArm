from __future__ import annotations

from typing import Sequence

from ..legacy import run_python_module
from ..workflows import WorkflowSpec

VISUALIZATION_MODULES = {
    "realsense_stream": "robotarm.visualization.realsense_stream",
    "vis_mjcf": "robotarm.visualization.vis_mjcf",
    "vis_urdf": "robotarm.visualization.vis_urdf",
}


def run_visualization(name: str, extra_args: Sequence[str] | None = None) -> int:
    if name not in VISUALIZATION_MODULES:
        raise KeyError(f"Unknown visualization workflow: {name}")
    return run_python_module(VISUALIZATION_MODULES[name], extra_args)


def run_realsense_stream(extra_args: Sequence[str] | None = None) -> int:
    return run_visualization("realsense_stream", extra_args)


def run_vis_mjcf(extra_args: Sequence[str] | None = None) -> int:
    return run_visualization("vis_mjcf", extra_args)


def run_vis_urdf(extra_args: Sequence[str] | None = None) -> int:
    return run_visualization("vis_urdf", extra_args)


def _run_module(module: str):
    def handler(extra_args: Sequence[str] | None = None) -> int:
        return run_python_module(module, extra_args)

    return handler


WORKFLOWS = {
    "viewer": WorkflowSpec("Open the Piper ManiSkill camera viewer.", _run_module("robotarm.simulation.viewer")),
    "scripted-pick-place": WorkflowSpec("Run the scripted Piper pick-and-place baseline.", _run_module("robotarm.simulation.scripted_pick_place")),
    "force-scripted-pick-place": WorkflowSpec("Run the physical-contact scripted baseline.", _run_module("robotarm.simulation.force_scripted_pick_place")),
    "force-pick-cup": WorkflowSpec("Run force-guarded cup-to-container pick-and-place.", _run_module("robotarm.simulation.force_pick_cup")),
    "collect-lerobot": WorkflowSpec("Collect qualified cup demonstrations in LeRobot format.", _run_module("robotarm.simulation.collect_lerobot")),
    "run-act": WorkflowSpec("Run an ACT checkpoint in simulation.", _run_module("robotarm.simulation.run_act")),
    "train-ppo": WorkflowSpec("Train the Piper PPO force policy.", _run_module("robotarm.simulation.train_ppo_force")),
    "evaluate-ppo": WorkflowSpec("Evaluate the Piper PPO force policy.", _run_module("robotarm.simulation.eval_ppo_force")),
    "small-vla": WorkflowSpec("Collect, train, or evaluate the small VLA experiment.", _run_module("robotarm.simulation.small_vla")),
    "open-language-vla": WorkflowSpec("Run the open-language VLA experiment.", _run_module("robotarm.simulation.open_language_vla")),
    "hierarchical-vla": WorkflowSpec("Run the hierarchical VLA experiment.", _run_module("robotarm.simulation.hierarchical_vla")),
    "setup-physx": WorkflowSpec("Install or repair the SAPIEN PhysX GPU runtime.", _run_module("robotarm.simulation.setup_physx")),
    "realsense-stream": WorkflowSpec("Stream RealSense RGB or RGB+depth images.", run_realsense_stream),
    "vis-mjcf": WorkflowSpec("Visualize an MJCF scene with MuJoCo.", run_vis_mjcf),
    "vis-urdf": WorkflowSpec("Visualize a URDF model with Placo/Meshcat.", run_vis_urdf),
}
