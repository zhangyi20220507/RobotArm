from __future__ import annotations

from typing import Sequence

from .legacy import run_python_module
from .workflows import WorkflowSpec


def read_piper_status(extra_args: Sequence[str] | None = None) -> int:
    """Continuously print Piper joint feedback."""
    return run_python_module("robotarm.hardware.read_status", extra_args)


def enable_piper_control(extra_args: Sequence[str] | None = None) -> int:
    """Run the Piper enable/disable/reset control workflow."""
    return run_python_module("robotarm.hardware.enable_control", extra_args)


def collect_piper_episode(extra_args: Sequence[str] | None = None) -> int:
    """Run the Piper gamepad data-collection runtime."""
    return run_python_module("robotarm.teleop.gamepad.runtime", extra_args)


def collect_leader_follower_episode(
    extra_args: Sequence[str] | None = None,
) -> int:
    """Run dual-Piper software leader-follower data collection."""
    return run_python_module("robotarm.teleop.leader_follower.runtime", extra_args)


def convert_piper_raw_dataset(extra_args: Sequence[str] | None = None) -> int:
    """Convert raw Piper episodes into a Lerobot-compatible dataset."""
    return run_python_module("robotarm.record.convert", extra_args)


def merge_piper_raw_sessions(extra_args: Sequence[str] | None = None) -> int:
    """Copy and renumber raw sessions into a new piper_raw_v1 session."""
    return run_python_module("robotarm.record.merge", extra_args)


def run_xr_bridge(extra_args: Sequence[str] | None = None) -> int:
    """Keep the XRoboToolkit SDK session alive behind a local bridge."""
    return run_python_module("robotarm.teleop.xr.bridge_host", extra_args)


def run_piper_xr(extra_args: Sequence[str] | None = None) -> int:
    """Run the Piper XR teleoperation entrypoint."""
    return run_python_module("robotarm.teleop.xr.runtime", extra_args)


def run_piper_bimanual_xr(extra_args: Sequence[str] | None = None) -> int:
    """Run two-controller, two-Piper XR teleoperation."""
    return run_python_module("robotarm.teleop.xr.bimanual_runtime", extra_args)


def restore_piper_role(extra_args: Sequence[str] | None = None) -> int:
    """Restore one Piper to the follower role without moving it."""
    return run_python_module("robotarm.hardware.restore_role", extra_args)


def home_piper(extra_args: Sequence[str] | None = None) -> int:
    """Home one selected Piper without a teleoperation runtime."""
    return run_python_module("robotarm.hardware.home", extra_args)


WORKFLOWS = {
    "home-piper": WorkflowSpec("Home one Piper selected by CAN channel.", home_piper),
    "restore-piper-role": WorkflowSpec(
        "Restore one Piper to follower mode without joint motion.", restore_piper_role
    ),
    "read-status": WorkflowSpec("Read Piper status from the controller.", read_piper_status),
    "enable-control": WorkflowSpec("Enable or disable Piper control.", enable_piper_control),
    "collect-piper": WorkflowSpec("Collect a Piper teleoperation episode.", collect_piper_episode),
    "collect-leader-follower": WorkflowSpec(
        "Collect episodes with two Piper arms in software leader-follower mode.",
        collect_leader_follower_episode,
    ),
    "convert-piper-raw": WorkflowSpec("Convert raw Piper data into a Lerobot dataset.", convert_piper_raw_dataset),
    "merge-piper-raw": WorkflowSpec(
        "Merge raw Piper sessions into a new raw session; preview unless --execute.",
        merge_piper_raw_sessions,
    ),
    "xr-bridge": WorkflowSpec("Run the XRoboToolkit bridge process.", run_xr_bridge),
    "piper-xr": WorkflowSpec("Run the Piper XR teleoperation entrypoint.", run_piper_xr),
    "piper-bimanual-xr": WorkflowSpec(
        "Control two Piper arms with the left and right PICO controllers.",
        run_piper_bimanual_xr,
    ),
}
