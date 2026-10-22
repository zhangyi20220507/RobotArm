from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence

from . import data_acquisition, modeling, simulation
from .workflows import WorkflowSpec


def _strip_remainder(extra_args: Sequence[str] | None) -> list[str]:
    if not extra_args:
        return []
    result = list(extra_args)
    if result and result[0] == "--":
        result = result[1:]
    return result


def _add_group_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    help_text: str,
    workflows: Mapping[str, WorkflowSpec],
) -> None:
    parser = subparsers.add_parser(name, help=help_text)
    group = parser.add_subparsers(dest="workflow", required=True)
    for command_name, workflow in workflows.items():
        command_parser = group.add_parser(command_name, help=workflow.help_text)
        command_parser.add_argument("extra_args", nargs=argparse.REMAINDER)
        command_parser.set_defaults(handler=workflow.handler)


WORKFLOW_GROUPS: dict[str, tuple[str, Mapping[str, WorkflowSpec]]] = {
    "data-acquisition": ("Data acquisition workflows", data_acquisition.WORKFLOWS),
    "modeling": ("Training and inference workflows", modeling.WORKFLOWS),
    "simulation": ("Simulation and visualization workflows", simulation.WORKFLOWS),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robotarm",
        description="Unified module entrypoint for the RobotArm workspace",
    )
    subparsers = parser.add_subparsers(dest="module", required=True)

    for name, (help_text, workflows) in WORKFLOW_GROUPS.items():
        _add_group_parser(subparsers, name, help_text, workflows)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler")
    extra_args = _strip_remainder(getattr(args, "extra_args", []))
    return int(handler(extra_args))


if __name__ == "__main__":
    raise SystemExit(main())
