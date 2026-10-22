from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _normalize_args(extra_args: Sequence[str] | None) -> list[str]:
    if not extra_args:
        return []
    return [str(arg) for arg in extra_args]


def run_process(command: Sequence[str], cwd: Path | None = None) -> int:
    try:
        result = subprocess.run([str(item) for item in command], cwd=str(cwd) if cwd else None, check=False)
    except KeyboardInterrupt:
        # The terminal also delivers Ctrl+C to the child; it handles its own
        # motion cleanup. Do not print a second traceback from this wrapper.
        return 130
    return int(result.returncode)


def run_python_script(script: Path, extra_args: Sequence[str] | None = None, cwd: Path | None = None) -> int:
    script = script.resolve()
    if not script.is_file():
        raise FileNotFoundError(f"Python script not found: {script}")
    return run_process([sys.executable, str(script), *_normalize_args(extra_args)], cwd=cwd or script.parent)


def run_python_module(module: str, extra_args: Sequence[str] | None = None, cwd: Path | None = None, python: str | None = None) -> int:
    return run_process([python or sys.executable, "-m", module, *_normalize_args(extra_args)], cwd=cwd)


def run_shell_script(script: Path, extra_args: Sequence[str] | None = None, cwd: Path | None = None) -> int:
    script = script.resolve()
    if not script.is_file():
        raise FileNotFoundError(f"Shell script not found: {script}")
    return run_process(["bash", str(script), *_normalize_args(extra_args)], cwd=cwd or script.parent)

