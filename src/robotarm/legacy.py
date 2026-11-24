from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence

from .teleop.keyboard import preserve_terminal_settings


def _normalize_args(extra_args: Sequence[str] | None) -> list[str]:
    if not extra_args:
        return []
    return [str(arg) for arg in extra_args]


def run_process(command: Sequence[str], cwd: Path | None = None) -> int:
    # Own restoration outside the entire child lifetime: child atexit handlers
    # and SDK/encoder shutdown must finish before handing the TTY to the shell.
    with preserve_terminal_settings(ensure_shell_input=True):
        process = subprocess.Popen(
            [str(item) for item in command], cwd=str(cwd) if cwd else None,
        )
        interrupted = False
        while True:
            try:
                code = process.wait()
                break
            except KeyboardInterrupt:
                # Ctrl+C reaches both processes through the terminal group.
                # Keep waiting for the child's homing/cleanup; returning here
                # creates a shell prompt while the child still owns the TTY.
                # Further Ctrl+C still reaches the child for its emergency stop.
                interrupted = True
        return 130 if interrupted else int(code)


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

