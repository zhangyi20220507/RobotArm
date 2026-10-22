from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

CommandHandler = Callable[[Sequence[str] | None], int]


@dataclass(frozen=True, slots=True)
class WorkflowSpec:
    help_text: str
    handler: CommandHandler
