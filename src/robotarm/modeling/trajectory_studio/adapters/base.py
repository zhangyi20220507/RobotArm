from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ModelAdapter(ABC):
    name: str
    label: str

    @abstractmethod
    def describe(self, checkpoint: Path | None = None) -> dict[str, Any]:
        """Return frontend capabilities and parameter limits."""

    @abstractmethod
    def evaluate(self, request: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        """Run offline inference and create viewer artifacts."""
