#!/usr/bin/env python3
"""Direct wrapper for robotarm.modeling.dataset_extend."""

from pathlib import Path
import sys


# Also work from a source checkout before ``pip install -e .``.
source_root = Path(__file__).resolve().parents[1] / "src"
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

from robotarm.modeling.dataset_extend import main


if __name__ == "__main__":
    raise SystemExit(main())
