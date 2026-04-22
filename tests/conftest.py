"""Shared pytest fixtures + sys.path bootstrap.

Ensures the repo root is on sys.path so ``import hmd`` resolves when tests are
run from anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
