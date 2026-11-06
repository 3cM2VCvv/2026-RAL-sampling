"""Experiment entry points; importing this package installs sys.path."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import _paths  # noqa: F401
