"""Shared paths and ``sys.path`` setup for the DualGuard-MPPI scripts.

Mirrors ``experiments/_env.py``: the project-root helper modules
(``warehouse_track``) are imported as top-level modules, and buzzracer package
modules use bare ``from common import *``, so both the project root and
``Buzzracer/buzzracer/`` have to be on the path.  Importing this module does that
once.

Every path below is resolved from this file, not from the working directory, so
the scripts run identically whether DVC invokes them from ``dualguard_mppi/`` or
you invoke them by hand from anywhere else.
"""

import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
DUALGUARD_DIR = os.path.abspath(os.path.join(SCRIPTS_DIR, os.pardir))
PROJECT_ROOT = os.path.abspath(os.path.join(DUALGUARD_DIR, os.pardir))
BUZZRACER_PKG_DIR = os.path.join(PROJECT_ROOT, "Buzzracer", "buzzracer")

for _path in (BUZZRACER_PKG_DIR, PROJECT_ROOT, SCRIPTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)


MATLAB_DIR = os.path.join(DUALGUARD_DIR, "matlab")
RUNS_DIR = os.path.join(DUALGUARD_DIR, "runs")

# Inputs, all dvc-tracked (`dvc pull` / `dvc repro` puts them on disk).
BRT_RC_WH = os.path.join(MATLAB_DIR, "brt_rc_wh_track.mat")
BRT_BUZZRACER = os.path.join(MATLAB_DIR, "brt_buzzracer_environment.mat")
BUZZRACER_TRACK = os.path.join(MATLAB_DIR, "buzzracer_track_data.mat")


def require(path, hint):
    """Fail with a fetch instruction rather than a bare FileNotFoundError."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} is missing.  It is dvc-tracked, not committed; fetch it with\n"
            f"    uv run dvc pull {hint}\n"
            f"or rebuild it with\n"
            f"    uv run dvc repro {hint}"
        )
    return path
