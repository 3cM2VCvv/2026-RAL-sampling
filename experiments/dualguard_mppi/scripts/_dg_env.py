"""Canonical paths for the DualGuard-MPPI scripts.

Import paths are installed by ``_paths`` (through the ``experiments`` package);
every path below is resolved from this file, not the working directory.
"""

import os

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
DUALGUARD_DIR = os.path.abspath(os.path.join(SCRIPTS_DIR, os.pardir))
PROJECT_ROOT = os.path.abspath(os.path.join(DUALGUARD_DIR, os.pardir, os.pardir))

MATLAB_DIR = os.path.join(PROJECT_ROOT, "outputs", "dualguard_mppi", "matlab")
RUNS_DIR = os.path.join(PROJECT_ROOT, "outputs", "dualguard_mppi", "runs")

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
