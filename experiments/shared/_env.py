"""Canonical input/output paths for the experiment scripts.

Import paths are installed by ``_paths`` (through the ``experiments`` package);
this module only resolves the directories experiments read from and write to.

Environment overrides:
    ONLINE_REACHABILITY_OUTPUTS_DIR -> OUTPUTS_DIR
    BUZZRACER_RUNS_DIR              -> legacy run root; overrides runs_dir()
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = Path(
    os.environ.get("ONLINE_REACHABILITY_OUTPUTS_DIR", PROJECT_ROOT / "outputs")
)
_LEGACY_RUNS_ROOT = os.environ.get("BUZZRACER_RUNS_DIR")


def runs_dir(experiment_name):
    """Return the per-trial run directory for ``experiment_name``."""
    if _LEGACY_RUNS_ROOT is not None:
        return os.path.join(_LEGACY_RUNS_ROOT, experiment_name)
    return str(OUTPUTS_DIR / experiment_name / "runs")


def analysis_dir(experiment_name, part="analysis"):
    """Return an analysis-output directory for ``experiment_name``."""
    return str(OUTPUTS_DIR / experiment_name / part)
