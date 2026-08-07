"""Shared environment setup and canonical paths for the experiment scripts.

This is the single source of truth for:

  1. ``sys.path`` setup. The buzzracer package modules use bare
     ``from common import *`` (which needs ``Buzzracer/buzzracer/`` on the
     path), and the project-root helper modules (``immrax_controller``,
     ``reachset_cost``, ``immrax_dynamic_bicycle_*``, ``replay_controller``)
     are imported as top-level modules (which needs the project root on the
     path). Importing this module performs that setup once.

  2. Input/output directories used by the experiments and analysis scripts.
     Defaults can be overridden with environment variables so the experiment
     scripts (which own their own CLI) can be redirected without code edits.

Every experiment script imports this module *before* importing anything from
buzzracer or the root-level helpers. Because each script is run as
``python experiments/<script>.py``, the ``experiments/`` directory is always
``sys.path[0]``, so a bare ``import _env`` resolves regardless of the working
directory.

Environment overrides:
    BUZZRACER_RUNS_DIR      -> RUNS_DIR     (per-trial run output)
    BUZZRACER_ANALYSIS_DIR  -> ANALYSIS_DIR (analysis script output)
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
BUZZRACER_PKG_DIR = os.path.join(PROJECT_ROOT, "Buzzracer", "buzzracer")


def _setup_import_paths():
    """Put the buzzracer package dir and project root on ``sys.path`` (once)."""
    for path in (BUZZRACER_PKG_DIR, PROJECT_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)


_setup_import_paths()


# Canonical input/output directories (env-overridable).
RUNS_DIR = os.environ.get("BUZZRACER_RUNS_DIR", os.path.join(PROJECT_ROOT, "runs"))
ANALYSIS_DIR = os.environ.get(
    "BUZZRACER_ANALYSIS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_output"),
)


def runs_dir(experiment_name):
    """Return the run-output directory for ``experiment_name``."""
    return os.path.join(RUNS_DIR, experiment_name)


def analysis_dir(experiment_name):
    """Return the analysis-output directory for ``experiment_name``."""
    return os.path.join(ANALYSIS_DIR, experiment_name)
