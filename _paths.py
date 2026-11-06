"""Single source of truth for sys.path setup.

The repo keeps core modules and experiment helpers as flat top-level modules
(the Buzzracer submodule imports ``immrax_controller``, ``replay_controller``
and ``warehouse_track`` by bare name).  Adding their directories here lets
those imports resolve without editing the submodule.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
VENDOR_DIR = REPO_ROOT / "vendor"

_DIRS = (
    REPO_ROOT / "core",
    REPO_ROOT / "vendor" / "Buzzracer" / "buzzracer",
    REPO_ROOT / "experiments" / "shared",
    REPO_ROOT / "experiments" / "reachset_sweep",
    REPO_ROOT / "experiments" / "controller_sweep",
    REPO_ROOT / "experiments" / "dualguard_rc",
    REPO_ROOT / "experiments" / "reachset_recording",
    REPO_ROOT,
)


def setup():
    """Put every import root on ``sys.path`` (idempotent, stable order)."""
    for directory in reversed(_DIRS):
        path = str(directory)
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


setup()
