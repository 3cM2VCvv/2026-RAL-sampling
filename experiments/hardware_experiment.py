"""Run the real-world stanley_immrax_exp experiment and capture its log dir.

This is the first stage of the reachset-recording pipeline. It drives the
``stanley_immrax_exp.xml`` config (ImmraxController + Optitrack), so it only
runs on the lab experiment machine with the motion-capture hardware attached.

buzzracer's ``Logger`` extension writes the run logs into the submodule's
``Buzzracer/log/<experiment_name>/`` directory, which is ``.dvcignore``d (the
whole ``Buzzracer`` dir is a dependency of the reachset_trial sweep, so its
hash must stay stable). We therefore treat that location as scratch: clear it,
run the experiment, then copy the produced logs out to ``--out-dir`` (under
``runs/``), which is the DVC-tracked output of this stage.

Usage:
    python experiments/hardware_experiment.py --out-dir runs/stanley_immrax_exp
"""

import argparse
import os
import shutil

import _env  # sets up sys.path for buzzracer + root helpers; canonical dirs

from buzzracer.common import BASEDIR
from buzzracer.scripts.run import Main

EXPERIMENT_NAME = "stanley_immrax_exp"
CONFIG = os.path.join(BASEDIR, "buzzracer", "configs", f"{EXPERIMENT_NAME}.xml")
# Logger writes here (BASEDIR/log/<experiment_name>); .dvcignore'd scratch.
SCRATCH_LOG_DIR = os.path.join(BASEDIR, "log", EXPERIMENT_NAME)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", dest="out_dir", required=True,
                   help="tracked output dir for the captured run logs")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Clean scratch so Logger numbering is deterministic (full_state1.p, ...).
    shutil.rmtree(SCRATCH_LOG_DIR, ignore_errors=True)

    experiment = Main(CONFIG)
    experiment.run()
    experiment.timer.summary()

    if not os.path.isdir(SCRATCH_LOG_DIR):
        raise SystemExit(
            f"expected logs at {SCRATCH_LOG_DIR}; was the Logger extension enabled?"
        )

    # Capture the scratch logs into the tracked output dir.
    if os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)
    shutil.copytree(SCRATCH_LOG_DIR, args.out_dir)
    print(f"captured run logs -> {args.out_dir}")


if __name__ == "__main__":
    main()
