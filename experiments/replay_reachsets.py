"""Replay a captured run through ReplayController and capture the reachset webp.

Second stage of the reachset-recording pipeline. ``replay_immrax.xml`` drives
``ReplayController`` (a repo-root module), which reads the logged controls from
``Buzzracer/log/<log_dir>/`` and renders a transparent reachsets-only animated
WebP into ``Buzzracer/gifs/replays/<log_dir basename>/``.

Both of those submodule locations are ``.dvcignore``d scratch (the whole
``Buzzracer`` dir is a dependency of the reachset_trial sweep, so its content
hash must stay stable -- see hardware_experiment.py). So this wrapper:

  1. stages the tracked input log (``--log-dir``, a DVC dep produced by the
     hardware_experiment stage) into a private scratch dir under the ignored
     ``Buzzracer/log/``;
  2. runs the replay against a *temporary* XML config derived from
     ``replay_immrax.xml`` with ``log_dir`` pointed at that scratch dir, via the
     shared ``_buzzracer_config.write_temp_config`` helper -- so the committed
     submodule config is never edited (keeping the sweep's Buzzracer dep hash
     untouched);
  3. copies the produced WebP and the per-step timestamp log out to
     ``--out-dir`` (under ``runs/``) -- the DVC-tracked output of this stage.

The timestamp log (``full_state1.p``, captured as ``timestamps.pkl``) is what
blend_retime_reachsets.py reads to resample frames to wall-clock time.

Usage:
    python experiments/replay_reachsets.py \
        --log-dir runs/stanley_immrax_exp --out-dir runs/stanley_immrax_replay
"""

import argparse
import os
import shutil

import _env  # sets up sys.path for buzzracer + root helpers; canonical dirs
from _buzzracer_config import write_temp_config

from buzzracer.common import BASEDIR
from buzzracer.scripts.run import Main

# Template config (read-only) and the scratch name we override it to use.
TEMPLATE_CONFIG = os.path.join(BASEDIR, "buzzracer", "configs", "replay_immrax.xml")
SCRATCH_NAME = "stanley_immrax_replay"
LOG_NUM = 1  # replay_immrax.xml's controller log_num

# Scratch locations inside the (.dvcignore'd) submodule.
SCRATCH_LOG_DIR = os.path.join(BASEDIR, "log", SCRATCH_NAME)
SCRATCH_GIF_DIR = os.path.join(BASEDIR, "gifs", "replays", SCRATCH_NAME)
SCRATCH_WEBP = os.path.join(SCRATCH_GIF_DIR, f"reachsets_{LOG_NUM}.webp")
SCRATCH_TIMESTAMPS = os.path.join(SCRATCH_LOG_DIR, f"full_state{LOG_NUM}.p")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log-dir", dest="log_dir", required=True,
                   help="tracked run-log dir to replay (hardware_experiment out)")
    p.add_argument("--out-dir", dest="out_dir", required=True,
                   help="tracked output dir for the captured webp + timestamps")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # 1. Stage the tracked input log into the scratch path the controller reads.
    shutil.rmtree(SCRATCH_LOG_DIR, ignore_errors=True)
    shutil.copytree(args.log_dir, SCRATCH_LOG_DIR)

    # 2. Clean gif scratch so webp numbering is deterministic.
    shutil.rmtree(SCRATCH_GIF_DIR, ignore_errors=True)

    # Derive a temp config that reads from our scratch log dir, leaving the
    # committed submodule XML untouched.
    def point_log_dir_at_scratch(config):
        config.getElementsByTagName("controller")[0].setAttribute(
            "log_dir", f"log/{SCRATCH_NAME}"
        )

    tmp_config = write_temp_config(TEMPLATE_CONFIG, point_log_dir_at_scratch)
    try:
        experiment = Main(tmp_config)
        experiment.run()
        experiment.timer.summary()
    finally:
        os.unlink(tmp_config)

    if not os.path.isfile(SCRATCH_WEBP):
        raise SystemExit(f"expected reachset webp at {SCRATCH_WEBP}; replay failed?")

    # 3. Capture the webp + timestamp log into the tracked output dir.
    os.makedirs(args.out_dir, exist_ok=True)
    shutil.copyfile(SCRATCH_WEBP, os.path.join(args.out_dir, "reachsets.webp"))
    shutil.copyfile(SCRATCH_TIMESTAMPS, os.path.join(args.out_dir, "timestamps.pkl"))
    print(f"captured replay artifacts -> {args.out_dir}")


if __name__ == "__main__":
    main()
