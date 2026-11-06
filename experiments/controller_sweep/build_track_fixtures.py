"""Build the normalized track fixtures used by the controller sweep.

Each supported ``Buzzracer`` track is built once with its native preparation,
any missing ``Track`` fields are derived, and the result is written to
``outputs/controller_sweep/track_fixtures/<name>.npz``.  ``raceline_track.RacelineTrack`` loads that
file back as a complete ``Track``.

The build_track_fixtures DVC stage reads ``controller_sweep.track`` from
``params.yaml`` by default.
"""

import argparse
import os

os.environ["MPLBACKEND"] = "Agg"

import _env
import numpy as np
import yaml
from _track_setup import SUPPORTED_TRACKS, build_fixture

DEFAULT_OUT_DIR = os.path.join(_env.OUTPUTS_DIR, "controller_sweep", "track_fixtures")
DEFAULT_PARAMS = os.path.join(_env.PROJECT_ROOT, "params.yaml")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tracks",
        nargs="+",
        default=None,
        help="track names (default: controller_sweep.track from params.yaml)",
    )
    p.add_argument("--params", default=DEFAULT_PARAMS)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return p.parse_args(argv)


def load_track_list(params_path):
    with open(params_path) as f:
        params = yaml.safe_load(f) or {}
    tracks = params.get("controller_sweep", {}).get("track")
    if not tracks:
        raise SystemExit(f"no controller_sweep.track list in {params_path}")
    return list(tracks)


def main(argv=None):
    args = parse_args(argv)
    names = args.tracks or load_track_list(args.params)
    unknown = [name for name in names if name not in SUPPORTED_TRACKS]
    if unknown:
        raise SystemExit(
            f"unsupported track(s) {unknown}; choose from {list(SUPPORTED_TRACKS)}"
        )

    os.makedirs(args.out_dir, exist_ok=True)
    for filename in os.listdir(args.out_dir):
        if filename.endswith(".npz") and filename[:-4] not in names:
            os.remove(os.path.join(args.out_dir, filename))

    rows = []
    for name in names:
        data, summary = build_fixture(name)
        path = os.path.join(args.out_dir, f"{name}.npz")
        np.savez_compressed(path, **data)
        rows.append((name, summary))
        print(
            f"{name:8s} n={summary['n']:4d} len={summary['length']:7.3f}m "
            f"v=[{summary['v_min']:.2f},{summary['v_max']:.2f}]m/s "
            f"left_min={summary['left_min']:.3f} right_min={summary['right_min']:.3f}"
        )

    print(f"\nWrote {len(rows)} fixture(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
