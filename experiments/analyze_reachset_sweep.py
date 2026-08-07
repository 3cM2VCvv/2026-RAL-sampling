"""Summarize reachset_sweep DVCLive trial outputs.

Walks ``runs/reachset_sweep/*`` (one DVCLive directory per trial, written by
``reachset_sweep.py``) and produces:
  - Outcome breakdown (race_finished / crash / stall / incomplete) by
    noise type and reachset mode.
  - Mean lap time among successful laps by noise type and reachset mode.
  - SVG scatter of crash locations (projected onto the raceline for
    runs that predate ``final_x``/``final_y`` logging; actual positions
    otherwise) overlaid on a track rendering identical to the
    ``Visualization`` extension output.
  - Per-run correlations of failure rate with state_noise_direction
    and velocity_scale_factor.
  - Mean unsafe_fraction (best-trajectory cost was infinite) among
    successful laps by noise type and reachset mode.

Two consistency checks also run, each warning (rather than failing) when it
finds anything:
  - Safety regressions: reachset-enabled runs that crashed while their
    otherwise-identical reachset-disabled twin did not.
  - Unforeseen crashes: reachset-enabled crashes where the filter flagged
    fewer than ``min(total_control_steps, sim.horizon)`` of the final control
    steps unsafe, i.e. the crash was not predicted across the whole planning
    horizon (or across the whole run, when it ended sooner than that).
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml

import _env  # sets up sys.path for buzzracer; canonical dirs

pd.set_option("display.width", None)
pd.set_option("display.max_columns", None)

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import splev

# LaTeX-friendly serif styling: match Computer Modern so SVGs blend into the
# surrounding document. (Underscores are avoided in every rendered label
# because they break SVG inclusion in LaTeX.)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["CMU Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        # 1.5x matplotlib's default 10pt, matching the Vega templates' bump.
        # Everything else (titles, tick labels) is a relative size, so it
        # scales along with this.
        "font.size": 15,
    }
)

from buzzracer.tracks.rcp_track import RCPTrack

CONFIG_KEYS = [
    "use_reachsets",
    "state_noise_direction",
    "centering_weight_factor",
    "velocity_scale_factor",
    "init_point_index",
    "init_offset_name",
    "prng_seed",
]

# Every sweep axis except use_reachsets: two runs agreeing on all of these are
# the same scenario planned with and without the reachable-set filter.
PAIR_KEYS = [k for k in CONFIG_KEYS if k != "use_reachsets"]

METRIC_KEYS = [
    "init_x",
    "init_y",
    "init_heading",
    "init_v_forward",
    "final_x",
    "final_y",
    "collision_occurred",
    "collision_count",
    "stalled",
    "distance_progressed",
    "distance_fraction",
    "unsafe_fraction",
    "total_control_steps",
    "unsafe_control_steps",
    "recent_unsafe_control_steps",
    "num_laps_complete",
    "average_lap_time",
    "sim_duration",
    "oob_fraction",
]


def load_runs(runs_dir):
    """Load every DVCLive trial directory under ``runs_dir``.

    Each trial directory contains ``params.yaml`` (config) and ``metrics.json``
    (metrics plus the ``stop_reason`` summary), as written by ``reachset_sweep.py``.
    A trial is considered COMPLETED iff it produced a ``metrics.json``.
    """
    rows = []
    if not os.path.isdir(runs_dir):
        return pd.DataFrame(rows)
    for name in sorted(os.listdir(runs_dir)):
        rd = os.path.join(runs_dir, name)
        if not os.path.isdir(rd):
            continue
        params_path = os.path.join(rd, "params.yaml")
        metrics_path = os.path.join(rd, "metrics.json")
        if not (os.path.exists(params_path) and os.path.exists(metrics_path)):
            continue

        with open(params_path) as f:
            cfg = yaml.safe_load(f) or {}
        with open(metrics_path) as f:
            metrics = json.load(f)

        row = {"run_id": name, "status": "COMPLETED"}
        for k in CONFIG_KEYS:
            row[k] = cfg.get(k)
        for k in METRIC_KEYS:
            row[k] = metrics.get(k, np.nan)
        row["stop_reason"] = metrics.get("stop_reason")
        rows.append(row)
    return pd.DataFrame(rows)


def categorize_outcome(row):
    sr = row["stop_reason"]
    if sr == "race_finished":
        return "race_finished"
    if sr == "crash":
        return "crash"
    if sr == "stall":
        return "stall"
    return "incomplete"


def project_to_arclength(track, x, y):
    """Closest arc-length s on the raceline to point (x, y)."""
    pts = track.raceline_points
    d2 = (pts[0] - x) ** 2 + (pts[1] - y) ** 2
    return float(track.ss[int(np.argmin(d2))])


def position_at_arclength(track, s):
    """(x, y) on the raceline spline at arc-length s (mod track length)."""
    s_mod = float(s) % track.raceline_len_m
    x, y = splev(s_mod, track.raceline_s, der=0)
    return float(x), float(y)


def add_crash_locations(df, track):
    """Fill crash_x/crash_y per row.

    Prefer logged ``final_x``/``final_y`` when present (true crash
    position including offset). Fall back to projecting
    ``init + distance_progressed`` onto the raceline for older runs
    that predate final-position logging.
    """
    track_len = track.raceline_len_m
    xs, ys = [], []
    for _, row in df.iterrows():
        if row["outcome"] != "crash":
            xs.append(np.nan)
            ys.append(np.nan)
            continue
        if not (pd.isna(row.get("final_x")) or pd.isna(row.get("final_y"))):
            xs.append(float(row["final_x"]))
            ys.append(float(row["final_y"]))
            continue
        if pd.isna(row["init_x"]) or pd.isna(row["distance_progressed"]):
            xs.append(np.nan)
            ys.append(np.nan)
            continue
        s0 = project_to_arclength(track, row["init_x"], row["init_y"])
        s_crash = (s0 + float(row["distance_progressed"])) % track_len
        cx, cy = position_at_arclength(track, s_crash)
        xs.append(cx)
        ys.append(cy)
    df["crash_x"] = xs
    df["crash_y"] = ys
    return df


def outcome_summary(df):
    g = df.groupby(["state_noise_direction", "use_reachsets", "outcome"]).size()
    totals = (
        df.groupby(["state_noise_direction", "use_reachsets"]).size().rename("total")
    )
    counts = g.unstack("outcome", fill_value=0)
    for col in ["race_finished", "crash", "stall", "incomplete"]:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[["race_finished", "crash", "stall", "incomplete"]]
    counts = counts.join(totals)
    pct = (
        counts[["race_finished", "crash", "stall", "incomplete"]].div(
            counts["total"], axis=0
        )
        * 100
    )
    pct.columns = [f"{c}_pct" for c in pct.columns]
    out = counts.join(pct.round(1))
    return out


def lap_time_summary(df):
    success = df[df["outcome"] == "race_finished"]
    return (
        success.groupby(["state_noise_direction", "use_reachsets"])["sim_duration"]
        .agg(["count", "mean", "std", "min", "max"])
        .round(3)
    )


def unsafe_fraction_summary(df):
    success = df[df["outcome"] == "race_finished"]
    return (
        success.groupby(["state_noise_direction", "use_reachsets"])["unsafe_fraction"]
        .agg(["count", "mean", "std", "min", "max"])
        .round(4)
    )


def correlation_failure_vs_velocity(df):
    df = df[df["status"] == "COMPLETED"].copy()
    df["failed"] = df["outcome"].isin(["crash", "stall"]).astype(int)
    rows = []
    for (noise, reach), grp in df.groupby(["state_noise_direction", "use_reachsets"]):
        if grp["velocity_scale_factor"].nunique() < 2 or grp["failed"].nunique() < 2:
            corr = np.nan
        else:
            corr = grp["velocity_scale_factor"].corr(grp["failed"])
        rows.append(
            {
                "state_noise_direction": noise,
                "use_reachsets": reach,
                "n": len(grp),
                "failure_rate": grp["failed"].mean(),
                "pearson_r(vsf, failed)": corr,
            }
        )
    return (
        pd.DataFrame(rows)
        .set_index(["state_noise_direction", "use_reachsets"])
        .round(4)
    )


def correlation_failure_vs_noise_direction(df):
    df = df[df["status"] == "COMPLETED"].copy()
    df["failed"] = df["outcome"].isin(["crash", "stall"]).astype(int)
    df["noise_adv"] = (df["state_noise_direction"] == "adversarial_away").astype(int)
    rows = []
    for reach, grp in df.groupby("use_reachsets"):
        if grp["noise_adv"].nunique() < 2 or grp["failed"].nunique() < 2:
            corr = np.nan
        else:
            corr = grp["noise_adv"].corr(grp["failed"])
        rows.append(
            {
                "use_reachsets": reach,
                "n": len(grp),
                "failure_rate": grp["failed"].mean(),
                "pearson_r(noise_adversarial, failed)": corr,
            }
        )
    return pd.DataFrame(rows).set_index(["use_reachsets"]).round(4)


def load_planning_horizon(params_path):
    """Number of control steps the planner predicts ahead (``sim.horizon``).

    Only the root ``params.yaml`` carries it: the per-trial ``params.yaml``
    records the swept axes, and the horizon is passed to every trial as a
    fixed ``--horizon`` flag.
    """
    with open(params_path) as f:
        params = yaml.safe_load(f) or {}
    return int(params["sim"]["horizon"])


def reachset_only_crashes(df):
    """Runs the reachable-set filter crashed but the trajectory-only planner did not.

    Joins each reachsets-enabled run to the reachsets-disabled run sharing
    every other sweep axis, and keeps the pairs where enabling the filter
    turned a non-crash into a crash. Returns the pair columns plus both
    outcomes, empty when the filter never made things worse.
    """
    done = df[df["status"] == "COMPLETED"]
    on = done[done["use_reachsets"] == True]  # noqa: E712 - pandas mask, not `is`
    off = done[done["use_reachsets"] == False]  # noqa: E712
    if on.empty or off.empty:
        return pd.DataFrame(columns=[*PAIR_KEYS, "run_id", "twin_run_id", "twin_outcome"])

    pairs = on[[*PAIR_KEYS, "run_id", "outcome"]].merge(
        off[[*PAIR_KEYS, "run_id", "outcome"]],
        on=PAIR_KEYS,
        suffixes=("", "_twin"),
        how="inner",
    )
    regressed = pairs[(pairs["outcome"] == "crash") & (pairs["outcome_twin"] != "crash")]
    return regressed.rename(
        columns={"run_id_twin": "twin_run_id", "outcome_twin": "twin_outcome"}
    )[[*PAIR_KEYS, "run_id", "twin_run_id", "twin_outcome"]].reset_index(drop=True)


def unforeseen_crashes(df, planning_horizon):
    """Reachset-enabled crashes the filter did not flag for the whole horizon.

    ``recent_unsafe_control_steps`` counts how many of the final
    ``planning_horizon`` control steps were flagged unsafe. A crash the filter
    saw coming should have all of them flagged, so anything short of that is
    worth a look -- except that a run which ended before the horizon elapsed
    never had that many steps to flag, so the bar is capped at the run's own
    ``total_control_steps``.
    """
    done = df[df["status"] == "COMPLETED"]
    crashed = done[(done["use_reachsets"] == True) & (done["outcome"] == "crash")].copy()  # noqa: E712
    crashed["predicted_steps"] = crashed["total_control_steps"].clip(
        upper=planning_horizon
    )
    short = crashed[
        crashed["recent_unsafe_control_steps"] < crashed["predicted_steps"]
    ]
    return short[
        [
            *PAIR_KEYS,
            "run_id",
            "total_control_steps",
            "predicted_steps",
            "recent_unsafe_control_steps",
        ]
    ].reset_index(drop=True)


def render_track_background(track):
    """Render the track exactly as the ``Visualization`` extension does."""
    img = track.draw_track()
    img = track.draw_raceline(img=img)
    img = track.plot_obstacles(img)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def plot_crash_scatter(df, track, out_path):
    crashes = df.dropna(subset=["crash_x", "crash_y"]).copy()
    bg = render_track_background(track)

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(bg, extent=(0.0, track.x_limit, 0.0, track.y_limit), origin="upper")

    color_map = {True: "tab:orange", False: "tab:blue"}
    marker_map = {True: "o", False: "x"}
    for reach_mode, sub in crashes.groupby("use_reachsets"):
        state = "enabled" if reach_mode else "disabled"
        ax.scatter(
            sub["crash_x"],
            sub["crash_y"],
            c=color_map[reach_mode],
            marker=marker_map[reach_mode],
            s=40,
            alpha=0.6,
            label=f"Reachable-set filter {state} (n={len(sub)})",
            edgecolors="none" if reach_mode else None,
        )

    ax.set_aspect("equal")
    ax.set_xlim(0.0, track.x_limit)
    ax.set_ylim(0.0, track.y_limit)
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_title("Crash locations on the track")
    # Pinned rather than "best": at the larger font size the automatic
    # placement drifts to the middle of the track.
    ax.legend(loc="upper right", fontsize=13.5)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        default=_env.runs_dir("reachset_sweep"),
        help="Directory of reachset_sweep DVCLive trial dirs (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=_env.analysis_dir("reachset_sweep"),
        help="Directory for analysis output (default: %(default)s)",
    )
    parser.add_argument(
        "--planning-horizon",
        type=int,
        default=None,
        help="Control steps the planner predicts ahead (default: sim.horizon "
        "from the root params.yaml)",
    )
    args = parser.parse_args(argv)
    runs_dir = args.runs_dir
    output_dir = args.output_dir
    planning_horizon = args.planning_horizon
    if planning_horizon is None:
        planning_horizon = load_planning_horizon(
            os.path.join(_env.PROJECT_ROOT, "params.yaml")
        )

    print(f"Loading runs from {runs_dir} ...")
    df = load_runs(runs_dir)
    if df.empty:
        print(f"No runs found in {runs_dir}.")
        return
    print(f"Loaded {len(df)} runs ({(df['status'] == 'COMPLETED').sum()} completed)")

    os.makedirs(output_dir, exist_ok=True)

    df["outcome"] = df.apply(categorize_outcome, axis=1)

    track = RCPTrack(main=None, config=None)
    track.load()
    df = add_crash_locations(df, track)

    tables = [
        (
            "Outcome counts and percentages by noise type and reachset mode",
            outcome_summary(df),
        ),
        (
            "Lap time (sim_duration, s) of successful laps by noise type and reachset mode",
            lap_time_summary(df),
        ),
        (
            "Failure rate vs state_noise_direction (adversarial_away=1, uniform=0) by reachset mode",
            correlation_failure_vs_noise_direction(df),
        ),
        (
            "Failure rate vs velocity_scale_factor by noise type and reachset mode",
            correlation_failure_vs_velocity(df),
        ),
        (
            "Unsafe-control fraction (best cost was inf) for successful laps by noise type and reachset mode",
            unsafe_fraction_summary(df),
        ),
    ]

    for title, table in tables:
        print(f"\n{title}")
        print(table)

    regressions = reachset_only_crashes(df)
    print("\nSafety regressions (reachsets crashed, trajectory-only planner did not)")
    if regressions.empty:
        print("None.")
    else:
        print(
            f"WARNING: {len(regressions)} run(s) crashed with the reachable-set "
            "filter enabled whose otherwise-identical filter-disabled twin did not:"
        )
        print(regressions.to_string(index=False))

    unforeseen = unforeseen_crashes(df, planning_horizon)
    print(
        f"\nUnforeseen crashes (reachsets crashed with recent_unsafe_control_steps"
        f" < min(total_control_steps, {planning_horizon}))"
    )
    if unforeseen.empty:
        print("None.")
    else:
        print(
            f"WARNING: {len(unforeseen)} reachset-enabled crash(es) flagged "
            "fewer unsafe steps than the planning horizon predicts (or than "
            "the run was long, whichever is smaller):"
        )
        print(unforeseen.to_string(index=False))

    csv_path = os.path.join(output_dir, "runs.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nWrote per-run dataframe: {csv_path}")

    svg_path = os.path.join(output_dir, "crash_locations.svg")
    plot_crash_scatter(df, track, svg_path)
    print(f"Wrote crash scatter:    {svg_path}")


if __name__ == "__main__":
    main()
