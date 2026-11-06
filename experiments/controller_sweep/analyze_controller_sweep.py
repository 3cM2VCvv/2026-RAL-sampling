"""Summarize the controller × track × disturbance sweep.

Walks ``outputs/controller_sweep/runs/*`` (one DVCLive directory per trial, written by
``controller_sweep.py``) and produces:

  - ``runs.csv``   one row per trial: config, outcome, control-loop time,
                   average lap time, and the usual progress/safety metrics.
  - ``summary.csv`` one row per (controller, track, disturbance) cell:
                   outcome counts/percentages, mean control-loop time, and
                   mean lap time over trials that completed at least one lap.

``controller_sweep.track``, resolved through ``dvc.api.params_show`` for this
stage, limits the summary to the sweep's configured tracks; runs left behind by
tracks removed from the matrix are ignored.  Outcome is read from the trial's
recorded ``outcome`` summary, falling back to the same ``stop_reason`` mapping
used by ``analyze_reachset_sweep.py``.
"""

import os

import _env
import dvc.api
import pandas as pd
from analysis_utils import OUTCOMES as _OUTCOMES
from analysis_utils import load_runs as _load_runs

CONFIG_KEYS = [
    "controller",
    "track",
    "state_noise_direction",
    "prng_seed",
    "trial_seed",
]

METRIC_KEYS = [
    "init_x",
    "init_y",
    "init_heading",
    "init_v_forward",
    "track_len",
    "final_x",
    "final_y",
    "collision_occurred",
    "collision_count",
    "stalled",
    "distance_progressed",
    "distance_fraction",
    "num_laps_complete",
    "average_lap_time",
    "control_time_ms_mean",
    "control_time_ms_std",
    "control_time_ms_median",
    "control_steps_timed",
    "planner_compute_ms_mean",
    "unsafe_fraction",
    "total_control_steps",
    "unsafe_control_steps",
    "recent_unsafe_control_steps",
    "sim_duration",
    "oob_fraction",
]


def load_runs(runs_dir):
    return _load_runs(
        runs_dir,
        config_keys=CONFIG_KEYS,
        metric_keys=METRIC_KEYS,
        with_status=True,
        with_outcome=True,
    )


def configured_tracks():
    """Track names declared for this stage in ``dvc.yaml``."""
    params = dvc.api.params_show(stages="analyze_controller_sweep")
    return params["controller_sweep"]["track"]


def filter_tracks(df, tracks):
    """Drop runs from tracks outside the configured sweep."""
    if df.empty or not tracks:
        return df
    keep = df["track"].isin(tracks)
    dropped = df.loc[~keep, "track"].unique()
    if len(dropped):
        print(f"Ignoring {int((~keep).sum())} run(s) from track(s) {sorted(dropped)}")
    return df[keep]


def outcome_summary(df):
    groups = ["controller", "track", "state_noise_direction"]
    counts = df.groupby(groups + ["outcome"]).size().unstack("outcome", fill_value=0)
    for col in _OUTCOMES:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[_OUTCOMES]
    counts["n_trials"] = counts.sum(axis=1)
    pct = counts[_OUTCOMES].div(counts["n_trials"], axis=0) * 100
    pct.columns = [f"{col}_pct" for col in pct.columns]
    return counts.join(pct.round(1))


def performance_summary(df):
    groups = ["controller", "track", "state_noise_direction"]
    has_laps = df["num_laps_complete"].fillna(0) > 0
    lap = (
        df[has_laps]
        .groupby(groups)["average_lap_time"]
        .agg(["mean", "std", "count"])
        .rename(
            columns={
                "mean": "average_lap_time_mean",
                "std": "average_lap_time_std",
                "count": "lap_trials",
            }
        )
    )
    perf = (
        df.groupby(groups)
        .agg(
            n_trials=("run_id", "size"),
            control_time_ms_mean=("control_time_ms_mean", "mean"),
            control_time_ms_std=("control_time_ms_mean", "std"),
            num_laps_complete_mean=("num_laps_complete", "mean"),
            sim_duration_mean=("sim_duration", "mean"),
        )
        .join(lap)
    )
    return perf.round(3)


def controller_overall(df):
    has_laps = df["num_laps_complete"].fillna(0) > 0
    lap = (
        df[has_laps]
        .groupby("controller")["average_lap_time"]
        .agg(["mean", "std", "count"])
        .rename(
            columns={
                "mean": "average_lap_time_mean",
                "std": "average_lap_time_std",
                "count": "lap_trials",
            }
        )
    )
    out = (
        df.groupby("controller")
        .agg(
            n_trials=("run_id", "size"),
            control_time_ms_mean=("control_time_ms_mean", "mean"),
            num_laps_complete_mean=("num_laps_complete", "mean"),
            race_finished_rate=(
                "outcome",
                lambda s: float((s == "race_finished").mean()),
            ),
            crash_rate=("outcome", lambda s: float((s == "crash").mean())),
            stall_rate=("outcome", lambda s: float((s == "stall").mean())),
        )
        .join(lap)
    )
    return out.round(3)


def main():
    runs_dir = _env.runs_dir("controller_sweep")
    out_dir = _env.analysis_dir("controller_sweep")
    os.makedirs(out_dir, exist_ok=True)

    df = filter_tracks(load_runs(runs_dir), configured_tracks())
    if df.empty:
        print(f"No runs found in {runs_dir}.")
        return

    print(f"Loaded {len(df)} trial(s) from {runs_dir}")

    outcomes = outcome_summary(df)
    perf = performance_summary(df)
    summary = outcomes.join(perf.drop(columns=["n_trials"]))
    overall = controller_overall(df)

    pd.set_option("display.width", None)
    pd.set_option("display.max_columns", None)
    print("\nOutcome counts and percentages:")
    print(outcomes.to_string())
    print("\nPerformance by controller / track / disturbance:")
    print(perf.to_string())
    print("\nController overall:")
    print(overall.to_string())

    runs_csv = os.path.join(out_dir, "runs.csv")
    summary_csv = os.path.join(out_dir, "summary.csv")
    df.to_csv(runs_csv, index=False)
    summary.to_csv(summary_csv)
    print(f"\nWrote per-trial dataframe: {runs_csv}")
    print(f"Wrote summary:              {summary_csv}")


if __name__ == "__main__":
    main()
