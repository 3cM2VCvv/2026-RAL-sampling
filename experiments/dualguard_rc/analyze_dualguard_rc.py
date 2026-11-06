"""Summarize the DualGuard-MPPI RC-car comparison trials.

Walks ``outputs/dualguard_rc/runs/*`` (one DVCLive directory per trial, written by
``dualguard_rc_trial.py``), aggregates CompTime / Speed / cost / safety by
planner mode (reachset-aware vs. point), and writes a summary CSV that places
our numbers next to the paper's Table III ("DualGuard (Ours)": CompTime
2.5 +/- 0.4 ms, Speed 1.10 +/- 0.11 m/s, RelCost 1.000).

RelCost is intentionally NOT computed: per the plan we report absolute eq.-16
cost only (no same-track DualGuard reference run to normalize against).
"""

import os

import _env
import numpy as np
import pandas as pd
from analysis_utils import load_runs as _load_runs

CONFIG_KEYS = [
    "use_reachsets",
    "cost_type",
    "buffer",
    "prng_seed",
    "horizon",
    "num_samples",
    "centering_weight_factor",
    "progress_weight",
    "delta_max_deg",
    "v_max",
    "disturbance",
]

METRIC_KEYS = [
    "comptime_ms_mean",
    "comptime_ms_std",
    "comptime_ms_median",
    "speed_mean",
    "speed_std",
    "num_laps_complete",
    "total_cost",
    "cost_velocity",
    "cost_centerline",
    "mean_cost_per_step",
    "unsafe_fraction",
    "boundary_exit_fraction",
    "crashed",
    "distance_progressed",
    "sim_duration",
]

# Paper Table III, "DualGuard (Ours)" reference (for side-by-side context).
PAPER_DUALGUARD = {
    "comptime_ms_mean": 2.5,
    "comptime_ms_std": 0.4,
    "speed_mean": 1.10,
    "speed_std": 0.11,
    "relcost": 1.000,
}


def load_runs(runs_dir):
    return _load_runs(
        runs_dir,
        config_keys=CONFIG_KEYS,
        metric_keys=METRIC_KEYS,
    )


def summarize(df):
    """One row per (planner mode, cost formulation): mean over seeds.

    Note the two cost_type arms are not cost-comparable to each other -- `total_cost`
    is in each arm's own units (l(x) deficit vs. Frenet lateral error).  CompTime,
    Speed, laps and the safety fractions are.
    """
    if df.empty:
        return pd.DataFrame()

    def method_name(use_reachsets, cost_type):
        planner = "Reachset-MPPI (ours)" if use_reachsets else "Point-MPPI"
        return f"{planner} [{cost_type}]"

    out = []
    for (use_reachsets, cost_type), group in df.groupby(["use_reachsets", "cost_type"]):
        rec = {
            "method": method_name(use_reachsets, cost_type),
            "n_trials": len(group),
            # CompTime: mean of per-trial means; std pooled as mean of per-trial stds.
            "comptime_ms_mean": group["comptime_ms_mean"].mean(),
            "comptime_ms_std": group["comptime_ms_std"].mean(),
            # Speed: mean/std across trials that completed at least one lap.
            "speed_mean": group.loc[
                group["num_laps_complete"] > 0, "speed_mean"
            ].mean(),
            "speed_std": group.loc[group["num_laps_complete"] > 0, "speed_mean"].std(),
            "laps_mean": group["num_laps_complete"].mean(),
            "total_cost_mean": group["total_cost"].mean(),
            "mean_cost_per_step": group["mean_cost_per_step"].mean(),
            "unsafe_fraction_mean": group["unsafe_fraction"].mean(),
            "boundary_exit_fraction_mean": group["boundary_exit_fraction"].mean(),
            "crashed_fraction": group["crashed"].mean(),
        }
        out.append(rec)

    # Paper reference row for side-by-side comparison.
    out.append(
        {
            "method": "DualGuard (paper, Table III)",
            "n_trials": np.nan,
            "comptime_ms_mean": PAPER_DUALGUARD["comptime_ms_mean"],
            "comptime_ms_std": PAPER_DUALGUARD["comptime_ms_std"],
            "speed_mean": PAPER_DUALGUARD["speed_mean"],
            "speed_std": PAPER_DUALGUARD["speed_std"],
            "laps_mean": np.nan,
            "total_cost_mean": np.nan,
            "mean_cost_per_step": np.nan,
            "unsafe_fraction_mean": np.nan,
            "boundary_exit_fraction_mean": np.nan,
            "crashed_fraction": np.nan,
        }
    )
    return pd.DataFrame(out)


def main():
    runs_dir = _env.runs_dir("dualguard_rc")
    out_dir = _env.analysis_dir("dualguard_rc")
    os.makedirs(out_dir, exist_ok=True)

    df = load_runs(runs_dir)
    summary = summarize(df)

    out_csv = os.path.join(out_dir, "summary.csv")
    summary.to_csv(out_csv, index=False)

    pd.set_option("display.width", None)
    pd.set_option("display.max_columns", None)
    print(f"Loaded {len(df)} trial(s) from {runs_dir}")
    if not df.empty:
        print("\nPer-trial:")
        print(df.to_string(index=False))
    print("\nSummary vs. paper Table III:")
    print(summary.to_string(index=False))
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
