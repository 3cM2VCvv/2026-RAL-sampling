"""Shared DVCLive trial loading and outcome mapping for analysis scripts."""

import json
import os

import numpy as np
import pandas as pd
import yaml

OUTCOMES = ["race_finished", "crash", "stall", "incomplete"]


def outcome_from_stop_reason(stop_reason):
    """Map a simulator stop reason to the coarse analysis outcome."""
    if stop_reason == "race_finished":
        return "race_finished"
    if stop_reason == "crash":
        return "crash"
    if stop_reason == "stall":
        return "stall"
    return "incomplete"


def load_runs(
    runs_dir,
    *,
    config_keys,
    metric_keys,
    with_status=False,
    with_outcome=False,
    extra=None,
):
    """Load every complete DVCLive trial directory under ``runs_dir``.

    A trial is complete iff it has both ``params.yaml`` and ``metrics.json``.
    ``extra(row, metrics)`` may add experiment-specific columns.
    """
    rows = []
    if not os.path.isdir(runs_dir):
        return pd.DataFrame(rows)
    for name in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, name)
        if not os.path.isdir(run_dir):
            continue
        params_path = os.path.join(run_dir, "params.yaml")
        metrics_path = os.path.join(run_dir, "metrics.json")
        if not (os.path.exists(params_path) and os.path.exists(metrics_path)):
            continue

        with open(params_path) as f:
            cfg = yaml.safe_load(f) or {}
        with open(metrics_path) as f:
            metrics = json.load(f)

        row = {"run_id": name}
        if with_status:
            row["status"] = "COMPLETED"
        for key in config_keys:
            row[key] = cfg.get(key)
        for key in metric_keys:
            row[key] = metrics.get(key, np.nan)
        if extra is not None:
            extra(row, metrics)
        row["stop_reason"] = metrics.get("stop_reason")
        if with_outcome:
            row["outcome"] = metrics.get("outcome") or outcome_from_stop_reason(
                row["stop_reason"]
            )
        rows.append(row)
    return pd.DataFrame(rows)
