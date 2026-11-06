"""Inspect reachset-enabled sweep crashes under scaled adversarial disturbances.

For every ``reachset_trial`` run with ``use_reachsets=True`` that ended in a crash,
reports those whose ``unsafe_fraction`` was below 1 separately from those where the
planner flagged every control step unsafe.  For the fully-unsafe crashes it
re-initializes ``ImmraxController`` at each trial's starting state, samples one
control batch, and rolls every control sequence through the point dynamics under the
away-from-raceline adversarial disturbance.

The adversarial disturbance is scaled by each factor in ``SCALES``.  A crash counts
as fully unsafe at a factor only when *every* sampled control sequence collides at
that factor; its reported ``min_scale_factor`` is the smallest factor that satisfies
this.  The smallest factor making every crash fully unsafe is the largest of the
per-crash minima.

All ``compute_trajectory`` predictions -- every crash, control sequence, and scale --
are evaluated in one ``jax.lax.map`` call.  Its ``batch_size`` runs the underlying
``vmap`` in bounded batches, keeping the peak memory independent of the total number
of trajectories.

This is a counterfactual reconstruction: the sweep built its controllers at a
placeholder start state before overwriting ``car.state``, so the sampled batch is not
the one the trial actually used.  The rollout propagates the disturbance through the
dynamics RHS rather than adding it after the Euler step; the two agree to O(dt^2) per
step.
"""

import argparse
import gc
import os

import _env  # sets up sys.path for buzzracer + root helpers; canonical dirs
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from _config import build_xml_config
from _trial_utils import trial_seed_from_combo
from buzzracer.cars.car import Car
from buzzracer.extensions.extension import Extension
from buzzracer.scripts.run import Main
from immrax_planner import _get_control_index

SCALES = np.linspace(1.0, 10.0, 100)

# Trajectories evaluated per parallel batch.  ``jax.lax.map`` runs its ``vmap`` in
# batches of this size, so the boundary check's (batch, horizon, raceline points)
# scratch dominates peak memory.  At 4096 that is roughly 0.5 GB per scratch array,
# well under the ~8 GB budget.
TRAJECTORY_BATCH = 4096

CASE_COLUMNS = [
    "run_id",
    "init_point_index",
    "init_offset_name",
    "centering_weight_factor",
    "velocity_scale_factor",
    "state_noise_direction",
    "prng_seed",
    "init_x",
    "init_y",
    "init_heading",
    "init_v_forward",
    "unsafe_fraction",
    "total_control_steps",
    "unsafe_control_steps",
    "stop_reason",
]

RESULT_COLUMNS = [
    *CASE_COLUMNS,
    "n_controls",
    "n_scales",
    "controls_collided_at_scale_1",
    "controls_collided_at_max_scale",
    "controls_never_collided",
    "max_collision_fraction",
    "max_scale_factor",
    "fully_unsafe_at_scale_1",
    "min_scale_factor",
    "min_scale_index",
]


# ---------------------------------------------------------------------------
# Trial selection
# ---------------------------------------------------------------------------


def load_crashes(runs_csv):
    """Reachset-enabled crash rows, split by ``unsafe_fraction``.

    Returns ``(partially_unsafe, fully_unsafe)`` where the latter holds rows the
    planner flagged unsafe at every control step.  The split is computed from the
    current analysis output rather than a fixed list of trials.
    """
    df = pd.read_csv(runs_csv)
    crashes = df[(df["use_reachsets"] == True) & (df["outcome"] == "crash")]
    fully_unsafe = crashes[np.isclose(crashes["unsafe_fraction"], 1.0)]
    partially_unsafe = crashes[~np.isclose(crashes["unsafe_fraction"], 1.0)]
    return partially_unsafe.reset_index(drop=True), fully_unsafe.reset_index(drop=True)


def trial_seed(row):
    """JAX control-sampling seed the trial used (``run_id`` is the DVC combo key)."""
    return trial_seed_from_combo(int(row["prng_seed"]), row["run_id"])


# ---------------------------------------------------------------------------
# Trial reconstruction
# ---------------------------------------------------------------------------


def build_config(row, sim, seed):
    """Temp XML for the trial, with the controller built at the trial's start state."""
    overrides = dict(sim)
    overrides.update(
        use_reachsets=True,
        prng_seed=seed,
        centering_weight_factor=float(row["centering_weight_factor"]),
        velocity_scale_factor=float(row["velocity_scale_factor"]),
        init_states=(
            float(row["init_x"]),
            float(row["init_y"]),
            float(row["init_heading"]),
            float(row["init_v_forward"]),
        ),
    )
    return build_xml_config(overrides)


def sample_control_batch(controller):
    """One planner control batch, shape ``(num_samples, horizon, 2)``.

    Channels are ``[steering, throttle]``.  ``sample_controls`` is pure, so the
    planner's PRNG key advances only on the next ``step()``.
    """
    planner = controller.planner

    # Run under jit: the controller leaves the host-to-device transfer guard
    # disallowed, and the sampler indexes with Python constants.
    @jax.jit
    def sample(planned_dt, planned, key):
        return planner.sample_controls(planned_dt, planned, key)

    _, controls, _ = sample(
        planner.planned_controls_dt, planner.planned_controls, planner.prng_key
    )
    return np.asarray(jax.device_get(controls))


def state_noise_bounds(simulator, car, controller):
    """Per-channel half-width of the simulator's lateral noise box."""
    direction = simulator.state_noise_direction
    simulator.state_noise_direction = None  # legacy constant-magnitude branch
    try:
        bounds = np.asarray(simulator._compute_lateral_w(car), dtype=float)
    finally:
        simulator.state_noise_direction = direction

    declared = np.asarray(controller.planner.disturbance_spec.half_widths, dtype=float)
    assert np.allclose(bounds, declared), (
        f"simulator noise bounds {bounds} disagree with the planner's "
        f"disturbance box {declared}"
    )
    return bounds


def collect_case(row, sim):
    """Reconstruct one crash's start state and sampled control batch."""
    seed = trial_seed(row)
    Extension.extensions = []
    Car.reset()
    tmp_config = build_config(row, sim, seed)
    experiment = None
    try:
        experiment = Main(tmp_config)
        car = experiment.cars[0]
        controller = car.controller
        planner = controller.planner
        assert float(planner.planning_dt) == float(experiment.dt), (
            "planner and simulator timesteps differ"
        )

        return {
            "record": {column: row[column] for column in CASE_COLUMNS},
            "x0": np.asarray(car.state, dtype=np.float32)[: planner.state_dim],
            "controls": sample_control_batch(controller).astype(np.float32),
            "bounds": state_noise_bounds(experiment.simulator, car, controller),
            "ref": raceline_arrays(experiment.track),
            "predictor": planner.predictor,
            "planning_dt": float(planner.planning_dt),
            "horizon": int(planner.planning_horizon),
            "t_start": planner._t_start,
            "t_end": planner._t_end,
        }
    finally:
        try:
            os.unlink(tmp_config)
        except OSError:
            pass
        experiment = None
        car = None
        controller = None
        gc.collect()
        jax.clear_caches()


def check_compatible(cases):
    """Assert every case shares the track, disturbance box, and car dynamics."""
    first = cases[0]
    first_ref = {key: jax.device_get(value) for key, value in first["ref"].items()}
    for case in cases[1:]:
        assert case["controls"].shape == first["controls"].shape, (
            "control batch shapes differ across crashes"
        )
        assert case["horizon"] == first["horizon"]
        assert case["planning_dt"] == first["planning_dt"]
        assert np.allclose(case["bounds"], first["bounds"])
        for key, expected in first_ref.items():
            assert np.array_equal(jax.device_get(case["ref"][key]), expected), (
                f"raceline array {key!r} differs across crashes"
            )
        for attr in ("lf", "lr", "L", "Iz", "m"):
            assert np.allclose(
                getattr(case["predictor"], attr), getattr(first["predictor"], attr)
            ), f"car parameter {attr!r} differs across crashes"


# ---------------------------------------------------------------------------
# Rollouts and boundary check
# ---------------------------------------------------------------------------


def raceline_arrays(track):
    """Discretized raceline columns used by ``BoundaryChecker``."""
    ref = np.asarray(track.discretized_raceline)
    return {
        "x": jnp.asarray(ref[:, 0]),
        "y": jnp.asarray(ref[:, 1]),
        "heading": jnp.asarray(ref[:, 2]),
        "left": jnp.asarray(ref[:, 4]),
        "right": jnp.asarray(ref[:, 5]),
    }


def away_disturbance(ref, bounds, state):
    """``_away_from_raceline_direction`` applied to one state, times the box width."""
    dx_ref = ref["x"][None, :] - state[0]
    dy_ref = ref["y"][None, :] - state[1]
    idx = jnp.argmin(dx_ref**2 + dy_ref**2, axis=1)

    heading = ref["heading"][idx]
    dx = state[0] - ref["x"][idx]
    dy = state[1] - ref["y"][idx]
    lateral_err = jnp.cos(heading) * dy - jnp.sin(heading) * dx

    direction = jnp.where(jnp.abs(lateral_err) > 1e-3, jnp.sign(lateral_err), 1.0)
    # ``bounds`` is a rollout operand, so it is already a device array; no
    # ``jnp.asarray`` here or it would become a fresh lowering-time constant.
    return direction * bounds


def is_out_of_boundary(ref, xy):
    """Vectorized ``BoundaryChecker.is_out_of_boundary_discrete`` over points."""
    dx_ref = ref["x"][None, :] - xy[:, 0:1]
    dy_ref = ref["y"][None, :] - xy[:, 1:2]
    idx = jnp.argmin(dx_ref**2 + dy_ref**2, axis=1)

    dx = ref["x"][idx] - xy[:, 0]
    dy = ref["y"][idx] - xy[:, 1]
    dist = jnp.sqrt(dx**2 + dy**2)
    angle = jnp.arctan2(dy, dx)
    heading_diff = (angle - ref["heading"][idx] + jnp.pi) % (2 * jnp.pi) - jnp.pi

    return jnp.where(
        heading_diff > 0, dist > ref["right"][idx], dist > ref["left"][idx]
    )


def make_grid_rollout(case, batch_size):
    """Rollout every (crash, control sequence, scale) in one ``lax.map`` call.

    ``x0s`` is ``(B, state_dim)``, ``controls`` is ``(B, horizon, 2)``, and
    ``scale_grid`` is ``(B,)``, where ``B = C * N * S``.  ``ref``, ``bounds``,
    ``t_start``, and ``t_end`` are jit operands rather than closed-over
    constants, so they stay device-resident instead of being copied back to the
    host at lowering.  Returns ``(B,)`` whether each control sequence collides
    under the adversarial disturbance at its scale.  ``batch_size`` caps the
    batched ``vmap`` that ``lax.map`` runs internally.
    """
    predictor = case["predictor"]
    dt = case["planning_dt"]
    horizon = case["horizon"]

    @jax.jit
    def rollout_all(x0s, controls, scale_grid, ref, bounds, t_start, t_end):
        def one_trajectory(inputs):
            x0, control, scale = inputs

            def u(t, x):
                return control[_get_control_index(t, dt, horizon)]

            def d(t, x):
                return scale * away_disturbance(ref, bounds, x)

            traj = predictor.compute_trajectory(
                t_start,
                t_end,
                x0,
                (u, d),
                dt=dt,
                solver="euler",
            )
            # diffrax pads the saved output to its fixed max_steps; only the
            # first horizon + 1 entries are real.  Skip the start point,
            # matching the planner's predicted-step slice.
            ys = traj.ys[: horizon + 1]
            return jnp.any(is_out_of_boundary(ref, ys[1:, :2]))

        return jax.lax.map(
            one_trajectory, (x0s, controls, scale_grid), batch_size=batch_size
        )

    return rollout_all


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def inspect_crashes(rows, sim, scales=SCALES, batch_size=TRAJECTORY_BATCH):
    """Reconstruct each crash and evaluate the scaled adversarial criterion."""
    cases = [collect_case(row, sim) for _, row in rows.iterrows()]
    check_compatible(cases)

    n_cases = len(cases)
    n_controls, horizon = cases[0]["controls"].shape[:2]
    state_dim = cases[0]["x0"].shape[0]
    scales = np.asarray(scales)
    n_scales = len(scales)

    # Flatten (crash, control, scale) into one leading axis for lax.map.  The
    # broadcast grids are the only full-size host arrays (~0.7 GB at 26 x 1024 x
    # 100); the per-batch rollout scratch stays bounded by TRAJECTORY_BATCH.
    x0s = np.stack([case["x0"] for case in cases])
    controls = np.stack([case["controls"] for case in cases])
    x0_grid = np.broadcast_to(
        x0s[:, None, None, :], (n_cases, n_controls, n_scales, state_dim)
    ).reshape(-1, state_dim)
    control_grid = np.broadcast_to(
        controls[:, :, None, :, :], (n_cases, n_controls, n_scales, horizon, 2)
    ).reshape(-1, horizon, 2)
    scale_grid = np.broadcast_to(
        scales[None, None, :], (n_cases, n_controls, n_scales)
    ).reshape(-1)

    rollout = make_grid_rollout(cases[0], batch_size)
    # Lowering embeds device-constant literals created inside diffrax and the
    # interval comparison in the dynamics; their host copies cannot be requested
    # explicitly from here.  Only compilation transfers -- the rollout itself
    # moves no data -- so allow transfers for this call and restore the guard
    # set by the controller afterwards.
    with jax.transfer_guard("allow"):
        collisions = rollout(
            jnp.asarray(x0_grid),
            jnp.asarray(control_grid),
            jnp.asarray(scale_grid),
            cases[0]["ref"],
            jnp.asarray(cases[0]["bounds"], dtype=jnp.float32),
            cases[0]["t_start"],
            cases[0]["t_end"],
        )
    # (C, N, S) -> (C, S): a crash is fully unsafe at a scale iff every control
    # sequence collides there.
    collisions = np.asarray(jax.device_get(collisions)).reshape(
        n_cases, n_controls, n_scales
    )
    all_controls_collide = collisions.all(axis=1)

    records = []
    for case, case_collisions, case_all in zip(cases, collisions, all_controls_collide):
        if case_all.any():
            min_index = int(np.argmax(case_all))
            min_scale = float(scales[min_index])
        else:
            min_index = -1
            min_scale = float("nan")

        record = dict(case["record"])
        record.update(
            n_controls=int(case_collisions.shape[0]),
            n_scales=int(case_collisions.shape[1]),
            controls_collided_at_scale_1=int(case_collisions[:, 0].sum()),
            controls_collided_at_max_scale=int(case_collisions[:, -1].sum()),
            controls_never_collided=int((~case_collisions.any(axis=1)).sum()),
            max_collision_fraction=float(case_collisions.mean(axis=0).max()),
            max_scale_factor=float(scales[-1]),
            fully_unsafe_at_scale_1=bool(case_all[0]),
            min_scale_factor=min_scale,
            min_scale_index=min_index,
        )
        records.append(record)

    return pd.DataFrame(records, columns=RESULT_COLUMNS)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-csv",
        default=os.path.join(_env.analysis_dir("reachset_sweep"), "runs.csv"),
        help="Per-run reachset_sweep dataframe (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=_env.analysis_dir("reachset_sweep", "crashes"),
        help="Directory for analysis output (default: %(default)s)",
    )
    parser.add_argument(
        "--trajectory-batch",
        type=int,
        default=TRAJECTORY_BATCH,
        help="Trajectories per parallel batch; bounds JAX's rollout memory "
        "(default: %(default)s)",
    )
    return parser.parse_args(argv)


def load_sim(params_path):
    with open(params_path) as f:
        return yaml.safe_load(f)["sim"]


def main(argv=None):
    args = parse_args(argv)
    sim = load_sim(os.path.join(_env.PROJECT_ROOT, "params.yaml"))
    partially_unsafe, fully_unsafe = load_crashes(args.runs_csv)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reachset crashes with unsafe_fraction < 1 ({len(partially_unsafe)}):")
    if partially_unsafe.empty:
        print("None.")
    else:
        print(partially_unsafe[CASE_COLUMNS].to_string(index=False))
    partial_path = os.path.join(args.output_dir, "smaller_unsafe_fraction.csv")
    partially_unsafe.reindex(columns=CASE_COLUMNS).to_csv(partial_path, index=False)
    print(f"Wrote {partial_path}")

    print(
        f"\nInspecting {len(fully_unsafe)} fully-unsafe crashes "
        f"(unsafe_fraction == 1) under {len(SCALES)} adversarial scales "
        f"[{SCALES[0]:g}, {SCALES[-1]:g}] ..."
    )
    results = inspect_crashes(fully_unsafe, sim, batch_size=args.trajectory_batch)
    result_path = os.path.join(args.output_dir, "fully_unsafe_inspection.csv")
    results.to_csv(result_path, index=False)
    print(f"Wrote {result_path}")

    at_scale_1 = results[results["fully_unsafe_at_scale_1"]]
    print(
        f"\nFully unsafe at scale 1 (all control samples collide): "
        f"{len(at_scale_1)} of {len(results)}"
    )

    print("\nPer-crash minimum adversarial scale factor:")
    print(
        results[
            [
                "run_id",
                "n_controls",
                "controls_collided_at_scale_1",
                "controls_collided_at_max_scale",
                "controls_never_collided",
                "min_scale_factor",
            ]
        ].to_string(index=False)
    )

    missing = results[~np.isfinite(results["min_scale_factor"])]
    if len(missing):
        print(
            f"\nNo scale in [{SCALES[0]:g}, {SCALES[-1]:g}] makes every control "
            f"sample collide for {len(missing)} crash(es):"
        )
        print(missing["run_id"].to_string(index=False))
        print("No single factor makes every crash fully unsafe.")
    else:
        global_factor = float(results["min_scale_factor"].max())
        print(f"\nSmallest factor making every crash fully unsafe: {global_factor:.6g}")


if __name__ == "__main__":
    main()
