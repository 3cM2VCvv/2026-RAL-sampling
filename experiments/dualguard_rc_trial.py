"""Recreate DualGuard-MPPI's RC-car experiment (arXiv 2502.01924, Section VI)
with this repo's reachset-aware planner.

A 3-state Dubins car (``[x, y, heading]``, control ``[V, delta]``, eq. 15) is driven
around the paper's "RC warehouse" track for several laps by ``ImmraxPlanner`` using the
eq.-16 cost.  The track is reconstructed from the paper's own HJB artifact by the
``reconstruct_track`` stage, so the geometry is the paper's, not an approximation of it.
The plant is integrated from the same Dubins dynamics, with the additive position-rate
disturbance ``d_x, d_y ~ U[-0.1, 0.1]`` of eq. 15 (model mismatch / state-estimation
error) drawn from the same ``Disturbance`` object the planner over-approximates.

Safety = staying inside the track boundaries.  Unlike DualGuard there is no HJ BRT and
no least-restrictive filter: safety comes from the reachset over-approximation entering
the cost, so an unsafe rollout is one the planner *predicts* will collide.

Two axes are swept (see ``rc_sweep`` in params.yaml):
  * ``--use_reachsets``  reachset (interval) rollouts vs. point trajectories.
  * ``--cost_type``      how eq. 16 measures "distance from the track center":
       ``lx``      - the paper's l(x) distance-to-obstacle field, read off the grid.
       ``frenet``  - lateral error against the min-curvature racing line.

Metrics logged (DVCLive), matching the paper's Table III where applicable:
  * CompTime (ms)  - per-cycle planner compute time (mean / std / median).
  * Speed (m/s)    - average speed per lap over the completed laps.
  * total cost     - accumulated eq.-16 running cost of the executed trajectory
                     (absolute; no normalized RelCost -- we have no same-track
                     DualGuard reference run to normalize against).
  * safety         - planner unsafe fraction (a rollout was predicted to collide)
                     and count/fraction of executed boundary exits.

Usage (single run):
    uv run python experiments/dualguard_rc_trial.py \
        --out-dir runs/dualguard_rc/manual --use_reachsets --cost_type lx --lap_count 3
"""

import argparse
import gc
import math
import os

import _env  # noqa: F401  sets up sys.path for buzzracer + root helpers

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import splev
from dvclive import Live

from disturbance import Disturbance
from immrax_dubins_car import DubinsCar, DUALGUARD_RC_CAR
from immrax_planner import ImmraxPlanner, SampleBounds
from reachset_cost import (
    DubinsRacingCost,
    LxDubinsRacingCost,
    PointDubinsRacingCost,
    PointLxDubinsRacingCost,
)
from warehouse_track import WarehouseTrack

# eq.-16 cost, indexed by (cost_type, use_reachsets).
COST_CLASSES = {
    ("lx", True): LxDubinsRacingCost,
    ("lx", False): PointLxDubinsRacingCost,
    ("frenet", True): DubinsRacingCost,
    ("frenet", False): PointDubinsRacingCost,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", dest="out_dir", required=True,
                   help="DVCLive directory for this trial's params/metrics")
    # Vehicle (eq. 15) -- defaults are the paper's RC car.
    p.add_argument("--L", type=float, default=DUALGUARD_RC_CAR.L, help="wheelbase (m)")
    p.add_argument("--v_max", type=float, default=DUALGUARD_RC_CAR.v_max)
    p.add_argument("--v_min", type=float, default=DUALGUARD_RC_CAR.v_min)
    p.add_argument("--delta_max_deg", type=float,
                   default=math.degrees(DUALGUARD_RC_CAR.delta_max))
    p.add_argument("--disturbance", type=float, default=0.1,
                   help="half-width of d_x, d_y ~ U[-d, d]")
    # Cost (eq. 16)
    p.add_argument("--cost_type", choices=("lx", "frenet"), default="lx",
                   help="how to measure distance from the track center: the paper's "
                        "l(x) field, or Frenet error against the racing line")
    p.add_argument("--centering_weight_factor", type=float, default=120.0, help="K_c")
    p.add_argument("--collision_weight", type=float, default=50.0,
                   help="graded P(x) penalty per colliding step (lx cost only)")
    p.add_argument("--progress_weight", type=float, default=0.0,
                   help="reward on lap progress; augments eq.16 to counter the "
                        "myopic (BRT-free) reachset safety term. 0 = strict eq.16.")
    # Planner (MPPI) settings -- defaults match Table III (1000 rollouts x 100 steps @ 50Hz)
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--num_samples", type=int, default=1000)
    p.add_argument("--planning_dt", type=float, default=0.02)
    p.add_argument("--use_reachsets", action="store_true")
    # Sampling std for the control time-derivatives (per channel). Value clipping is
    # handled by the value bounds; these only set exploration breadth. They are the
    # closest analogue to the reference implementation's control-value noise [0.2, 0.4].
    p.add_argument("--v_dt_std", type=float, default=5.0)
    p.add_argument("--delta_dt_std", type=float, default=3.0 / 4.0 * float(np.pi))
    # Simulation
    p.add_argument("--lap_count", type=int, default=3)
    p.add_argument("--wall_clock", type=float, default=120.0,
                   help="max simulated seconds before giving up")
    p.add_argument("--stall_timeout", type=float, default=10.0,
                   help="stop if no forward lap-progress for this many sim seconds")
    p.add_argument("--substeps", type=int, default=4, help="plant euler substeps/cycle")
    p.add_argument("--crash_stop", action="store_true",
                   help="stop the run on the first boundary exit")
    p.add_argument("--prng_seed", type=int, default=42)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Track helpers (numpy)
# ---------------------------------------------------------------------------


class FrenetProjector:
    """Numpy nearest-raceline projection for the executed (plant) trajectory.

    Supplies lap progress (arc length) for every run, plus the lateral error the
    ``frenet`` cost arm is scored on.  Boundary exits are *not* judged here -- that is
    ``WarehouseTrack.is_outside``, off the true l(x) field.
    """

    def __init__(self, track):
        self.rx = np.asarray(track.raceline_points[0])
        self.ry = np.asarray(track.raceline_points[1])
        self.rh = np.asarray(track.raceline_headings)
        self.ss = np.asarray(track.ss)
        self.track_len = float(track.raceline_len_m)

    def project(self, x, y):
        """Return (progress_s, signed_lateral_err, closest_idx)."""
        idx = int(np.argmin((self.rx - x) ** 2 + (self.ry - y) ** 2))
        heading = self.rh[idx]
        dx = x - self.rx[idx]
        dy = y - self.ry[idx]
        lateral = -dx * np.sin(heading) + dy * np.cos(heading)
        return self.ss[idx], lateral, idx


def initial_state(track):
    """Place the car on the raceline at s=0, aligned with the raceline heading."""
    rr = splev(0.0, track.raceline_s, der=0)
    drr = splev(0.0, track.raceline_s, der=1)
    heading = float(np.arctan2(drr[1], drr[0]))
    return np.array([float(rr[0]), float(rr[1]), heading])


# ---------------------------------------------------------------------------
# Plant (eq. 15 with additive disturbance)
# ---------------------------------------------------------------------------


def plant_step(state, control, disturbance, L, dt, substeps):
    x, y, theta = state
    V, delta = control
    d_x, d_y = disturbance
    h = dt / substeps
    for _ in range(substeps):
        x += h * (V * np.cos(theta) + d_x)
        y += h * (V * np.sin(theta) + d_y)
        theta += h * (V * np.tan(delta) / L)
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    return np.array([x, y, theta])


# ---------------------------------------------------------------------------
# Trial
# ---------------------------------------------------------------------------


def build_planner(args, track, disturbance):
    predictor = DubinsCar(L=args.L)
    delta_max = np.radians(args.delta_max_deg)

    control_bounds = [
        SampleBounds(min=args.v_min, std=(args.v_max - args.v_min) / 2, max=args.v_max),
        SampleBounds(min=-delta_max, std=delta_max / 2, max=delta_max),
    ]
    control_dt_bounds = [
        SampleBounds(min=-100.0, std=args.v_dt_std, max=100.0),
        SampleBounds(min=-1.5 * float(np.pi), std=args.delta_dt_std, max=1.5 * float(np.pi)),
    ]

    cost_kwargs = dict(
        v_max=args.v_max,
        centering_weight_factor=args.centering_weight_factor,
        terminal_progress_weight=args.progress_weight,
    )
    if args.cost_type == "lx":
        cost_kwargs["collision_weight"] = args.collision_weight
    rollout_cost = COST_CLASSES[(args.cost_type, args.use_reachsets)](
        track, args.horizon, args.planning_dt, **cost_kwargs
    )

    # Warm start: drive straight at a nominal speed; the sampler explores steering.
    v_nominal = 0.5 * (args.v_min + args.v_max)
    initial_controls = jnp.tile(
        jnp.array([v_nominal, 0.0]), (args.horizon, 1)
    )

    planner = ImmraxPlanner(
        predictor,
        control_bounds=control_bounds,
        control_dt_bounds=control_dt_bounds,
        rollout_cost=rollout_cost,
        disturbance=disturbance,
        initial_controls=initial_controls,
        num_samples=args.num_samples,
        planning_horizon=args.horizon,
        planning_dt=args.planning_dt,
        prng_seed=args.prng_seed,
        use_reachsets=args.use_reachsets,
    )
    return planner


def run(args, live):
    # ------------------------------------------------------------------
    # 1. Track + planner + plant setup
    # ------------------------------------------------------------------
    track = WarehouseTrack()
    proj = FrenetProjector(track)

    for k in ("v_max", "v_min", "delta_max_deg", "disturbance", "cost_type",
              "centering_weight_factor", "collision_weight", "progress_weight",
              "horizon", "num_samples", "planning_dt", "use_reachsets",
              "lap_count", "prng_seed", "L"):
        live.log_param(k, getattr(args, k))
    # The corridor buffer is baked into the reconstructed track, not passed in here --
    # record it so runs built against different geometry are distinguishable.
    live.log_param("buffer", track.buffer)

    # One disturbance object, over-approximated by the planner and sampled by the plant,
    # so the two cannot disagree about what the car is subjected to.
    disturbance = Disturbance((args.disturbance, args.disturbance))
    planner = build_planner(args, track, disturbance)

    state = initial_state(track)
    live.log_metric("init_x", float(state[0]))
    live.log_metric("init_y", float(state[1]))
    live.log_metric("init_heading", float(state[2]))

    # JIT warmup (excluded from CompTime statistics).
    planner.warmup(state)

    rng = np.random.default_rng(args.prng_seed)
    delta_max = np.radians(args.delta_max_deg)

    # "Distance from the track center", scored on the executed trajectory the same way
    # the planner's chosen cost scored it during rollouts.
    if args.cost_type == "lx":
        def center_distance(state, lateral):
            return track.l_center - track.lx_at(state[0], state[1])
    else:
        def center_distance(state, lateral):
            return abs(lateral)

    # ------------------------------------------------------------------
    # 2. Receding-horizon loop
    # ------------------------------------------------------------------
    dt = args.planning_dt
    max_steps = int(args.wall_clock / dt)

    comptimes_ms = []
    total_cost = 0.0
    cost_velocity = 0.0
    cost_centerline = 0.0
    boundary_exit_steps = 0

    # Lap / progress tracking.
    prev_s, _, _ = proj.project(state[0], state[1])
    cumulative_progress = 0.0
    lap_boundary_times = [0.0]  # sim-time at each completed lap
    laps_completed = 0
    best_progress = 0.0
    last_progress_time = 0.0

    crashed = False
    stop_reason = "wall_clock"
    step = 0
    for step in range(max_steps):
        control, info = planner.step(state)
        comptimes_ms.append(info["compute_time"] * 1000.0)

        V = float(control[0])
        delta = float(np.clip(control[1], -delta_max, delta_max))
        if not np.isfinite(V) or not np.isfinite(delta):
            stop_reason = "invalid_control"
            break

        # Disturbance drawn once per control cycle, held over the cycle.
        d = disturbance.sample(rng)

        # Executed-trajectory cost (eq. 16 terms) at the applied state/control.  The
        # centering term is scored the same way the planner's cost measured it, so the
        # two cost_type arms are each self-consistent (but not comparable to each other).
        s, lateral, _ = proj.project(state[0], state[1])
        vel_term = (args.v_max - V) ** 2
        center_term = args.centering_weight_factor * center_distance(state, lateral)
        cost_velocity += vel_term
        cost_centerline += center_term
        total_cost += vel_term + center_term
        # Ground truth, straight off the l(x) field: the same verdict for both cost_type
        # arms, so neither is graded by its own notion of where the track is.
        if track.is_outside(state[:2]):
            boundary_exit_steps += 1
            if not crashed:
                crashed = True
            if args.crash_stop:
                stop_reason = "crash"
                state = plant_step(state, (V, delta), d, args.L, dt, args.substeps)
                break

        # Advance the plant.
        state = plant_step(state, (V, delta), d, args.L, dt, args.substeps)

        # Lap detection via cumulative arc-length progress.
        new_s, _, _ = proj.project(state[0], state[1])
        ds = new_s - prev_s
        if ds < -proj.track_len / 2:
            ds += proj.track_len
        elif ds > proj.track_len / 2:
            ds -= proj.track_len
        cumulative_progress += ds
        prev_s = new_s
        while cumulative_progress >= (laps_completed + 1) * proj.track_len:
            laps_completed += 1
            lap_boundary_times.append((step + 1) * dt)
        if laps_completed >= args.lap_count:
            stop_reason = "laps_complete"
            break

        # Stall detection: give up if no forward progress for a while.
        now = (step + 1) * dt
        if cumulative_progress > best_progress + 1e-3:
            best_progress = cumulative_progress
            last_progress_time = now
        elif now - last_progress_time > args.stall_timeout:
            stop_reason = "stall"
            break

    sim_duration = (step + 1) * dt

    # ------------------------------------------------------------------
    # 3. Metrics
    # ------------------------------------------------------------------
    comptimes = np.asarray(comptimes_ms)
    live.log_metric("comptime_ms_mean", float(np.mean(comptimes)))
    live.log_metric("comptime_ms_std", float(np.std(comptimes)))
    live.log_metric("comptime_ms_median", float(np.median(comptimes)))

    lap_times = np.diff(np.asarray(lap_boundary_times))
    if len(lap_times) > 0:
        lap_speeds = proj.track_len / lap_times
        live.log_metric("speed_mean", float(np.mean(lap_speeds)))
        live.log_metric("speed_std", float(np.std(lap_speeds)))
    else:
        live.log_metric("speed_mean", 0.0)
        live.log_metric("speed_std", 0.0)

    live.log_metric("num_laps_complete", laps_completed)
    live.log_metric("total_cost", float(total_cost))
    live.log_metric("cost_velocity", float(cost_velocity))
    live.log_metric("cost_centerline", float(cost_centerline))
    live.log_metric("mean_cost_per_step", float(total_cost / max(step + 1, 1)))
    live.log_metric("unsafe_fraction", float(planner.unsafe_fraction))
    live.log_metric("total_control_steps", int(planner._total_control_steps))
    live.log_metric("boundary_exit_steps", int(boundary_exit_steps))
    live.log_metric("boundary_exit_fraction", float(boundary_exit_steps / max(step + 1, 1)))
    live.log_metric("crashed", int(crashed))
    live.log_metric("distance_progressed", float(cumulative_progress))
    live.log_metric("sim_duration", float(sim_duration))
    live.summary["stop_reason"] = stop_reason


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    with Live(dir=args.out_dir, save_dvc_exp=False, dvcyaml=False, report=None) as live:
        try:
            run(args, live)
        finally:
            gc.collect()
            jax.clear_caches()


if __name__ == "__main__":
    main()
