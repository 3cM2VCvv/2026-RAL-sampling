"""Single reachset_sweep trial, instrumented with DVCLive.

Runs one Buzzracer simulation with a parameterized initial state, cost weights,
disturbance type, and reachset mode.  Collects safety and performance metrics
and records them to a DVCLive directory (``--out-dir``).

This is the unit stage of the ``reachset_trial`` matrix in ``dvc.yaml``; the
matrix expands the ``sweep`` axes in ``params.yaml`` into one invocation per
combination.  The swept axes and the fixed ``sim`` params are passed as CLI
flags (DVC's ``${item}`` / ``${sim}`` interpolation); booleans render as
store_true flags.

Usage (single run):
    uv run python experiments/reachset_sweep.py \
        --init_point_index 0 --init_offset_name center --use_reachsets \
        --centering_weight_factor 1.0 --velocity_scale_factor 1.0 \
        --state_noise_direction uniform --prng_seed 42 \
        --out-dir runs/reachset_sweep/manual
"""

import argparse
import gc
import os
import zlib

import jax

import _env  # sets up sys.path for buzzracer + root helpers; canonical dirs
from _buzzracer_config import write_temp_config

import numpy as np
from scipy.interpolate import splev
from dvclive import Live

from buzzracer.cars.car import Car
from buzzracer.common import BASEDIR
from buzzracer.extensions.extension import Extension
from buzzracer.scripts.run import Main

BASE_XML = os.path.join(BASEDIR, "buzzracer", "configs", "reachset_sweep_base.xml")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    # Swept axes (dvc.yaml ${item})
    p.add_argument("--init_point_index", type=int, default=0,
                   help="0-9: which of 10 evenly-spaced raceline points")
    p.add_argument("--init_offset_name", default="center",
                   choices=["center", "inside", "outside"])
    p.add_argument("--centering_weight_factor", type=float, default=1.0)
    p.add_argument("--velocity_scale_factor", type=float, default=1.0)
    p.add_argument("--state_noise_direction", default="uniform",
                   choices=["uniform", "adversarial_away"])
    p.add_argument("--use_reachsets", action="store_true")
    # Fixed sim params (dvc.yaml ${sim})
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--state_noise_enabled", action="store_true")
    p.add_argument("--state_noise_magnitude", type=float, default=0.1)
    p.add_argument("--state_noise_type", default="lateral_only")
    p.add_argument("--wall_clock", type=float, default=60.0)
    p.add_argument("--crash_stop", action="store_true")
    p.add_argument("--stall_timeout", type=float, default=10.0)
    p.add_argument("--lap_count", type=int, default=1)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--max_speed", type=float, default=2.0)
    p.add_argument("--prng_seed", type=int, default=42,
                   help="base seed; combined with the trial combo into a "
                        "deterministic per-trial seed")
    # Output
    p.add_argument("--out-dir", dest="out_dir", required=True,
                   help="DVCLive directory for this trial's params/metrics")
    return p.parse_args(argv)


def trial_seed(base_seed, combo_str):
    """Deterministic per-trial seed from a base seed + the combo identifier.

    Uses crc32 (stable across processes, unlike ``hash``) so the same trial
    always draws the same disturbance sequence -> DVC caching is meaningful,
    while distinct trials still get distinct seeds.
    """
    return (int(base_seed) * 1_000_003 + zlib.crc32(combo_str.encode())) % (2**31)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_initial_state(track, point_index, offset_name):
    """Compute (x, y, heading, v_forward) for a raceline point with offset.

    Parameters
    ----------
    track : Track
        Loaded track object with raceline spline.
    point_index : int
        Index 0-9 selecting one of 10 evenly-spaced arc-length points.
    offset_name : str
        "center" (on raceline), "inside" (-0.05 m perpendicular),
        or "outside" (+0.05 m perpendicular).

    Returns
    -------
    tuple of (x, y, heading, v_forward)
    """
    track_len = track.raceline_len_m
    s = (point_index / 10) * track_len

    rr = splev(s % track_len, track.raceline_s, der=0)
    drr = splev(s % track_len, track.raceline_s, der=1)
    heading = float(np.arctan2(drr[1], drr[0]))
    v_ref = float(track.sToV(np.array([s]))[0])

    # Perpendicular normal: left of heading direction
    normal = np.array([-np.sin(heading), np.cos(heading)])

    offsets = {"center": 0.0, "inside": -0.05, "outside": 0.05}
    d = offsets[offset_name]

    x = float(rr[0] + d * normal[0])
    y = float(rr[1] + d * normal[1])

    return x, y, heading, v_ref


def build_xml_config(overrides):
    """Create a temporary XML config file with the given overrides applied.

    Returns the path to the temporary file (caller must delete it).
    """
    def mutate(config):
        # --- Settings ---
        settings_nodes = config.getElementsByTagName("settings")
        if settings_nodes and "dt" in overrides:
            settings_nodes[0].setAttribute("dt", str(overrides["dt"]))

        # --- Extensions (simulator + auto_stopper) ---
        for ext_node in config.getElementsByTagName("extension"):
            handle = ext_node.getAttribute("handle")
            if handle == "simulator":
                for key in (
                    "state_noise_direction",
                    "state_noise_enabled",
                    "state_noise_magnitude",
                    "state_noise_type",
                ):
                    if key in overrides:
                        ext_node.setAttribute(key, str(overrides[key]))
            if handle == "auto_stopper":
                for key in (
                    "wall_clock",
                    "crash_stop",
                    "stall_timeout",
                    "lap_count",
                ):
                    if key in overrides:
                        ext_node.setAttribute(key, str(overrides[key]))

        # --- Controller attributes ---
        ctrl_node = config.getElementsByTagName("controller")[0]
        for key in (
            "use_reachsets",
            "prng_seed",
            "centering_weight_factor",
            "velocity_scale_factor",
            "horizon",
            "max_speed",
        ):
            if key in overrides:
                ctrl_node.setAttribute(key, str(overrides[key]))

        # --- Initial state ---
        car_node = config.getElementsByTagName("car")[0]
        init_nodes = car_node.getElementsByTagName("init_states")
        init_text = repr(overrides["init_states"])
        if init_nodes:
            init_nodes[0].firstChild.data = init_text
        else:
            elem = config.createElement("init_states")
            elem.appendChild(config.createTextNode(init_text))
            car_node.appendChild(elem)

    return write_temp_config(BASE_XML, mutate)


# ---------------------------------------------------------------------------
# Main trial
# ---------------------------------------------------------------------------


def run(args, live):
    # ------------------------------------------------------------------
    # 1. Bootstrap: load track to compute the real initial state
    # ------------------------------------------------------------------
    # We need the track geometry to translate (point_index, offset_name)
    # into Cartesian coordinates.  Build Main with a placeholder state,
    # read the track, then override car.state before running.
    # ------------------------------------------------------------------
    Extension.extensions = []
    Car.reset()

    # Deterministic per-trial seed derived from the combo identifier.
    combo = (
        f"{args.init_point_index}-{args.init_offset_name}-"
        f"{args.centering_weight_factor}-{args.velocity_scale_factor}-"
        f"{args.state_noise_direction}-{args.use_reachsets}"
    )
    seed = trial_seed(args.prng_seed, combo)

    # Log config params (inputs).
    live.log_param("init_point_index", args.init_point_index)
    live.log_param("init_offset_name", args.init_offset_name)
    live.log_param("centering_weight_factor", args.centering_weight_factor)
    live.log_param("velocity_scale_factor", args.velocity_scale_factor)
    live.log_param("state_noise_direction", args.state_noise_direction)
    live.log_param("use_reachsets", args.use_reachsets)
    live.log_param("prng_seed", args.prng_seed)
    live.log_param("trial_seed", seed)

    # Build XML with placeholder init — will be overridden below
    placeholder_state = (0.0, 0.0, 0.0, 0.1)
    xml_overrides = {
        "dt": args.dt,
        "state_noise_direction": args.state_noise_direction,
        "state_noise_enabled": args.state_noise_enabled,
        "state_noise_magnitude": args.state_noise_magnitude,
        "state_noise_type": args.state_noise_type,
        "wall_clock": args.wall_clock,
        "crash_stop": args.crash_stop,
        "stall_timeout": args.stall_timeout,
        "lap_count": args.lap_count,
        "use_reachsets": args.use_reachsets,
        "prng_seed": seed,
        "centering_weight_factor": args.centering_weight_factor,
        "velocity_scale_factor": args.velocity_scale_factor,
        "horizon": args.horizon,
        "max_speed": args.max_speed,
        "init_states": placeholder_state,
    }
    tmp_config = build_xml_config(xml_overrides)

    experiment = None
    car = None
    controller = None
    stopper = None
    bc = None
    track = None
    lap_ext = None
    try:
        experiment = Main(tmp_config)
        track = experiment.track

        # Compute the real initial state
        init_x, init_y, init_heading, init_v = compute_initial_state(
            track,
            args.init_point_index,
            args.init_offset_name,
        )
        init_state = (init_x, init_y, init_heading, init_v, 0.0, 0.0)

        # Override car state.  The Laptimer baselines lap/progress tracking
        # lazily from this state on its first update (inside experiment.run()),
        # so no further start-position wiring is needed.
        car = experiment.cars[0]
        car.state = np.array(init_state)

        # Log computed initial state for post-hoc analysis
        live.log_metric("init_x", init_x)
        live.log_metric("init_y", init_y)
        live.log_metric("init_heading", init_heading)
        live.log_metric("init_v_forward", init_v)

        # ------------------------------------------------------------------
        # 2. Run the simulation
        # ------------------------------------------------------------------
        experiment.run()

        # ------------------------------------------------------------------
        # 3. Collect metrics
        # ------------------------------------------------------------------
        controller = car.controller
        stopper = experiment.auto_stopper
        bc = experiment.boundary_checker

        # Whether a collision occurred
        collision_occurred = bc.collision_count[car] > 0
        live.log_metric("collision_occurred", int(collision_occurred))
        live.log_metric("collision_count", int(bc.collision_count[car]))

        # Whether the car stalled
        stalled = stopper.stop_reason == "stall"
        live.log_metric("stalled", int(stalled))

        # Stop reason (categorical -> summary, lands in metrics.json)
        stop_reason = stopper.stop_reason or "unknown"
        live.summary["stop_reason"] = stop_reason

        # Final car position
        final_x = float(car.state[0])
        final_y = float(car.state[1])
        live.log_metric("final_x", final_x)
        live.log_metric("final_y", final_y)

        # Distance progressed: net arc-length travelled from the start, lap-aware
        # and always non-negative (from the Laptimer's progress tracking).
        track_len = track.raceline_len_m
        lap_ext = experiment.laptimer
        distance = max(float(lap_ext.progress_by_car[car]), 0.0)
        live.log_metric("distance_progressed", distance)
        live.log_metric("distance_fraction", distance / track_len)

        # Lap completion metrics
        num_laps = int(lap_ext.total_laps_by_car[car])
        laptimes = lap_ext.laptime_vec_by_car[car]
        live.log_metric("num_laps_complete", num_laps)
        live.log_metric(
            "average_lap_time", float(np.mean(laptimes)) if laptimes else 0.0
        )

        # Unsafe fraction
        unsafe_fraction = float(controller.unsafe_fraction)
        live.log_metric("unsafe_fraction", unsafe_fraction)
        tcs = int(controller._total_control_steps)
        live.log_metric("total_control_steps", tcs)
        live.log_metric("unsafe_control_steps", len(controller._unsafe_control_steps))
        live.log_metric("recent_unsafe_control_steps", len(controller._unsafe_control_steps & set(np.arange(tcs-controller.planning_horizon, tcs+1))))

        # Simulation time
        sim_duration = experiment.simulator.sim_t
        live.log_metric("sim_duration", sim_duration)

        # OOB fraction
        oob_frac = bc.oob_steps[car] / max(bc.total_steps[car], 1)
        live.log_metric("oob_fraction", oob_frac)
    finally:
        try:
            os.unlink(tmp_config)
        except OSError:
            pass
        # Release Python references so the controller (and its JAX device
        # arrays / cost evaluator) become unreachable, then evict JAX's
        # compile cache — `ImmraxController` JIT-decorates instance methods
        # with static_argnums=0, so the cache pins `self` until cleared.
        car = None
        controller = None
        stopper = None
        bc = None
        track = None
        lap_ext = None
        experiment = None
        gc.collect()
        jax.clear_caches()


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    with Live(
        dir=args.out_dir,
        save_dvc_exp=False,
        dvcyaml=False,
        report=None,
    ) as live:
        run(args, live)


if __name__ == "__main__":
    main()
