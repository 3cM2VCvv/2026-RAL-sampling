"""Single controller_sweep trial, instrumented with DVCLive.

Runs one Buzzracer simulation for one combination of controller, track and
disturbance type.  The track is loaded from a normalized fixture built by
``build_track_fixtures``; the controller is selected from
``_controller_registry``.

Recorded per trial: outcome (race_finished / crash / stall / incomplete,
derived from ``SimTimeAutoStopper.stop_reason``), average control-loop time,
average lap completion time, plus the usual progress/safety context.

This is the unit stage of the ``controller_trial`` matrix in ``dvc.yaml``.
The swept axes and fixed ``sim`` / ``controller`` params are passed as CLI
flags; booleans render as store_true flags.

Usage (single run):
    uv run python -m experiments.controller_sweep.controller_sweep \
        --controller immrax --track saved --state_noise_direction uniform \
        --dt 0.02 --state_noise_enabled --state_noise_magnitude 0.1 \
        --state_noise_type lateral_only --wall_clock 180 --crash_stop \
        --stall_timeout 10 --lap_count 10 --horizon 30 --max_speed 2.0 \
        --prng_seed 42 --use_reachsets --centering_weight_factor 1.0 \
        --velocity_scale_factor 1.0 --out-dir outputs/controller_sweep/runs/manual
"""

import argparse
import gc
import os
import time

import _env  # noqa: F401  sets up sys.path for buzzracer + root helpers
import jax
import numpy as np
from _controller_registry import available, ensure_registered, resolve
from _extensions import register_simtime_stopper
from _track_registry import register_fixture_tracks
from _trial_utils import build_xml_config, compute_initial_state, trial_seed
from buzzracer.cars.car import Car
from buzzracer.common import BASEDIR
from buzzracer.extensions.extension import Extension
from buzzracer.scripts.run import Main
from dvclive import Live
from raceline_track import FIXTURE_DIR, RacelineTrack

BASE_XML = os.path.join(BASEDIR, "buzzracer", "configs", "reachset_sweep_base.xml")
WARMUP_STEPS = 5

_OUTCOME = {
    "race_finished": "race_finished",
    "crash": "crash",
    "stall": "stall",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    # Swept axes (dvc.yaml ${item})
    p.add_argument("--controller", required=True, choices=available())
    p.add_argument("--track", required=True)
    p.add_argument(
        "--state_noise_direction",
        required=True,
        choices=["uniform", "adversarial_away"],
    )
    # Fixed sim params (dvc.yaml ${sim})
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--state_noise_enabled", action="store_true")
    p.add_argument("--state_noise_magnitude", type=float, default=0.1)
    p.add_argument("--state_noise_type", default="lateral_only")
    p.add_argument("--wall_clock", type=float, default=60.0)
    p.add_argument("--crash_stop", action="store_true")
    p.add_argument("--stall_timeout", type=float, default=10.0)
    p.add_argument("--lap_count", type=int, default=3)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--max_speed", type=float, default=2.0)
    p.add_argument(
        "--prng_seed",
        type=int,
        default=42,
        help="base seed; combined with (track, disturbance) only, "
        "so all controllers share the plant-noise stream",
    )
    # Fixed controller params (dvc.yaml ${controller})
    p.add_argument("--use_reachsets", action="store_true")
    p.add_argument("--centering_weight_factor", type=float, default=1.0)
    p.add_argument("--velocity_scale_factor", type=float, default=1.0)
    # Shield-MPPI settings (ignored by ImmraxController)
    p.add_argument("--shield_num_samples", type=int, default=1024)
    p.add_argument("--shield_param_lambda", type=float, default=1.0)
    p.add_argument("--shield_param_alpha", type=float, default=0.0)
    p.add_argument("--shield_param_beta", type=float, default=0.5)
    p.add_argument("--shield_cbf_cost_weight", type=float, default=1000.0)
    p.add_argument("--shield_w_T", type=float, default=0.0)
    p.add_argument("--shield_repair_horizon", type=int, default=10)
    p.add_argument("--shield_repair_steps", type=int, default=20)
    # MPPI settings (ignored by ImmraxController / ShieldMPPIController)
    p.add_argument("--mppi_num_samples", type=int, default=1024)
    p.add_argument("--mppi_param_lambda", type=float, default=1.0)
    # Tube-MPPI settings (ignored by ImmraxController / ShieldMPPIController)
    # None -> min of the cost's safety-critical weights.
    p.add_argument("--tube_accept_margin", type=float, default=None)
    p.add_argument("--tube_control_cost_lambda", type=float, default=0.03)
    p.add_argument("--tube_ancillary_horizon", type=int, default=0)
    p.add_argument("--tube_ancillary_iterations", type=int, default=1)
    p.add_argument("--tube_ancillary_mu", type=float, default=1e-2)
    p.add_argument("--tube_ancillary_mu_factor", type=float, default=5.0)
    p.add_argument("--tube_ancillary_q_scale", type=float, default=1.0)
    p.add_argument("--tube_ancillary_r_scale", type=float, default=1.0)
    p.add_argument("--tube_ancillary_terminal_scale", type=float, default=10.0)
    # Output
    p.add_argument(
        "--out-dir",
        dest="out_dir",
        required=True,
        help="DVCLive directory for this trial's params/metrics",
    )
    return p.parse_args(argv)


def run(args, live):
    # ------------------------------------------------------------------
    # 1. Set up registries and the initial state
    # ------------------------------------------------------------------
    Extension.extensions = []
    Car.reset()

    spec = resolve(args.controller)
    ensure_registered(spec)
    register_simtime_stopper()
    register_fixture_tracks([args.track])

    # The controller is intentionally absent from the seed: all controllers
    # face the same disturbance stream on a given (track, disturbance) cell.
    seed = trial_seed(args.prng_seed, args.track, args.state_noise_direction)

    params = {
        "controller": args.controller,
        "track": args.track,
        "state_noise_direction": args.state_noise_direction,
        "prng_seed": args.prng_seed,
        "trial_seed": seed,
        "use_reachsets": args.use_reachsets,
        "centering_weight_factor": args.centering_weight_factor,
        "velocity_scale_factor": args.velocity_scale_factor,
        "state_noise_magnitude": args.state_noise_magnitude,
        "state_noise_type": args.state_noise_type,
        "lap_count": args.lap_count,
        "horizon": args.horizon,
        "max_speed": args.max_speed,
    }
    for key, value in params.items():
        live.log_param(key, value)

    fixture_path = os.path.join(FIXTURE_DIR, f"{args.track}.npz")
    init_track = RacelineTrack(main=None, config=None, fixture_path=fixture_path)
    init_x, init_y, init_heading, init_v = compute_initial_state(init_track)
    init_track = None

    sim = {
        "dt": args.dt,
        "state_noise_direction": args.state_noise_direction,
        "state_noise_enabled": args.state_noise_enabled,
        "state_noise_magnitude": args.state_noise_magnitude,
        "state_noise_type": args.state_noise_type,
        "wall_clock": args.wall_clock,
        "crash_stop": args.crash_stop,
        "stall_timeout": args.stall_timeout,
        "lap_count": args.lap_count,
    }
    controller = {
        "use_reachsets": args.use_reachsets,
        "prng_seed": seed,
        "centering_weight_factor": args.centering_weight_factor,
        "velocity_scale_factor": args.velocity_scale_factor,
        "horizon": args.horizon,
        "max_speed": args.max_speed,
        "shield_num_samples": args.shield_num_samples,
        "shield_param_lambda": args.shield_param_lambda,
        "shield_param_alpha": args.shield_param_alpha,
        "shield_param_beta": args.shield_param_beta,
        "shield_cbf_cost_weight": args.shield_cbf_cost_weight,
        "shield_w_T": args.shield_w_T,
        "shield_repair_horizon": args.shield_repair_horizon,
        "shield_repair_steps": args.shield_repair_steps,
        "mppi_num_samples": args.mppi_num_samples,
        "mppi_param_lambda": args.mppi_param_lambda,
        "tube_control_cost_lambda": args.tube_control_cost_lambda,
        "tube_ancillary_horizon": args.tube_ancillary_horizon,
        "tube_ancillary_iterations": args.tube_ancillary_iterations,
        "tube_ancillary_mu": args.tube_ancillary_mu,
        "tube_ancillary_mu_factor": args.tube_ancillary_mu_factor,
        "tube_ancillary_q_scale": args.tube_ancillary_q_scale,
        "tube_ancillary_r_scale": args.tube_ancillary_r_scale,
        "tube_ancillary_terminal_scale": args.tube_ancillary_terminal_scale,
    }
    if args.tube_accept_margin is not None:
        controller["tube_accept_margin"] = args.tube_accept_margin
    tmp_config = build_xml_config(
        BASE_XML,
        track=args.track,
        controller_type=spec.type_name,
        init_states=(init_x, init_y, init_heading, init_v),
        sim=sim,
        controller=controller,
    )

    experiment = None
    car = None
    controller_obj = None
    stopper = None
    bc = None
    track = None
    lap_ext = None
    try:
        experiment = Main(tmp_config)
        car = experiment.cars[0]
        controller_obj = car.controller
        track = experiment.track

        live.log_metric("init_x", init_x)
        live.log_metric("init_y", init_y)
        live.log_metric("init_heading", init_heading)
        live.log_metric("init_v_forward", init_v)
        live.log_metric("track_len", float(track.raceline_len_m))

        # ------------------------------------------------------------------
        # 2. Run, timing every control() call
        # ------------------------------------------------------------------
        np.random.seed(seed)
        control_times_ms = []
        original_control = controller_obj.control

        def timed_control(*a, **kw):
            t0 = time.perf_counter()
            result = original_control(*a, **kw)
            control_times_ms.append((time.perf_counter() - t0) * 1000.0)
            return result

        controller_obj.control = timed_control
        experiment.run()

        # ------------------------------------------------------------------
        # 3. Collect metrics
        # ------------------------------------------------------------------
        stopper = experiment.auto_stopper
        bc = experiment.boundary_checker
        lap_ext = experiment.laptimer

        stop_reason = stopper.stop_reason or "unknown"
        outcome = _OUTCOME.get(stop_reason, "incomplete")
        live.summary["stop_reason"] = stop_reason
        live.summary["outcome"] = outcome
        live.log_metric("stalled", int(stop_reason == "stall"))

        collision_count = int(bc.collision_count[car])
        live.log_metric("collision_occurred", int(collision_count > 0))
        live.log_metric("collision_count", collision_count)
        live.log_metric(
            "oob_fraction",
            bc.oob_steps[car] / max(bc.total_steps[car], 1),
        )

        live.log_metric("final_x", float(car.state[0]))
        live.log_metric("final_y", float(car.state[1]))

        track_len = float(track.raceline_len_m)
        distance = max(float(lap_ext.progress_by_car[car]), 0.0)
        live.log_metric("distance_progressed", distance)
        live.log_metric("distance_fraction", distance / track_len)

        num_laps = int(lap_ext.total_laps_by_car[car])
        laptimes = lap_ext.laptime_vec_by_car[car]
        live.log_metric("num_laps_complete", num_laps)
        live.log_metric(
            "average_lap_time", float(np.mean(laptimes)) if laptimes else 0.0
        )

        times = np.asarray(control_times_ms, dtype=float)
        timed = times[WARMUP_STEPS:] if times.size > WARMUP_STEPS else times
        live.log_metric("control_steps_timed", int(timed.size))
        live.log_metric(
            "control_time_ms_mean", float(timed.mean()) if timed.size else 0.0
        )
        live.log_metric(
            "control_time_ms_std", float(timed.std()) if timed.size else 0.0
        )
        live.log_metric(
            "control_time_ms_median", float(np.median(timed)) if timed.size else 0.0
        )

        planner_log = getattr(controller_obj, "_timing_log", None)
        if planner_log:
            live.log_metric("planner_compute_ms_mean", float(np.mean(planner_log)))

        live.log_metric("sim_duration", float(experiment.simulator.sim_t))

        if hasattr(controller_obj, "unsafe_fraction"):
            live.log_metric("unsafe_fraction", float(controller_obj.unsafe_fraction))
        if hasattr(controller_obj, "_total_control_steps"):
            tcs = int(controller_obj._total_control_steps)
            unsafe_steps = getattr(controller_obj, "_unsafe_control_steps", set())
            live.log_metric("total_control_steps", tcs)
            live.log_metric("unsafe_control_steps", len(unsafe_steps))
            horizon = getattr(controller_obj, "planning_horizon", None)
            if horizon is not None:
                live.log_metric(
                    "recent_unsafe_control_steps",
                    len(unsafe_steps & set(np.arange(tcs - horizon, tcs + 1))),
                )
    finally:
        try:
            os.unlink(tmp_config)
        except OSError:
            pass
        car = None
        controller_obj = None
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
