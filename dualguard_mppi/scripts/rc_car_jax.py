"""DualGuard-MPPI on the RC car (3-state kinematic bicycle), rc_wh warehouse track.

    state x = [x, y, theta]      control u = [v, delta]

Repo-local version of the paper authors' ``rc_car_jax.ipynb``, which was written on
another machine and does not run here as pulled.  ``rc_car_model.py`` holds the
dynamics, the BRT tables and the MPPI update -- including the two quirks kept from
the original on purpose (one PRNG key reused for every step, and no ``min(cost)``
baseline subtraction in the weights).  This file runs the experiment.

Changes from the notebook, beyond the port itself:

* the BRT is read from the ``brt_rc_wh`` DVC stage's own output rather than an
  absolute path on the author's machine, and ``Deriv`` (1.0 GiB, never read) is not
  loaded;
* rollout timings wait on ``block_until_ready``, so they measure compute rather than
  async dispatch;
* rollout bundles are kept every ``stride`` steps rather than all of them.  The
  notebook's ``hallucination_history`` cost ~1.2 GB at a fixed 1000 steps; the run
  length is no longer fixed, so keeping every one is unbounded.  One bundle per
  animation frame is what the GIF needs anyway.

Lap detection projects onto the racing line of ``WarehouseTrack``, this project's
canonical rc_wh geometry, which ``experiments/dualguard_rc_trial.py`` also drives.
That artifact lives in the *shifted* world frame ``reconstruct_track.py`` writes
(grid min corner moved to the origin), so the racing line is shifted back into BRT
grid coordinates before use -- everything else here is in grid coordinates.

Usage:
    uv run dvc repro dualguard_mppi/dvc.yaml:run_rc_car
    uv run python dualguard_mppi/scripts/rc_car_jax.py --out-dir /tmp/rc --laps 1
"""

import os
import time

import _env  # noqa: F401  sys.path setup; must precede the sibling/root imports

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

import plotting
import runner
from lap_tracking import FrenetProjector, LapCounter
from rc_car_model import (DELTA_MAX, DELTA_MIN, DT, GUARD2_THRESHOLD, HORIZON,
                          INITIAL_STATE, NUM_THREADS, V_MAX, V_MIN, build_sim,
                          initial_plan, load_brt, step_euler)
from warehouse_track import DATA_PATH as WAREHOUSE_NPZ, WarehouseTrack

SAVE_ROLLOUT_THREADS = 50     # threads kept per saved control step, for the GIF


def simulate(sim, run, seed, save_every):
    """Receding-horizon loop, in the notebook's order.

    Note that guard 2 is tested *after* the MPPI update here, not before it as in
    ``buzzracer_jax.py`` -- that is the original's ordering and it is kept.

    ``run.record`` owns the stopping rule (crash or laps); see ``runner.py``.
    """
    # Never split: the notebook drew the same block of noise on every control step,
    # and splitting would change the trajectory.  See rc_car_model's docstring.
    key = random.PRNGKey(seed)
    U, state = initial_plan(), INITIAL_STATE
    rollout_steps, rollout_states, step_ms = [], [], []

    for i in range(run.max_steps):
        t0 = time.time()
        U, states_parallel, unsafe, costs, ess = sim["update"](U, key, state)
        U.block_until_ready()

        value_now = float(sim["value"](state))
        guard2 = value_now < GUARD2_THRESHOLD
        action = sim["ctrl"](state) if guard2 else U[0]
        state = step_euler(state, action)

        # Shift the plan forward, holding the last entry (the notebook's roll).
        U = jnp.roll(U, -1, axis=0)
        U = U.at[-1].set(U[-2])
        step_ms.append((time.time() - t0) * 1000)

        if i % save_every == 0:
            rollout_steps.append(i)
            rollout_states.append(np.asarray(states_parallel[:SAVE_ROLLOUT_THREADS]))

        stop = run.record(state, action, guard2, value=value_now, ess=float(ess),
                          guard1_frac=float(jnp.mean(unsafe)),
                          cost_min=float(jnp.min(costs)))
        if i % 50 == 0 or stop:
            print(f"  k {i:4d}  t {(i+1)*DT:6.2f}s  V {value_now:+.4f}  "
                  f"lx {run.log['margin'][-1]:+.4f}  lap {run.laps.laps_completed} "
                  f"({run.laps.cumulative_progress:6.2f} m)  "
                  f"guard2 {'ON ' if guard2 else 'off'}  "
                  f"guard1 {run.log['guard1_frac'][-1]:.3f}  ESS {float(ess):7.1f}  "
                  f"{step_ms[-1]:6.1f} ms")
        if stop:
            break

    run.finish()
    run.log["step_ms"] = np.array(step_ms)
    return rollout_steps, rollout_states


def make_figures(run, brt, args, rollouts, stride):
    """The track panel, plus this experiment's own traces."""
    log, sh, ch = run.log, run.states, run.controls
    xs, ys, ths = brt["grid_axes"]
    rollout_steps, rollout_states = rollouts

    def static(ax):
        """Red: the lx = 0 track boundary -- the only thing on this panel that is fixed."""
        ax.contour(xs, ys, brt["lx"].T, levels=[0], colors="red", linewidths=1)
        ax.set(xlim=(xs[0], xs[-1]), ylim=(ys[0], ys[-1]))
        ax.set_aspect("equal")

    def dynamic(ax, upto, rollouts=True):
        """Blue: V = 0 at the car's heading slice.  Green: a subsample of the rollouts
        the controller had in hand.  Trajectory markers are blue where MPPI's control
        ran and red where guard 2 overrode it.
        """
        s = sh[upto]
        th_idx = int(np.argmin(np.abs(ths - s[2])))
        drawn = [ax.contour(xs, ys, brt["value"][:, :, th_idx].T, levels=[0],
                            colors="blue", linewidths=1)]
        if rollouts and rollout_steps:
            j = int(np.argmin(np.abs(np.array(rollout_steps) - upto)))
            for h in rollout_states[j][::5]:
                drawn += ax.plot(h[:, 0], h[:, 1], color="green", alpha=0.2, lw=0.6,
                                 zorder=2)
        drawn.append(ax.scatter(sh[:upto, 0], sh[:upto, 1], c=run.colors[:upto], s=5,
                                alpha=0.6, zorder=4))
        drawn.append(ax.scatter(s[0], s[1], s=25, c="k", zorder=5))
        drawn.append(ax.arrow(s[0], s[1], np.cos(s[2]) * 0.2, np.sin(s[2]) * 0.2,
                              head_width=0.08, head_length=0.08, fc="k", ec="k", zorder=5))
        return drawn

    def caption(k):
        return (f"t = {(k+1)*DT:5.2f} s   V = {log['value'][k]:+.3f}   "
                f"lx = {log['margin'][k]:+.3f}\n"
                f"guard 2 {'ON' if log['guard2'][k] else 'off'}   lap {log['laps'][k]}")

    dotted = dict(color="0.7", ls=":")
    plotting.standard_figures(
        run, args, DT, static, dynamic, caption, stride=stride, figsize=(9, 6),
        gif_figsize=(7, 5),
        trajectory_title=(f"{run.stop_reason} at {run.sim_duration:.1f} s, "
                          f"{log['laps'][-1]} laps\nred: lx = 0 boundary   "
                          "blue: V = 0 at the final heading slice\n"
                          "trajectory red where guard 2 overrode MPPI"),
        traces=[
            dict(name="velocity_control.png", series=[(ch[:, 0], "")], ylabel="v [m/s]",
                 title="executed velocity control",
                 hlines=[(V_MIN, dotted), (V_MAX, dotted)]),
            dict(name="steering_control.png", series=[(ch[:, 1], "")],
                 ylabel="delta [rad]", title="executed steering control",
                 hlines=[(DELTA_MIN, dotted), (DELTA_MAX, dotted)]),
            # The margin now ends the run, so it gets a trace of its own.  This is the
            # BRT's own l(x), the field the solve was run against.
            dict(name="margin.png", series=[(log["margin"], "")],
                 ylabel="signed distance [m]",
                 title="margin function: distance to the track boundary -- crash is < 0",
                 hlines=[(0.0, dict(color="k", lw=0.8))]),
            dict(name="value.png", series=[(log["value"], "V(x)")], ylabel="V",
                 title="BRT value along the executed trajectory",
                 hlines=[(0.0, dict(color="k", lw=0.8)), (GUARD2_THRESHOLD, dotted)],
                 shade=log["guard2"], shade_label="guard 2 active"),
        ])


def main(argv=None):
    p = runner.base_parser(__doc__, os.path.join(_env.RUNS_DIR, "rc_car"))
    p.add_argument("--track-npz", dest="track_npz", default=WAREHOUSE_NPZ,
                   help="WarehouseTrack geometry, for lap detection")
    args = p.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    print("jax", jax.__version__, "| devices:", jax.devices())

    brt = load_brt(_env.require(_env.BRT_RC_WH, "dualguard_mppi/dvc.yaml:brt_rc_wh"))
    sim = build_sim(brt)

    track = WarehouseTrack(data_path=_env.require(args.track_npz, "reconstruct_track"))
    proj = FrenetProjector.from_track(track)
    proj.rx += brt["grid_min"][0]      # shifted world frame -> BRT grid coordinates
    proj.ry += brt["grid_min"][1]
    print(f"  racing line {proj.track_len:.2f} m from {args.track_npz}")

    laps = LapCounter(proj, np.asarray(INITIAL_STATE)[:2], target_laps=args.laps)
    run = runner.Run(DT, laps, lambda s: sim["margin"](jnp.asarray(s)), args.max_sim_time)

    # One saved rollout bundle per animation frame, so the GIF shows the rollouts the
    # controller actually had in hand at that instant.
    stride = plotting.real_time_stride(DT, args.gif_fps)
    print(f"\nrunning until crash or {args.laps} laps "
          f"(backstop {args.max_sim_time:.0f} s = {run.max_steps} steps)")
    rollouts = simulate(sim, run, args.seed, save_every=stride)

    log = run.log
    timing_line, timing_fields = runner.timing(log["step_ms"], DT)
    summary = run.report(extra=[
        ("guard 1 mean / max share", f"{100*log['guard1_frac'].mean():.1f}% / "
                                     f"{100*log['guard1_frac'].max():.1f}%"),
        ("min V along trajectory", f"{log['value'].min():+.4f}"),
        ("steps with V <= 0", f"{(log['value'] <= 0).sum()}"),
        ("collisions (lx <= 0)", f"{(log['margin'] <= 0).sum()}"),
        (f"ESS mean of K = {NUM_THREADS}", f"{log['ess'].mean():.1f} "
                                           f"[{log['ess'].min():.1f}, {log['ess'].max():.1f}]"),
        ("sim step", timing_line),
    ])
    summary.update(
        **timing_fields,
        guard1_mean_fraction=float(log["guard1_frac"].mean()),
        value_min=float(log["value"].min()),
        steps_inside_brt=int((log["value"] <= 0).sum()),
        collisions=int((log["margin"] <= 0).sum()),
        ess_mean=float(log["ess"].mean()),
        horizon_steps=HORIZON,
        num_threads=NUM_THREADS)

    print("\nwriting outputs")
    make_figures(run, brt, args, rollouts, stride)
    plotting.save_json(summary, args.out_dir)


if __name__ == "__main__":
    main()
