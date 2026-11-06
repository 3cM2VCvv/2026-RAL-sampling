"""DualGuard-MPPI on the Buzzracer 6-state dynamic bicycle, RCP tile track.

    state x = [x, y, psi, v_forward, v_sideway, omega]
    control u = [steering, throttle]      disturbance d = [w_v_sideway, w_v_heading]

Follows the DualGuard-MPPI control profile as the paper authors' repo defines it
(``src/config/control_profiles.yaml``, implemented in ``src/controller_mppi.py``):
both guards on -- a least-restrictive filter inside every rollout step and again on
the executed control -- with ``cost_from_obstacles_or_BRT: obs``, so the running cost
penalises the failure set rather than the BRT.  ``buzzracer_model.py`` holds the
dynamics, the BRT tables and the MPPI update; this file runs the experiment.

Differences from ``rc_car_jax.py``, and why:

1. The PRNG is split every step.  The original notebook reused one key, so every
   control step drew the same block of noise; the rc_car script keeps that on purpose
   (it would change that trajectory), this does not.
2. Adversarial disturbance is injected into the plant at full magnitude from the
   exported ``dOpt_*`` tables (``|d| = 0.1``, bang-bang).  The rc_wh BRT was solved
   with ``dMax = 0``; this one against a +/-0.1 box with ``dMode = 'min'``, so the
   certified claim only means anything against that adversary.
3. Out-of-grid states read as unsafe rather than clamping to the nearest edge cell.
   In 6-D this matters: x and y are bounded by the track, but v_forward, v_sideway and
   omega have bounds a spinning car really can exceed.

Forward Euler is kept in plant and rollouts alike.  That makes MPPI's predictions
unbiased with respect to the plant but says nothing about the safety claim, which
comes from the BRT -- and ``HJIPDE_solve`` computed that for the continuous-time flow,
not for a 0.02 s Euler step.  One step here can move v_sideway by up to 1.1 m/s
against a cell width of 0.75-1.0 m/s, i.e. more than a full cell, which is exactly the
regime where a filter can certify the pre-step state and land the post-step state
outside the tube.

Usage:
    uv run dvc repro experiments/dualguard_mppi/dvc.yaml:run_buzzracer
    uv run python -m experiments.dualguard_mppi.scripts.buzzracer_jax --out-dir /tmp/bz --laps 1
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from scipy.io import loadmat

from . import _dg_env as _env
from . import plotting, runner
from .buzzracer_model import (
    DELTA_MAX,
    DT,
    GUARD2_THRESHOLD,
    HORIZON,
    INITIAL_STATE,
    LABELS,
    THROTTLE_MAX,
    THROTTLE_MIN,
    U_INIT,
    V_TARGET,
    build_sim,
    initial_plan,
    load_brt,
    step_euler,
    wrap_angle,
)
from .lap_tracking import FrenetProjector, LapCounter

SAVE_ROLLOUT_THREADS = 50  # threads kept per saved control step, for the GIF


def simulate(sim, run, seed, save_every):
    """Receding-horizon loop, ordered as ``ExperimentRunner.run``.

    1. shift the nominal plan forward, appending U_INIT;
    2. guard-2 test on the *measured* state, before MPPI runs;
    3. MPPI, which applies guard 1 inside every rollout step;
    4. override the MPPI action with the safety control if guard 2 fired;
    5. step the plant, against the worst-case disturbance from the dOpt tables.

    ``run.record`` owns the stopping rule (crash or laps); see ``runner.py``.
    """
    key, subkey = random.split(random.PRNGKey(seed))
    U, state = initial_plan(subkey), INITIAL_STATE
    rollout_steps, rollout_states, step_ms = [], [], []

    for i in range(run.max_steps):
        t0 = time.time()
        U = jnp.roll(U, -1, axis=0).at[-1].set(U_INIT)

        value_now, oob_now = float(sim["value"](state)), bool(sim["oob"](state))
        guard2 = value_now <= GUARD2_THRESHOLD

        key, subkey = random.split(key)  # fresh noise every step
        U, states_parallel, unsafe, costs, ess = sim["update"](U, subkey, state)
        U.block_until_ready()

        action = sim["ctrl"](state) if guard2 else U[0]
        state = step_euler(state, action, sim["dstb"](state))
        step_ms.append((time.time() - t0) * 1000)

        if i % save_every == 0:
            rollout_steps.append(i)
            rollout_states.append(np.asarray(states_parallel[:SAVE_ROLLOUT_THREADS]))

        stop = run.record(
            state,
            action,
            guard2,
            value=value_now,
            oob=oob_now,
            lx_coarse=float(sim["lx"](state)),
            ess=float(ess),
            guard1_frac=float(jnp.mean(unsafe)),
        )
        if i % 25 == 0 or stop:
            print(
                f"  k {i:4d}  t {(i + 1) * DT:6.2f}s  V {value_now:+.4f}  "
                f"lx {run.log['margin'][-1]:+.4f}  lap {run.laps.laps_completed} "
                f"({run.laps.cumulative_progress:6.2f} m)  "
                f"guard2 {'ON ' if guard2 else 'off'}  "
                f"guard1 {run.log['guard1_frac'][-1]:.3f}  ESS {float(ess):7.1f}  "
                f"{step_ms[-1]:6.1f} ms"
            )
        if stop:
            break

    run.finish()
    run.log["step_ms"] = np.array(step_ms)
    return rollout_steps, rollout_states


def make_figures(run, brt, track, args, rollouts, stride):
    """The track panel, plus this experiment's own traces."""
    log, sh, ch = run.log, run.states, run.controls
    ax_g = brt["grid_axes"]
    rollout_steps, rollout_states = rollouts

    def nearest(axis, v):
        return int(
            np.clip(round((v - axis[0]) / (axis[1] - axis[0])), 0, axis.size - 1)
        )

    def static(ax):
        """The walls and both failure-set contours -- everything that never moves.

        Grey dashed is the 1 cm reference boundary, red the failure set as the 6-D
        solver resolved it; the gap between them is the grid-sparsity pitfall drawn to
        scale.
        """
        for x1, y1, x2, y2 in track["wall_segments"]:
            ax.plot([x1, x2], [y1, y2], color="k", lw=1.2, zorder=3)
        for cx, cy, r, th0, th1 in track["wall_arcs"]:
            th = np.linspace(th0, th1, 60)
            ax.plot(
                cx + r * np.cos(th), cy + r * np.sin(th), color="k", lw=1.2, zorder=3
            )
        ax.contour(
            track["lx_ref_x"],
            track["lx_ref_y"],
            track["lx_ref"].T,
            [0],
            colors="0.6",
            linewidths=1,
            linestyles="--",
        )
        ax.contour(
            brt["lx_axes"][0],
            brt["lx_axes"][1],
            brt["lx"].T,
            [0],
            colors="red",
            linewidths=1,
        )
        ax.set(xlim=track["track_xlim"], ylim=track["track_ylim"])
        ax.set_aspect("equal")

    def dynamic(ax, upto, rollouts=True):
        """The V = 0 slice, the rollouts, and the trajectory up to step ``upto``.

        Blue is V = 0 sliced at the car's current (psi, v_fwd, v_side, omega) -- in 3-D
        one index had to be fixed, here four do, so the slice changes shape as the
        velocity state moves.  Green are the rollouts the controller had in hand.
        """
        s = sh[upto]
        drawn = [
            ax.contour(
                ax_g[0],
                ax_g[1],
                brt["value"][
                    :,
                    :,
                    nearest(ax_g[2], float(wrap_angle(s[2]))),
                    nearest(ax_g[3], s[3]),
                    nearest(ax_g[4], s[4]),
                    nearest(ax_g[5], s[5]),
                ].T,
                [0],
                colors="blue",
                linewidths=1,
            )
        ]
        if rollouts and rollout_steps:
            j = int(np.argmin(np.abs(np.array(rollout_steps) - upto)))
            for h in rollout_states[j][::5]:
                drawn += ax.plot(
                    h[:, 0], h[:, 1], color="green", alpha=0.2, lw=0.6, zorder=2
                )
        drawn.append(
            ax.scatter(
                sh[:upto, 0],
                sh[:upto, 1],
                c=run.colors[:upto],
                s=5,
                alpha=0.6,
                zorder=4,
            )
        )
        drawn.append(ax.scatter(s[0], s[1], s=25, c="k", zorder=5))
        drawn.append(
            ax.arrow(
                s[0],
                s[1],
                np.cos(s[2]) * 0.15,
                np.sin(s[2]) * 0.15,
                head_width=0.05,
                head_length=0.05,
                fc="k",
                ec="k",
                zorder=5,
            )
        )
        return drawn

    def caption(k):
        return (
            f"t = {(k + 1) * DT:5.2f} s   V = {log['value'][k]:+.3f}   "
            f"lx = {log['margin'][k]:+.3f}\n"
            f"guard 2 {'ON' if log['guard2'][k] else 'off'}   lap {log['laps'][k]}"
        )

    dotted = dict(color="0.7", ls=":")
    plotting.standard_figures(
        run,
        args,
        DT,
        static,
        dynamic,
        caption,
        stride=stride,
        trajectory_title=(
            f"{run.stop_reason} at {run.sim_duration:.1f} s, "
            f"{log['laps'][-1]} laps\ngrey: true boundary   "
            "red: solver lx = 0   blue: V = 0 slice\n"
            "trajectory red where guard 2 overrode MPPI"
        ),
        traces=[
            dict(
                name="steering.png",
                series=[(ch[:, 0], "")],
                ylabel="delta [rad]",
                title="executed steering",
                hlines=[(DELTA_MAX, dotted), (-DELTA_MAX, dotted)],
            ),
            dict(
                name="throttle.png",
                series=[(ch[:, 1], "")],
                ylabel="throttle [-]",
                title="executed throttle",
                ylim=(THROTTLE_MIN - 0.1, THROTTLE_MAX + 0.1),
            ),
            dict(
                name="velocity_states.png",
                title="velocity states",
                series=[
                    (sh[:, 3], "v_fwd [m/s]"),
                    (sh[:, 4], "v_side [m/s]"),
                    (sh[:, 5], "omega [rad/s]"),
                ],
            ),
            # The margin is what now ends the run, so it gets a trace of its own.  Both
            # lx curves are shown: the walls are 5.22 cm thick, thinner than one cell of
            # the 6-D grid, so a trajectory the filter certifies can still be off the
            # track.  That gap is a property of the grid, not of DualGuard.
            dict(
                name="margin.png",
                ylabel="signed distance [m]",
                title="margin function -- crash is < 0",
                series=[
                    (log["margin"], "lx, 1 cm truth (margin)"),
                    (log["lx_coarse"], "lx, coarse BRT grid"),
                ],
                hlines=[(0.0, dict(color="k", lw=0.8))],
            ),
            dict(
                name="value.png",
                series=[(log["value"], "V(x)")],
                ylabel="V",
                title="BRT value along the executed trajectory",
                hlines=[(0.0, dict(color="k", lw=0.8)), (GUARD2_THRESHOLD, dotted)],
                shade=log["guard2"],
                shade_label="guard 2 active",
            ),
        ],
    )


def main(argv=None):
    args = runner.base_parser(
        __doc__, os.path.join(_env.RUNS_DIR, "buzzracer")
    ).parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    print("jax", jax.__version__, "| devices:", jax.devices())

    brt = load_brt(
        _env.require(
            _env.BRT_BUZZRACER, "experiments/dualguard_mppi/dvc.yaml:brt_buzzracer"
        )
    )
    track = loadmat(
        _env.require(
            _env.BUZZRACER_TRACK, "experiments/dualguard_mppi/dvc.yaml:raceline_to_mat"
        ),
        simplify_cells=True,
    )
    sim = build_sim(brt, track)
    # The raceline reaches 3.90 m/s but the BRT grid stops at v_forward = 3.2, so a
    # controller chasing raceline speed would drive the state off the grid, where the
    # value query (correctly) reports unsafe.  V_TARGET stays inside the grid.
    print(
        f"  track {track['track_xlim'][1]:.2f} x {track['track_ylim'][1]:.2f} m, "
        f"raceline {track['raceline_len_m']:.3f} m, corridor half-width "
        f"{sim['lx_max']:.4f} m, V_TARGET {V_TARGET} vs grid max "
        f"{brt['grid_axes'][3][-1]:.2f} m/s"
    )

    laps = LapCounter(
        FrenetProjector(
            track["raceline_xy"],
            track["raceline_heading"],
            track["raceline_s"],
            track["raceline_len_m"],
        ),
        np.asarray(INITIAL_STATE)[:2],
        target_laps=args.laps,
    )
    run = runner.Run(
        DT,
        laps,
        lambda s: sim["margin"](jnp.asarray(s, jnp.float32)),
        args.max_sim_time,
    )

    # One saved rollout bundle per animation frame, so the GIF shows the rollouts the
    # controller actually had in hand at that instant.
    stride = plotting.real_time_stride(DT, args.gif_fps)
    print(
        f"\nrunning until crash or {args.laps} laps "
        f"(backstop {args.max_sim_time:.0f} s = {run.max_steps} steps)"
    )
    rollouts = simulate(sim, run, args.seed, save_every=stride)

    log = run.log
    timing_line, timing_fields = runner.timing(log["step_ms"], DT)
    summary = run.report(
        extra=[
            (
                "guard 1 mean / max share",
                f"{100 * log['guard1_frac'].mean():.1f}% / "
                f"{100 * log['guard1_frac'].max():.1f}%",
            ),
            ("min V along trajectory", f"{log['value'].min():+.4f}"),
            (
                "steps with V <= 0 / off grid",
                f"{(log['value'] <= 0).sum()} / {log['oob'].sum()}",
            ),
            (
                "collisions, coarse / true lx",
                f"{(log['lx_coarse'] <= 0).sum()} / {(log['margin'] <= 0).sum()}",
            ),
            (
                "ESS mean of K = 1000",
                f"{log['ess'].mean():.1f} "
                f"[{log['ess'].min():.1f}, {log['ess'].max():.1f}]",
            ),
            ("sim step", timing_line),
        ]
    )
    summary.update(
        **timing_fields,
        guard1_mean_fraction=float(log["guard1_frac"].mean()),
        value_min=float(log["value"].min()),
        steps_inside_brt=int((log["value"] <= 0).sum()),
        steps_off_grid=int(log["oob"].sum()),
        collisions_coarse_lx=int((log["lx_coarse"] <= 0).sum()),
        collisions_true_lx=int((log["margin"] <= 0).sum()),
        corridor_half_width_m=sim["lx_max"],
        ess_mean=float(log["ess"].mean()),
        horizon_steps=HORIZON,
        # [min, max] reached vs [min, max] of the grid axis, per state.
        state_envelope={
            lab: [
                float(run.states[:, j].min()),
                float(run.states[:, j].max()),
                float(brt["grid_axes"][j][0]),
                float(brt["grid_axes"][j][-1]),
            ]
            for j, lab in enumerate(LABELS)
        },
    )

    print("\nwriting outputs")
    make_figures(run, brt, track, args, rollouts, stride)
    plotting.save_json(summary, args.out_dir)


if __name__ == "__main__":
    main()
