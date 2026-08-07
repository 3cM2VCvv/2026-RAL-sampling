"""Executed-trajectory history and the crash-or-laps stopping rule.

Replaces the notebooks' fixed 20 simulated seconds: a run ends when the margin
function goes negative (the car is through the wall) or the lap counter books its
target laps, whichever comes first.  ``max_sim_time`` is only a backstop against a
run that does neither, so a stuck controller reports rather than hangs.

The margin is evaluated on the *post-step* state and recorded before the run stops,
so the step that crashes the car is in the history and in the plots.
"""

import argparse

import numpy as np

# Executed trajectory: red where guard 2 overrode MPPI, blue where MPPI's own control ran.
COLOR_GUARD2 = (0.8, 0.4, 0.4)
COLOR_MPPI = (0.4, 0.4, 0.8)


def base_parser(description, default_out_dir):
    """The CLI both experiments share."""
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", dest="out_dir", default=default_out_dir,
                   help="directory for the figures, GIF and summary.json")
    p.add_argument("--laps", type=int, default=3, help="stop after this many laps")
    p.add_argument("--max-sim-time", dest="max_sim_time", type=float, default=120.0,
                   help="backstop in simulated seconds, for a run that neither "
                        "crashes nor finishes its laps")
    p.add_argument("--gif-fps", dest="gif_fps", type=float, default=25.0,
                   help="target GIF frame rate; the frame delay is always the real "
                        "simulated time between frames, so playback is real speed "
                        "whatever this is")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-gif", action="store_true",
                   help="figures and summary only (the GIF dominates output time)")
    return p


def timing(step_ms, dt, warmup=5):
    """Per-step compute stats with the warm-up dropped, as (report line, fields).

    The median is the headline.  JAX compiles a branch the first time it is taken, so
    a run whose safety filter first fires at step 900 pays a multi-second compile
    there; that one sample drags the mean past a budget the controller otherwise
    meets on every step.
    """
    s = np.asarray(step_ms[warmup:])
    med, mean = float(np.median(s)), float(s.mean())
    line = (f"{med:.1f} ms median, {mean:.1f} +/- {s.std():.1f} ms mean (budget "
            f"{dt*1000:.0f} ms: {'met' if med < dt * 1000 else 'NOT met'})")
    return line, dict(step_ms_median=med, step_ms_mean=mean,
                      real_time_budget_met=bool(med < dt * 1000))


class Run:
    def __init__(self, dt, lap_counter, margin_fn, max_sim_time):
        self.dt = dt
        self.laps = lap_counter
        self.margin_fn = margin_fn
        self.max_steps = int(round(max_sim_time / dt))
        self.states, self.controls, self.colors = [], [], []
        self.log = {}
        self.stop_reason = "max_sim_time"
        self.crashed = False

    def __len__(self):
        return len(self.states)

    @property
    def t(self):
        """Sim time of each recorded state.  These are *post*-step, so the first is dt."""
        return (np.arange(len(self.states)) + 1) * self.dt

    @property
    def sim_duration(self):
        return len(self.states) * self.dt

    def record(self, state, control, guard2, **fields):
        """Log one executed step.  Returns True when the run should stop."""
        state = np.asarray(state, dtype=float)
        self.states.append(state)
        self.controls.append(np.asarray(control, dtype=float))
        self.colors.append(COLOR_GUARD2 if guard2 else COLOR_MPPI)

        margin = float(self.margin_fn(state))
        t = len(self.states) * self.dt
        laps_done = self.laps.update(state[0], state[1], t)
        fields.update(margin=margin, guard2=bool(guard2),
                      progress=self.laps.cumulative_progress,
                      laps=self.laps.laps_completed)
        for k, v in fields.items():
            self.log.setdefault(k, []).append(v)

        if margin <= 0.0:
            self.crashed, self.stop_reason = True, "crash"
        elif laps_done:
            self.stop_reason = "laps_complete"
        elif len(self.states) >= self.max_steps:
            self.stop_reason = "max_sim_time"
        else:
            return False
        return True

    def finish(self):
        """Freeze the histories into arrays."""
        self.states = np.array(self.states)
        self.controls = np.array(self.controls)
        self.colors = np.array(self.colors)
        self.log = {k: np.array(v) for k, v in self.log.items()}
        return self

    def summary(self):
        """Outcome fields common to both experiments; each adds its own."""
        lc, log = self.laps, self.log
        return {
            "stop_reason": self.stop_reason,
            "crashed": bool(self.crashed),
            "laps_target": lc.target_laps,
            "laps_complete": lc.laps_completed,
            "lap_times_s": [float(v) for v in lc.lap_times],
            "lap_speeds_mps": [float(v) for v in lc.lap_speeds],
            "track_len_m": lc.proj.track_len,
            "distance_progressed_m": float(lc.cumulative_progress),
            "sim_duration_s": self.sim_duration,
            "control_steps": len(self.states),
            "margin_min_m": float(log["margin"].min()),
            "guard2_active_steps": int(log["guard2"].sum()),
            "guard2_active_fraction": float(log["guard2"].mean()),
        }

    def report(self, extra=()):
        """Compact stdout record of the run (the notebooks' printed cells)."""
        s = self.summary()
        print(f"\n  {'stop reason':<28s} {s['stop_reason']}")
        print(f"  {'laps complete':<28s} {s['laps_complete']} of {s['laps_target']}")
        print(f"  {'distance progressed':<28s} {s['distance_progressed_m']:.2f} m "
              f"(lap {s['track_len_m']:.2f} m)")
        print(f"  {'sim duration':<28s} {s['sim_duration_s']:.2f} s "
              f"({s['control_steps']} steps)")
        if s["lap_times_s"]:
            print(f"  {'lap times / speeds':<28s} "
                  f"{np.array2string(np.array(s['lap_times_s']), precision=2)} s / "
                  f"{np.array2string(np.array(s['lap_speeds_mps']), precision=2)} m/s")
        print(f"  {'min margin (crash < 0)':<28s} {s['margin_min_m']:+.4f} m")
        print(f"  {'guard 2 active':<28s} {s['guard2_active_steps']} steps "
              f"({100*s['guard2_active_fraction']:.1f}%)")
        for label, value in extra:
            print(f"  {label:<28s} {value}")
        return s
