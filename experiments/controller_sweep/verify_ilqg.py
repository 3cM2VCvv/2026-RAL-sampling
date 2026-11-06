"""Checks for the iLQG ancillary tracker.

Verifies Euler parity with the planner's rollout, that the solve reduces the
tracking cost and produces in-bounds controls, the feedback sign on an
integrator, and that the Riccati regularization stays finite on degenerate
systems and across the bicycle's kinematic/dynamic switch.

Usage:
    uv run python -m experiments.controller_sweep.verify_ilqg
"""

import os

import _env  # noqa: F401  sets up sys.path
import jax.numpy as jnp
import numpy as np
from immrax import System
from immrax_dynamic_bicycle_cartesian import DynamicBicycleCartesian
from immrax_planner import SampleBounds
from raceline_track import FIXTURE_DIR, RacelineTrack
from tube_ancillary import ILQGTracker
from tube_mppi import JaxMPPI

DT = 0.02
HORIZON = 30


class _StubParams:
    lf = 0.125
    lr = 0.125
    Iz = 0.03
    m = 1.5


class _StubCar:
    params = _StubParams()

    def init_param(self):
        pass


class _ConstantCost:
    def evaluate_rollout_cost(self, state, traj, u):
        return jnp.asarray(0.0, dtype=jnp.float32)

    def evaluate_trajectory_cost(self, state, traj):
        return jnp.asarray(0.0, dtype=jnp.float32)


class _DoubleIntegrator(System):
    evolution = "continuous"

    def __init__(self):
        super().__init__(evolution="continuous", xlen=2)
        self.wlen = 0

    def f(self, t, x, u, w):
        return jnp.array([x[1], u[0]])


class _StaticSystem(System):
    evolution = "continuous"

    def __init__(self):
        super().__init__(evolution="continuous", xlen=2)
        self.wlen = 0

    def f(self, t, x, u, w):
        return jnp.zeros(2)


def _track():
    return RacelineTrack(fixture_path=os.path.join(FIXTURE_DIR, "saved.npz"))


def _state(track):
    return np.array(
        [
            track.raceline_points[0, 0],
            track.raceline_points[1, 0],
            float(track.raceline_headings[0]),
            0.5,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )


def check_euler_parity(track):
    dynamics = DynamicBicycleCartesian(_StubCar())
    bounds = [SampleBounds(-0.45, 0.45, 0.45), SampleBounds(-1.0, 1.0, 1.0)]
    tracker = ILQGTracker(dynamics, bounds, horizon=8, dt=DT)
    planner = JaxMPPI(
        dynamics,
        _ConstantCost(),
        control_bounds=bounds,
        control_std=(0.1, 0.05),
        initial_controls=np.zeros((HORIZON, 2), dtype=np.float32),
        num_samples=4,
        planning_horizon=HORIZON,
        planning_dt=DT,
        param_lambda=1.0,
    )

    x = jnp.asarray(_state(track))
    u = jnp.asarray([0.02, 0.4], dtype=jnp.float32)
    traj = planner.rollout(x, jnp.tile(u, (HORIZON, 1)))
    expected = np.asarray(traj.ys[1])
    got = np.asarray(tracker._step(x, u))
    err = np.max(np.abs(expected - got))
    assert err < 1e-5, f"Euler parity error {err:.3e}"
    print(f"Euler parity: max abs error {err:.3e}")


def check_double_integrator():
    tracker = ILQGTracker(
        _DoubleIntegrator(),
        [SampleBounds(-2.0, 0.5, 2.0)],
        horizon=8,
        dt=0.1,
        track_weights=(1.0, 1.0),
        control_weights=(0.1,),
        iterations=8,
    )
    x_ref = jnp.zeros((9, 2), dtype=jnp.float32)
    u_ref = jnp.zeros((8, 1), dtype=jnp.float32)
    x0 = jnp.asarray([0.3, 0.0], dtype=jnp.float32)

    u0, info = tracker.solve(x0, x_ref, u_ref)
    u0 = float(u0[0])
    history = np.asarray(info["cost_history"])
    assert np.isfinite(history).all(), history
    assert history[-1] <= history[0] + 1e-5, history
    assert -2.0 - 1e-6 <= u0 <= 2.0 + 1e-6, u0
    assert u0 < 0.0, f"feedback sign wrong: {u0}"
    print(f"double integrator: u0 {u0:.3f}, cost {history[0]:.4f} -> {history[-1]:.4f}")


def check_regularization_stays_finite():
    tracker = ILQGTracker(
        _StaticSystem(),
        [SampleBounds(-1.0, 0.5, 1.0)],
        horizon=6,
        dt=0.1,
        track_weights=(1.0, 1.0),
        control_weights=(1e-8,),
        iterations=4,
    )
    x_ref = jnp.zeros((7, 2), dtype=jnp.float32)
    u_ref = jnp.zeros((6, 1), dtype=jnp.float32)
    u0, info = tracker.solve(jnp.asarray([0.5, -0.2]), x_ref, u_ref)
    assert np.all(np.isfinite(np.asarray(u0))), u0
    assert np.all(np.isfinite(np.asarray(info["mu_history"]))), info["mu_history"]
    print("degenerate system: gains and mu remain finite")


def check_bicycle_solve(track):
    dynamics = DynamicBicycleCartesian(_StubCar())
    bounds = [SampleBounds(-0.45, 0.45, 0.45), SampleBounds(-1.0, 1.0, 1.0)]
    tracker = ILQGTracker(dynamics, bounds, horizon=15, dt=DT, iterations=5)

    for v in (0.5, 0.05):
        x = jnp.asarray([0.0, 0.0, 0.0, v, 0.0, 0.0], dtype=jnp.float32)
        a, b = tracker._linearize(x, jnp.zeros(2, dtype=jnp.float32))
        assert np.all(np.isfinite(np.asarray(a))) and np.all(np.isfinite(np.asarray(b)))
    print("bicycle: linearization finite at v=0.5 and v=0.05")

    x0 = jnp.asarray(_state(track))
    u_ref = jnp.tile(jnp.asarray([0.02, 0.4], dtype=jnp.float32), (15, 1))
    x_ref = tracker._rollout(x0, u_ref)
    x_perturbed = x0.at[1].add(0.05)

    u0, info = tracker.solve(x_perturbed, x_ref, u_ref)
    history = np.asarray(info["cost_history"])
    assert np.all(np.isfinite(np.asarray(u0))), u0
    assert -0.45 - 1e-6 <= float(u0[0]) <= 0.45 + 1e-6, float(u0[0])
    assert -1.0 - 1e-6 <= float(u0[1]) <= 1.0 + 1e-6, float(u0[1])
    assert history[-1] <= history[0] + 1e-5, history
    print(f"bicycle solve: cost {history[0]:.4f} -> {history[-1]:.4f}, u0 {np.round(u0, 4)}")


def main():
    track = _track()
    check_euler_parity(track)
    check_double_integrator()
    check_regularization_stays_finite()
    check_bicycle_solve(track)
    print("all iLQG checks passed")


if __name__ == "__main__":
    main()
