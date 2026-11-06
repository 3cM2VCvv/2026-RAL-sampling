"""Consistency checks for the vanilla JAX MPPI core used by ``TubeMPPI``.

Kept separate from the implementation. Verifies that the rollout matches the
JAX ``DynamicBicycleCartesian`` model it wraps, that the update is a proper
weighted average, and that a zero-noise update is a no-op.

Usage:
    uv run python -m experiments.controller_sweep.verify_tube_mppi
"""

import os

import _env  # noqa: F401  sets up sys.path
import jax
import jax.numpy as jnp
import numpy as np
from immrax_dynamic_bicycle_cartesian import DynamicBicycleCartesian
from immrax_planner import SampleBounds
from raceline_track import FIXTURE_DIR, RacelineTrack
from reachset_cost import PointTrajectoryCost
from tube_mppi import JaxMPPI

DT = 0.02
HORIZON = 30
NUM_SAMPLES = 16


class _StubParams:
    lf = 0.125
    lr = 0.125
    Iz = 0.03
    m = 1.5


class _StubCar:
    params = _StubParams()

    def init_param(self):
        pass


def _track():
    return RacelineTrack(fixture_path=os.path.join(FIXTURE_DIR, "saved.npz"))


def _planner(track, control_std=(0.1, 0.05), seed=0):
    predictor = DynamicBicycleCartesian(_StubCar())
    cost = PointTrajectoryCost(track, HORIZON, DT, control_diff_weight=0.0)
    return JaxMPPI(
        predictor,
        cost,
        control_bounds=[
            SampleBounds(-0.45, 0.45, 0.45),
            SampleBounds(-1.0, 1.0, 1.0),
        ],
        control_std=control_std,
        initial_controls=np.zeros((HORIZON, 2), dtype=np.float32),
        num_samples=NUM_SAMPLES,
        planning_horizon=HORIZON,
        planning_dt=DT,
        param_lambda=10.0,
        prng_seed=seed,
    )


def _state(track):
    heading = float(track.raceline_headings[0])
    return np.array(
        [
            track.raceline_points[0, 0],
            track.raceline_points[1, 0],
            heading,
            1.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )


def check_rollout(track):
    planner = _planner(track)
    x0 = _state(track)
    controls = np.zeros((HORIZON, 2), dtype=np.float32)
    controls[:, 0] = 0.02
    controls[:, 1] = 0.4

    traj = planner.rollout(jnp.asarray(x0), jnp.asarray(controls))
    # Raw trajectories are padded with inf; the first HORIZON + 1 states are valid.
    ys = np.asarray(traj.ys)[: HORIZON + 1]
    assert ys.shape == (HORIZON + 1, 6), ys.shape

    # Manual explicit Euler with the same zero-disturbance model.
    x = x0
    expected = [x]
    for u in controls:
        x = (
            np.asarray(
                planner.predictor.f(0.0, jnp.asarray(x), jnp.asarray(u), jnp.zeros(2))
            )
            * DT
            + x
        )
        expected.append(x)
    err = np.max(np.abs(ys - np.asarray(expected)))
    assert err < 1e-4, f"rollout mismatch: {err:.3e}"
    print(f"rollout: max abs error {err:.3e}")


def check_update(track):
    planner = _planner(track)
    x0 = jnp.asarray(_state(track))
    updated, weights, costs = planner.update(x0, planner.plan, jax.random.key(0))

    assert np.isclose(float(jnp.sum(weights)), 1.0, atol=1e-5)
    assert np.all(np.isfinite(costs))
    assert np.all(np.asarray(updated) >= np.asarray(planner.u_min) - 1e-6)
    assert np.all(np.asarray(updated) <= np.asarray(planner.u_max) + 1e-6)
    ess = float(jnp.sum(weights**2))
    print(
        f"update: cost [{float(jnp.min(costs)):.3f}, "
        f"{float(jnp.max(costs)):.3f}], ESS {1.0 / ess:.1f}/{NUM_SAMPLES}"
    )


def check_zero_noise_is_noop(track):
    planner = _planner(track, control_std=(0.0, 0.0))
    plan = np.linspace([0.0, 0.4], [0.02, 0.5], HORIZON, dtype=np.float32)
    planner.set_plan(plan)
    updated, weights, _ = planner.update(
        jnp.asarray(_state(track)), planner.plan, jax.random.key(1)
    )
    err = float(jnp.max(jnp.abs(updated - planner.plan)))
    assert err < 1e-6, f"zero-noise update moved the plan: {err:.3e}"
    assert np.allclose(np.asarray(weights), 1.0 / NUM_SAMPLES, atol=1e-6)
    print(f"zero noise: plan unchanged (max error {err:.3e})")


def check_step_shifts(track):
    planner = _planner(track)
    planner.set_plan(np.linspace([0.0, 0.4], [0.02, 0.5], HORIZON, dtype=np.float32))
    control0, info = planner.step(_state(track))

    assert control0.shape == (2,)
    assert info["ess"] > 0.0
    assert planner.plan.shape == (HORIZON, 2)
    assert np.all(np.isfinite(control0))
    print(f"step: control {np.round(control0, 4)}, ESS {info['ess']:.1f}")


def check_cost_weights(track):
    """Default control weights are unchanged; ``control_diff_weight=0`` drops it."""
    planner = _planner(track)
    x0 = jnp.asarray(_state(track))
    u = jnp.asarray(np.linspace([0.0, 0.4], [0.02, 0.5], HORIZON, dtype=np.float32))
    traj = planner.rollout(x0, u)

    state_cost = float(
        PointTrajectoryCost(track, HORIZON, DT).evaluate_trajectory_cost(x0, traj)
    )
    default = float(
        PointTrajectoryCost(track, HORIZON, DT).evaluate_rollout_cost(x0, traj, u)
    )
    nodiff = float(
        PointTrajectoryCost(
            track, HORIZON, DT, control_diff_weight=0.0
        ).evaluate_rollout_cost(x0, traj, u)
    )

    diff_term = 1e1 * float(jnp.sum(jnp.square(jnp.diff(u, axis=0))))
    u_term = 5e0 * float(jnp.sum(jnp.square(u)))
    assert np.isclose(default, state_cost + diff_term + u_term, rtol=1e-5)
    assert np.isclose(nodiff, state_cost + u_term, rtol=1e-5)
    print(f"cost: default={default:.3f} (diff={diff_term:.3f}), no-diff={nodiff:.3f}")


def main():
    track = _track()
    check_rollout(track)
    check_update(track)
    check_zero_noise_is_noop(track)
    check_step_shifts(track)
    check_cost_weights(track)
    print("all vanilla MPPI core checks passed")


if __name__ == "__main__":
    main()
