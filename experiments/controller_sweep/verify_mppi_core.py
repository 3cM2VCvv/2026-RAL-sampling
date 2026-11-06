"""Checks for the vanilla ``JaxMPPI`` core and its paper-mode update.

Verifies that the default update is unchanged, that the unbounded-plan mode uses
raw sampling noise (so the plan may leave the control limits), that rollouts
always receive clamped controls, and that the noiseless plan evaluation behaves
as Algorithm 2 expects.

Usage:
    uv run python -m experiments.controller_sweep.verify_mppi_core
"""

import os

import _env  # noqa: F401  sets up sys.path
import jax
import jax.numpy as jnp
import numpy as np
from immrax_dynamic_bicycle_cartesian import DynamicBicycleCartesian
from immrax_planner import SampleBounds
from raceline_track import FIXTURE_DIR, RacelineTrack
from reachset_cost import (
    BOUNDARY_COST_WEIGHT,
    COLLISION_COST_WEIGHT,
    PointTrajectoryCost,
)
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


class _ConstantCost:
    def evaluate_rollout_cost(self, state, traj, u):
        return jnp.asarray(0.0, dtype=jnp.float32)

    def evaluate_trajectory_cost(self, state, traj):
        return jnp.asarray(0.0, dtype=jnp.float32)


class _ControlBoundCheckCost:
    def __init__(self, low, high):
        self.low = jnp.asarray(low, dtype=jnp.float32)
        self.high = jnp.asarray(high, dtype=jnp.float32)

    def evaluate_rollout_cost(self, state, traj, u):
        violation = jnp.maximum(u - self.high, 0.0) + jnp.maximum(self.low - u, 0.0)
        return jnp.sum(violation)

    def evaluate_trajectory_cost(self, state, traj):
        return jnp.asarray(0.0, dtype=jnp.float32)


def _track():
    return RacelineTrack(fixture_path=os.path.join(FIXTURE_DIR, "saved.npz"))


def _bounds():
    return [SampleBounds(-0.45, 0.45, 0.45), SampleBounds(-1.0, 1.0, 1.0)]


def _planner(track, *, bounded_plan=True, cost=None, control_std=(0.1, 0.05)):
    if cost is None:
        cost = PointTrajectoryCost(track, HORIZON, DT, control_diff_weight=0.0)
    return JaxMPPI(
        DynamicBicycleCartesian(_StubCar()),
        cost,
        control_bounds=_bounds(),
        control_std=control_std,
        initial_controls=np.zeros((HORIZON, 2), dtype=np.float32),
        num_samples=NUM_SAMPLES,
        planning_horizon=HORIZON,
        planning_dt=DT,
        param_lambda=10.0,
        prng_seed=0,
        bounded_plan=bounded_plan,
    )


def _state(track):
    return np.array(
        [
            track.raceline_points[0, 0],
            track.raceline_points[1, 0],
            float(track.raceline_headings[0]),
            1.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )


def _reference_update(planner, state, plan, key):
    """The pre-refactor update body, kept verbatim as the parity oracle."""
    noise_shape = (planner.num_samples, planner.planning_horizon, planner.sigma.shape[0])
    noise = jax.random.normal(key, noise_shape) * planner.sigma
    sampled = jnp.clip(plan[None] + noise, planner.u_min, planner.u_max)
    rollouts = jax.vmap(planner.rollout, in_axes=(None, 0))(state, sampled)
    costs = jax.vmap(planner.rollout_cost.evaluate_rollout_cost, in_axes=(None, 0, 0))(
        state, rollouts, sampled
    )
    weights = jnp.exp(-(costs - jnp.min(costs)) / planner.param_lambda)
    weights = weights / jnp.sum(weights)
    applied = sampled - plan[None]
    updated = jnp.clip(
        plan + jnp.sum(weights[:, None, None] * applied, axis=0),
        planner.u_min,
        planner.u_max,
    )
    return updated, weights, costs


def check_default_parity(track):
    planner = _planner(track)
    x0 = jnp.asarray(_state(track))
    plan = jnp.asarray(np.linspace([0.0, 0.4], [0.02, 0.5], HORIZON, np.float32))
    key = jax.random.key(3)

    expected = _reference_update(planner, x0, plan, key)
    actual = planner.update(x0, plan, key)
    for name, ref, got in zip(("plan", "weights", "costs"), expected, actual):
        err = float(jnp.max(jnp.abs(ref - got)))
        assert np.allclose(
            np.asarray(ref), np.asarray(got), rtol=1e-5, atol=1e-3
        ), f"default {name} differs from pre-refactor body: {err:.3e}"
    print("default update: identical to pre-refactor body")


def check_raw_noise_update(track):
    planner = _planner(
        track, bounded_plan=False, cost=_ConstantCost(), control_std=(5.0, 5.0)
    )
    x0 = jnp.asarray(_state(track))
    plan = jnp.asarray(np.tile([0.45, 1.0], (HORIZON, 1)), dtype=np.float32)
    key = jax.random.key(7)

    updated, weights, _ = planner.update(x0, plan, key)
    assert np.isclose(float(jnp.sum(weights)), 1.0, atol=1e-5)

    noise = jax.random.normal(key, (NUM_SAMPLES, HORIZON, 2)) * planner.sigma.astype(
        jnp.float32
    )
    cross = planner.control_cost_lambda * jnp.sum(
        plan[None] * noise / (planner.sigma.astype(jnp.float32) ** 2 + 1e-12),
        axis=(1, 2),
    )
    expected_weights = jnp.exp(-(cross - jnp.min(cross)) / planner.param_lambda)
    expected_weights = expected_weights / jnp.sum(expected_weights)
    assert np.allclose(
        np.asarray(weights), np.asarray(expected_weights), atol=1e-5
    ), "paper weights are not the lambda*u^T Sigma^-1 eps tilt"

    expected = jnp.clip(
        plan + jnp.sum(expected_weights[:, None, None] * noise, axis=0),
        planner.u_min,
        planner.u_max,
    )
    err = float(jnp.max(jnp.abs(updated - expected)))
    assert err < 1e-3, f"raw-noise update mismatch: {err:.3e}"
    assert np.all(np.asarray(updated) <= np.asarray(planner.u_max) + 1e-6)
    assert np.all(np.asarray(updated) >= np.asarray(planner.u_min) - 1e-6)

    # The raw-noise deviation must differ from the clipped-deviation update when
    # samples saturate at the limits.
    bounded = _planner(
        track, bounded_plan=True, cost=_ConstantCost(), control_std=(5.0, 5.0)
    )
    clipped, _, _ = bounded.update(x0, plan, key)
    assert not np.allclose(np.asarray(updated), np.asarray(clipped)), (
        "paper update matches the clipped-deviation update"
    )
    print("paper update: raw-noise mean projected onto the control limits")


def check_rollouts_are_clamped(track):
    cost = _ControlBoundCheckCost([-0.45, -1.0], [0.45, 1.0])
    bounds = [SampleBounds(-0.45, 0.45, 0.45), SampleBounds(-1.0, 1.0, 1.0)]
    planner = JaxMPPI(
        DynamicBicycleCartesian(_StubCar()),
        cost,
        control_bounds=bounds,
        control_std=(0.1, 0.05),
        initial_controls=np.zeros((HORIZON, 2), dtype=np.float32),
        num_samples=NUM_SAMPLES,
        planning_horizon=HORIZON,
        planning_dt=DT,
        param_lambda=10.0,
        prng_seed=0,
        bounded_plan=True,
    )
    plan = jnp.full((HORIZON, 2), 0.43, dtype=jnp.float32)
    _, _, costs = planner.update(jnp.asarray(_state(track)), plan, jax.random.key(1))
    assert np.allclose(np.asarray(costs), 0.0), "rollout received unclamped controls"
    print("rollouts: the cost receives g(u) inside the bounds")


def check_evaluate_plan(track):
    planner = _planner(track)
    x0 = jnp.asarray(_state(track))
    plan = jnp.asarray(np.linspace([0.0, 0.4], [0.02, 0.5], HORIZON, np.float32))
    cost, ys = planner.evaluate_plan(x0, plan)
    assert ys.shape == (HORIZON + 1, 6), ys.shape
    assert float(cost) > 0.0
    assert np.all(np.isfinite(np.asarray(ys)))

    other = planner.evaluate_plan(x0, plan.at[:, 0].add(0.2))
    assert not np.isclose(float(cost), float(other[0])), "plan change did not change cost"
    print(f"evaluate_plan: cost {float(cost):.3f}, shape {ys.shape}")


def check_constants():
    assert float(COLLISION_COST_WEIGHT) == 1e9
    assert float(BOUNDARY_COST_WEIGHT) == 10.0
    default = PointTrajectoryCost.__init__.__defaults__
    assert default[-2] == 1e1 and default[-1] == 5e0, default
    print("constants: named weights match the previous literals")


def main():
    track = _track()
    check_constants()
    check_default_parity(track)
    check_raw_noise_update(track)
    check_rollouts_are_clamped(track)
    check_evaluate_plan(track)
    print("all MPPI core checks passed")


if __name__ == "__main__":
    main()
