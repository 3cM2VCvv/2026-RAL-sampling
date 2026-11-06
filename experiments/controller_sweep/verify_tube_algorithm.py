"""Checks for the Tube-MPPI Algorithm 2 bookkeeping and the paper cost.

Uses a one-dimensional linear system with zero sampling noise so the plan
buffers are observable and a stubbed noiseless cost so the accept/reject branch
is exact.

Usage:
    uv run python -m experiments.controller_sweep.verify_tube_algorithm
"""

import os

import _env  # noqa: F401  sets up sys.path
import jax
import jax.numpy as jnp
import numpy as np
from immrax import System
from immrax_dynamic_bicycle_cartesian import DynamicBicycleCartesian
from immrax_planner import SampleBounds
from raceline_track import FIXTURE_DIR, RacelineTrack
from reachset_cost import (
    BOUNDARY_COST_WEIGHT,
    COLLISION_COST_WEIGHT,
    PointTrajectoryCost,
    SigmaWeightedControlCost,
)
from tube_mppi import JaxMPPI, TubeJaxMPPI
from tube_mppi_controller import TubeMPPI

HORIZON = 4
NUM_SAMPLES = 4
DT = 0.1


class _LinearSystem(System):
    evolution = "continuous"

    def __init__(self):
        super().__init__(evolution="continuous", xlen=1)
        self.wlen = 0

    def f(self, t, x, u, w):
        return u


class _ConstantCost:
    def evaluate_rollout_cost(self, state, traj, u):
        return jnp.asarray(0.0, dtype=jnp.float32)

    def evaluate_trajectory_cost(self, state, traj):
        return jnp.asarray(0.0, dtype=jnp.float32)


class _StubParams:
    lf = 0.125
    lr = 0.125
    Iz = 0.03
    m = 1.5


class _StubCar:
    params = _StubParams()

    def init_param(self):
        pass


def _planner(accept_margin):
    return TubeJaxMPPI(
        _LinearSystem(),
        _ConstantCost(),
        control_bounds=[SampleBounds(-1.0, 0.1, 1.0)],
        control_std=(0.0,),
        initial_controls=np.zeros((HORIZON, 1), dtype=np.float32),
        accept_margin=accept_margin,
        num_samples=NUM_SAMPLES,
        planning_horizon=HORIZON,
        planning_dt=DT,
        param_lambda=1.0,
        prng_seed=0,
    )


def _stub_plan_cost(planner, nominal_value=0.3, actual_value=0.7):
    """Noiseless cost = first state component; next nominal state = 2x start."""

    def evaluate_plan(state, plan):
        cost = state[0]
        traj = jnp.full((HORIZON + 1, 1), 2.0 * state[0], dtype=jnp.float32)
        return cost, traj

    planner.evaluate_plan = evaluate_plan
    planner.nominal_plan = jnp.full((HORIZON, 1), nominal_value, dtype=jnp.float32)
    planner.actual_plan = jnp.full((HORIZON, 1), actual_value, dtype=jnp.float32)
    planner.plan = planner.nominal_plan
    return planner


def _run(planner, actual, nominal):
    planner.nominal_state = jnp.asarray([nominal], dtype=jnp.float32)
    return planner.step(np.asarray([actual], dtype=np.float32))


def check_acceptance():
    # accept: actual cost -3 <= nominal cost 7 + 0.5
    planner = _stub_plan_cost(_planner(accept_margin=0.5))
    _, info = _run(planner, actual=-3.0, nominal=7.0)
    assert info["accepted"] is True
    assert np.isclose(float(planner.nominal_state[0]), -6.0), planner.nominal_state
    assert np.allclose(np.asarray(planner.nominal_plan), 0.7)
    assert np.allclose(np.asarray(planner.actual_plan), 0.7)
    print("accept: re-anchored to the actual state and the actual plan")

    # reject: actual cost 10 > nominal cost 0 + 0.5
    planner = _stub_plan_cost(_planner(accept_margin=0.5))
    _, info = _run(planner, actual=10.0, nominal=0.0)
    assert info["accepted"] is False
    assert np.isclose(float(planner.nominal_state[0]), 0.0), planner.nominal_state
    assert np.allclose(np.asarray(planner.nominal_plan), 0.3)
    assert np.allclose(np.asarray(planner.actual_plan), 0.7)
    print("reject: nominal state and nominal plan retained")

    # equality is accepted (<= in the paper)
    planner = _stub_plan_cost(_planner(accept_margin=0.5))
    _, info = _run(planner, actual=0.5, nominal=0.0)
    assert info["accepted"] is True
    print("boundary: S_act == S_nom + K is accepted")


def check_plan_shapes():
    planner = _stub_plan_cost(_planner(accept_margin=0.5))
    control0, info = _run(planner, actual=0.0, nominal=1.0)
    assert control0.shape == (1,)
    assert planner.plan.shape == (HORIZON, 1)
    assert info["nominal_trajectory"].shape == (HORIZON + 1, 1)
    assert info["nominal_controls"].shape == (HORIZON, 1)
    shifted = TubeJaxMPPI.shift_plan(jnp.arange(HORIZON).reshape(HORIZON, 1))
    assert np.allclose(np.asarray(shifted[:, 0]), [1, 2, 3, 3])
    print("bookkeeping: plans, trajectories and shifts have the expected shapes")


def check_accept_margin_defaults():
    class _Dummy:
        tube_accept_margin = None

    assert TubeMPPI._resolve_accept_margin(_Dummy()) == min(
        COLLISION_COST_WEIGHT, BOUNDARY_COST_WEIGHT
    )
    _Dummy.tube_accept_margin = 3.5
    assert TubeMPPI._resolve_accept_margin(_Dummy()) == 3.5
    print("accept margin: derived 10.0, explicit override wins")


def _track():
    return RacelineTrack(fixture_path=os.path.join(FIXTURE_DIR, "saved.npz"))


def check_sigma_weighted_cost():
    track = _track()
    sigma = (0.1, 0.05)
    lam = 2.0
    bounds = [SampleBounds(-0.45, 0.45, 0.45), SampleBounds(-1.0, 1.0, 1.0)]
    cost = SigmaWeightedControlCost(
        track, 10, 0.02, control_sigma=sigma, control_lambda=lam
    )
    point = PointTrajectoryCost(
        track, 10, 0.02, control_diff_weight=0.0, control_weight=0.0
    )
    planner = JaxMPPI(
        DynamicBicycleCartesian(_StubCar()),
        cost,
        control_bounds=bounds,
        control_std=sigma,
        initial_controls=np.zeros((10, 2), dtype=np.float32),
        num_samples=4,
        planning_horizon=10,
        planning_dt=0.02,
    )
    x0 = jnp.asarray(
        [
            track.raceline_points[0, 0],
            track.raceline_points[1, 0],
            float(track.raceline_headings[0]),
            1.0,
            0.0,
            0.0,
        ],
        dtype=jnp.float32,
    )
    u = jnp.asarray(np.linspace([0.0, 0.4], [0.02, 0.5], 10, np.float32))
    traj = planner.rollout(x0, u)

    state_cost = float(point.evaluate_rollout_cost(x0, traj, u))
    expected = state_cost + lam * float(
        jnp.sum(jnp.square(u) * jnp.asarray(1.0 / jnp.asarray(sigma) ** 2))
    )
    got = float(cost.evaluate_rollout_cost(x0, traj, u))
    assert np.isclose(got, expected, rtol=1e-4), (got, expected)
    assert np.allclose(np.asarray(cost.inv_sigma_sq), 1.0 / np.square(sigma))
    print(f"paper cost: {got:.3f} = state {state_cost:.3f} + Sigma-weighted control")


def main():
    check_acceptance()
    check_plan_shapes()
    check_accept_margin_defaults()
    check_sigma_weighted_cost()
    print("all Tube-MPPI algorithm checks passed")


if __name__ == "__main__":
    main()
