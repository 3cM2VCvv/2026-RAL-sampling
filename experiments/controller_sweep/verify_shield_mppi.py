"""Parity checks for the Shield-MPPI car adapter.

Verifies that the PyTorch dynamics and cost match the JAX model used by
``ImmraxController``, that the DCBF width is the tightest one-sided clearance,
and that rollouts and the local repair stay finite.  Kept separate from the
implementation.

Usage:
    uv run python -m experiments.controller_sweep.verify_shield_mppi
"""

import os
from types import SimpleNamespace

import _env  # noqa: F401  sets up sys.path
import jax
import jax.numpy as jnp
import numpy as np
import torch
from immrax.system.trajectory import RawDiscreteTrajectory
from immrax_dynamic_bicycle_cartesian import DynamicBicycleCartesian
from raceline_track import FIXTURE_DIR, RacelineTrack
from reachset_cost import PointTrajectoryCost
from shield_mppi_controller import ShieldMPPIController, _load_solver_class
from shield_mppi_cost import ImmraxRolloutCost, cbf_half_width
from shield_mppi_dynamics import DynamicBicycleTorch

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


def check_dynamics():
    sys_ref = DynamicBicycleCartesian(_StubCar())
    torch_model = DynamicBicycleTorch(_StubCar())

    rng = np.random.default_rng(0)
    n = 512
    x = np.column_stack(
        [
            rng.normal(0, 1, n),  # x
            rng.normal(0, 1, n),  # y
            rng.uniform(-np.pi, np.pi, n),  # heading
            rng.uniform(-0.05, 2.5, n),  # v_forward (both branches)
            rng.normal(0, 0.2, n),  # v_sideway
            rng.normal(0, 1.0, n),  # omega
        ]
    ).astype(np.float32)
    u = np.column_stack(
        [rng.uniform(-0.45, 0.45, n), rng.uniform(-1.0, 1.0, n)]
    ).astype(np.float32)
    w = np.zeros((n, 2), dtype=np.float32)

    ref = jax.vmap(sys_ref.f, in_axes=(None, 0, 0, 0))(0.0, x, u, w)
    got = torch_model.derivative(torch.as_tensor(x), torch.as_tensor(u))
    err = np.max(np.abs(np.asarray(got.detach()) - np.asarray(ref)))
    assert err < 1e-4, f"dynamics derivative mismatch: {err:.3e}"
    print(f"dynamics: max abs error {err:.3e}")


def _trajectory(n, seed=1, off_track=False):
    track = RacelineTrack(fixture_path=os.path.join(FIXTURE_DIR, "saved.npz"))
    rng = np.random.default_rng(seed)
    m = len(track.ss)
    idx = (np.arange(n) + 137) % m
    heading = track.raceline_headings[idx]
    lateral = rng.normal(0.0, 0.03, n)
    if off_track:
        lateral[n // 2] = 5.0

    ys = np.zeros((n, 6), dtype=np.float32)
    ys[:, 0] = track.raceline_points[0, idx] - np.sin(heading) * lateral
    ys[:, 1] = track.raceline_points[1, idx] + np.cos(heading) * lateral
    ys[:, 2] = heading
    ys[:, 3] = track.raceline_velocity[idx] + rng.normal(0.0, 0.05, n)
    return track, ys


def _ref_path_tensor(track):
    return torch.as_tensor(
        np.column_stack(
            [
                track.raceline_points.T,
                track.raceline_headings,
                track.raceline_velocity,
            ]
        ),
        dtype=torch.float32,
    )


def _make_solver(track, w_T):
    return _load_solver_class()(
        delta_t=DT,
        time_step=HORIZON,
        batch_size=8,
        sigma=torch.diag(torch.tensor([0.1, 0.1])),
        ref_track_path=_ref_path_tensor(track),
        param_lambda=1.0,
        param_alpha=0.0,
        cbf_cost_weight=1000.0,
        w_T=w_T,
        beta=0.5,
        device="cpu",
    )


def check_cost(off_track=False):
    track, ys = _trajectory(HORIZON + 1, off_track=off_track)
    ref_cost = PointTrajectoryCost(
        track,
        HORIZON,
        DT,
        centering_weight_factor=1.0,
        velocity_scale_factor=1.0,
    )
    ts = jnp.arange(HORIZON + 1) * DT
    traj = RawDiscreteTrajectory(ts, jnp.asarray(ys))
    ref = ref_cost._compute_cost_components(jnp.asarray(ys[0]), traj)

    cost = ImmraxRolloutCost(track, HORIZON, DT, device="cpu")
    got = cost.components(
        torch.as_tensor(ys[0]),
        torch.as_tensor(ys[None, 1:]),
    )

    for key in ("step_cost", "boundary_cost", "terminal_cost"):
        expected = float(ref[key])
        actual = float(got[key][0])
        assert np.isclose(actual, expected, rtol=1e-4, atol=1e-3), (
            f"{key}: torch={actual:.6f} jax={expected:.6f}"
        )

    print(
        "cost%s: step=%.3f boundary=%.3f terminal=%.3f"
        % (
            " (off-track)" if off_track else "",
            float(got["step_cost"][0]),
            float(got["boundary_cost"][0]),
            float(got["terminal_cost"][0]),
        )
    )
    return track


def check_cbf(track):
    half = cbf_half_width(track)
    assert half > 0.0
    assert np.all(track.raceline_left_boundary >= half)
    assert np.all(track.raceline_right_boundary >= half)

    solver = _make_solver(track, 2.0 * half)
    assert solver.dim_u == 2
    assert solver.u_prev.shape == (HORIZON, 2)
    assert solver.sigma_inv_diag.shape == (2,)

    ref = _ref_path_tensor(track)[0:1]
    on = torch.as_tensor(track.raceline_points[:, 0], dtype=torch.float32).view(1, 2)
    away = on + torch.tensor([[half + 0.1, 0.0]])
    h_on = float(solver.h(on, ref)[0])
    h_off = float(solver.h(away, ref)[0])
    assert h_on > 0.0
    assert h_off < 0.0
    print(f"cbf: half width {half:.4f} m, h_on={h_on:.4f}, h_off={h_off:.4f}")


def check_rollout_finite(track):
    model = DynamicBicycleTorch(_StubCar())
    rng = np.random.default_rng(2)
    k, n = 16, HORIZON
    x0 = torch.as_tensor(
        np.array(
            [
                track.raceline_points[0, 0],
                track.raceline_points[1, 0],
                track.raceline_headings[0],
                0.5,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
    )
    controls = torch.as_tensor(
        np.column_stack(
            [rng.uniform(-0.4, 0.4, (k, n)), rng.uniform(-1.0, 1.0, (k, n))]
        )
        .reshape(k, n, 2)
        .astype(np.float32)
    )
    states = model.rollout(x0, controls, DT)
    assert states.shape == (k, n, 6)
    assert torch.isfinite(states).all(), "non-finite state in rollout"
    print("rollout: finite")


def _plan_violation(solver, predict, x0, plan, ref_path, n):
    traj = predict(x0, plan.unsqueeze(0)).squeeze(0)
    xy = traj[:n, :2]
    h_next = solver.h(xy, ref_path[:n, :2])
    h_prev = torch.cat(
        [
            solver.h(x0[:2].unsqueeze(0), ref_path[:1, :2]),
            solver.h(xy[:-1], ref_path[: n - 1, :2]),
        ],
        dim=0,
    )
    return float(torch.clamp(-h_next + solver.alpha * h_prev, min=0.0).sum())


def check_repair(track):
    half = cbf_half_width(track)
    solver = _make_solver(track, 2.0 * half)
    heading = float(track.raceline_headings[0])
    offset = half + 0.2
    x0 = torch.as_tensor(
        np.array(
            [
                track.raceline_points[0, 0] - np.sin(heading) * offset,
                track.raceline_points[1, 0] + np.cos(heading) * offset,
                heading,
                0.5,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
    )
    solver.set_state(x0)

    stub = SimpleNamespace(
        repair_horizon=10,
        repair_steps=20,
        planning_horizon=HORIZON,
        planning_dt=DT,
        device=torch.device("cpu"),
        min_input=torch.tensor([-0.45, -1.0]),
        max_input=torch.tensor([0.45, 1.0]),
        dynamics=DynamicBicycleTorch(_StubCar()),
        mppi=solver,
    )
    stub.predict_trajectory = lambda initial, controls: stub.dynamics.rollout(
        initial, controls, DT
    )

    mean = torch.zeros((HORIZON, 2))
    mean[:, 0] = 0.45
    mean[:, 1] = 1.0
    mean_traj = stub.predict_trajectory(x0, mean.unsqueeze(0)).squeeze(0)
    ref_path = solver.get_nearest_waypoints(mean_traj.unsqueeze(0), 200).squeeze(0)
    before = _plan_violation(
        solver, stub.predict_trajectory, x0, mean, ref_path, stub.repair_horizon
    )
    safe = ShieldMPPIController._local_repair(stub, mean, x0)
    after = _plan_violation(
        solver, stub.predict_trajectory, x0, safe, ref_path, stub.repair_horizon
    )
    assert safe.shape == mean.shape
    assert torch.isfinite(safe).all()
    assert after <= before + 1e-6, f"repair increased violation: {before} -> {after}"
    print(f"repair: violation {before:.4f} -> {after:.4f}")


def main():
    check_dynamics()
    track = check_cost()
    check_cost(off_track=True)
    check_cbf(track)
    check_repair(track)
    check_rollout_finite(track)
    print("all Shield-MPPI adapter checks passed")


if __name__ == "__main__":
    main()
