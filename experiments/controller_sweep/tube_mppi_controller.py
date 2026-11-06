"""Tube-MPPI controller (Williams et al. 2018) on the dynamic bicycle.

The nominal controller runs MPPI from a disturbance-free nominal state.  A second
MPPI solve from the actual state is adopted (Algorithm 2) when its noiseless cost
is within ``tube_accept_margin`` of the nominal solution's cost.  An iLQG
ancillary controller tracks the accepted nominal trajectory and supplies the
applied control.

The cost follows the paper: the ``PointTrajectoryCost`` state terms (including
the sparse collision indicator) plus ``lambda * u^T Sigma^-1 u``, with ``Sigma``
equal to the planner's sampling covariance.
"""

import jax
import jax.numpy as jnp
import numpy as np
from mppi_controller import MPPI
from reachset_cost import (
    BOUNDARY_COST_WEIGHT,
    COLLISION_COST_WEIGHT,
    SigmaWeightedControlCost,
)
from tube_ancillary import ILQGTracker
from tube_mppi import TubeJaxMPPI


class TubeMPPI(MPPI):
    """Tube-MPPI over the car's dynamic-bicycle model and state cost."""

    def __init__(self, car, config, *, dynamics=None, cost=None):
        # Defaults are set before ``super().__init__`` so XML attributes win.
        self.tube_accept_margin = None  # None -> min critical cost weight
        self.tube_control_cost_lambda = 0.03  # weight on lambda*u^T Sigma^-1 u
        self.tube_ancillary_horizon = 0  # 0 -> planning_horizon // 2
        self.tube_ancillary_iterations = 1
        self.tube_ancillary_mu = 1e-2
        self.tube_ancillary_mu_factor = 5.0
        self.tube_ancillary_q_scale = 1.0
        self.tube_ancillary_r_scale = 1.0
        self.tube_ancillary_terminal_scale = 10.0

        super().__init__(car, config, dynamics=dynamics, cost=cost)

        self.ancillary_horizon = (
            int(self.tube_ancillary_horizon) or self.planning_horizon // 2
        )
        self.ancillary = ILQGTracker(
            self.dynamics,
            [self.steering_bounds, self.throttle_bounds],
            horizon=self.ancillary_horizon,
            dt=self.planning_dt,
            q_scale=float(self.tube_ancillary_q_scale),
            r_scale=float(self.tube_ancillary_r_scale),
            terminal_scale=float(self.tube_ancillary_terminal_scale),
            iterations=int(self.tube_ancillary_iterations),
            mu_init=float(self.tube_ancillary_mu),
            mu_factor=float(self.tube_ancillary_mu_factor),
        )
        self.last_info = {}
        self._ancillary_info = {}
        self._warm_ancillary()

    def _resolve_accept_margin(self):
        """Paper's threshold: smallest safety-critical cost weight."""
        if self.tube_accept_margin is not None:
            return float(self.tube_accept_margin)
        return min(COLLISION_COST_WEIGHT, BOUNDARY_COST_WEIGHT)

    def _make_cost(self):
        return SigmaWeightedControlCost(
            self.track,
            self.planning_horizon,
            self.planning_dt,
            control_sigma=self._input_std(),
            control_lambda=float(self.tube_control_cost_lambda),
            centering_weight_factor=self.centering_weight_factor,
            velocity_scale_factor=self.velocity_scale_factor,
        )

    def _build_planner(self):
        return TubeJaxMPPI(
            self.dynamics,
            self.rollout_cost,
            control_bounds=[self.steering_bounds, self.throttle_bounds],
            control_std=self._input_std(),
            initial_controls=self._stanley_plan(),
            accept_margin=self._resolve_accept_margin(),
            num_samples=self.num_samples,
            planning_horizon=self.planning_horizon,
            planning_dt=self.planning_dt,
            param_lambda=float(self.mppi_param_lambda),
            prng_seed=int(self.prng_seed),
            control_cost_lambda=float(self.tube_control_cost_lambda),
        )

    def _warm_ancillary(self):
        """Compile the ancillary solve against the planner's reference shapes."""
        state = jnp.asarray(self.car.state[:6], dtype=jnp.float32)
        h = self.ancillary_horizon
        x_ref = jnp.zeros((h + 1, 6), dtype=jnp.float32)
        u_ref = jnp.zeros((h, 2), dtype=jnp.float32)
        u0, _ = self.ancillary.solve(state, x_ref, u_ref)
        jax.block_until_ready(u0)

    def _compute_control(self, state):
        state_j = jnp.asarray(state, dtype=jnp.float32)
        nominal_control, info = self.planner.step(state_j)

        h = self.ancillary_horizon
        u0, anc_info = self.ancillary.solve(
            state_j,
            info["nominal_trajectory"][: h + 1],
            info["nominal_controls"][:h],
        )
        u0 = np.asarray(jax.device_get(u0))
        if not np.all(np.isfinite(u0)):
            u0 = np.asarray(nominal_control)

        self.last_info = info
        self._ancillary_info = anc_info
        return u0
