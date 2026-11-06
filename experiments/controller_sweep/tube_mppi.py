"""MPPI planner cores over an ``immrax`` system, in JAX.

``JaxMPPI`` is vanilla MPPI: each update samples full control sequences around
the current plan, scores them, and replaces the plan with the softmax-weighted
mean.  ``bounded_plan=False`` selects the paper's update (raw-noise mean, Alg. 1
of Williams et al. 2018) and, by default, the paper's sample weighting (state
cost plus the ``lambda * u^T Sigma^-1 eps`` cross term).

``TubeJaxMPPI`` adds Algorithm 2: a nominal state and two warm-start plans, with
the actual-state solution adopted when its noiseless cost is within a threshold
of the nominal-state solution's cost.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from immrax_planner import _get_control_index


class JaxMPPI:
    """Receding-horizon MPPI planner.

    Parameters
    ----------
    predictor : immrax.System
        Continuous system, rolled out with explicit Euler at ``planning_dt``.
        Rollouts use the zero disturbance.
    rollout_cost : RolloutCostEvaluator
        Object exposing ``evaluate_rollout_cost(state, traj, u)``.
    control_bounds : list[SampleBounds]
        Per-channel limits; their length sets the control dimension.
    control_std : sequence[float]
        Per-step sampling std, one entry per control channel.
    initial_controls : array-like, shape (planning_horizon, control_dim)
        Initial nominal plan.
    bounded_plan : bool
        True: clip samples and the plan, and update with the post-clip deviation
        (vanilla).  False: update with the raw sampling noise (paper Alg. 1); the
        plan is still projected onto the limits so sampling stays informative.
    paper_weights : bool, optional
        Use the paper's sample cost (state cost plus the control-noise cross
        term).  Defaults to ``not bounded_plan``.
    control_cost_lambda : float, optional
        Weight of ``u^T Sigma^-1 u`` in the cross term.  Defaults to
        ``param_lambda`` (the paper ties both to one lambda).
    """

    def __init__(
        self,
        predictor,
        rollout_cost,
        control_bounds,
        control_std,
        initial_controls,
        *,
        num_samples: int = 1024,
        planning_horizon: int = 30,
        planning_dt: float = 0.02,
        param_lambda: float = 1.0,
        prng_seed: int = 0,
        bounded_plan: bool = True,
        paper_weights: bool | None = None,
        control_cost_lambda: float | None = None,
    ):
        self.predictor = predictor
        self.rollout_cost = rollout_cost
        self.num_samples = int(num_samples)
        self.planning_horizon = int(planning_horizon)
        self.planning_dt = float(planning_dt)
        self.param_lambda = float(param_lambda)
        self.bounded_plan = bool(bounded_plan)
        if paper_weights is None:
            paper_weights = not self.bounded_plan
        self.paper_weights = bool(paper_weights)
        self.control_cost_lambda = float(
            self.param_lambda if control_cost_lambda is None else control_cost_lambda
        )

        self.u_min = jnp.asarray(
            [bound.min for bound in control_bounds], dtype=jnp.float32
        )
        self.u_max = jnp.asarray(
            [bound.max for bound in control_bounds], dtype=jnp.float32
        )
        self.sigma = jnp.asarray(control_std, dtype=jnp.float32)
        self.plan = jnp.asarray(initial_controls, dtype=jnp.float32)

        self._key = jax.random.key(int(prng_seed))
        self._t0 = jnp.asarray(0.0)
        self._tf = jnp.asarray(self.planning_horizon * self.planning_dt)
        self._dt = jnp.asarray(self.planning_dt)
        self._zero_disturbance = jnp.zeros(
            getattr(predictor, "wlen", 0), dtype=jnp.float32
        )

    # ------------------------------------------------------------------
    # Rollouts
    # ------------------------------------------------------------------
    def _control_action(self, controls):
        def action(t, x):
            idx = _get_control_index(t, self.planning_dt, self.planning_horizon)
            return controls[idx]

        return action

    def rollout(self, x0, controls):
        """Roll ``x0`` through one control sequence under zero disturbance."""
        return self.predictor.compute_trajectory(
            self._t0,
            self._tf,
            x0,
            (self._control_action(controls), lambda t, x: self._zero_disturbance),
            dt=self._dt,
            solver="euler",
        )

    # ------------------------------------------------------------------
    # MPPI update
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnums=0)
    def update(self, state, plan, key):
        """One MPPI update.

        Returns the new plan, the sample weights, and the sample costs.
        """
        noise_shape = (
            self.num_samples,
            self.planning_horizon,
            self.sigma.shape[0],
        )
        noise = jax.random.normal(key, noise_shape) * self.sigma
        # Anchor the samples at the feasible control.  Keeping an out-of-bounds
        # anchor would clamp every sample to the same limit and flatten the
        # weights (no learning); the plan iterate itself is projected below.
        anchor = jnp.clip(plan, self.u_min, self.u_max)
        sampled = jnp.clip(anchor[None] + noise, self.u_min, self.u_max)

        rollouts = jax.vmap(self.rollout, in_axes=(None, 0))(state, sampled)

        if self.paper_weights:
            # Alg. 1: weights use the state cost plus the control-noise cross term
            # ``lambda * u^T Sigma^-1 eps``.  The plan's quadratic control cost is
            # common to every sample and enters through the update's fixed point.
            state_costs = jax.vmap(
                self.rollout_cost.evaluate_trajectory_cost, in_axes=(None, 0)
            )(state, rollouts)
            cross = self.control_cost_lambda * jnp.sum(
                anchor[None]
                * noise
                / (self.sigma[None, None, :] ** 2 + 1e-12),
                axis=(1, 2),
            )
            costs = state_costs + cross
            weight_costs = costs
        else:
            costs = jax.vmap(
                self.rollout_cost.evaluate_rollout_cost, in_axes=(None, 0, 0)
            )(state, rollouts, sampled)
            weight_costs = costs

        weights = jnp.exp(-(weight_costs - jnp.min(weight_costs)) / self.param_lambda)
        weights = weights / jnp.sum(weights)

        if self.bounded_plan:
            # Deviation from the applied (clipped) samples.
            delta = sampled - anchor[None]
            updated = jnp.clip(
                anchor + jnp.sum(weights[:, None, None] * delta, axis=0),
                self.u_min,
                self.u_max,
            )
        else:
            # Alg. 1 update: the mean moves by the raw sampling noise.  The
            # result is projected onto the limits (an out-of-bounds iterate
            # would freeze sampling at a clamped anchor).
            updated = jnp.clip(
                anchor + jnp.sum(weights[:, None, None] * noise, axis=0),
                self.u_min,
                self.u_max,
            )
        return updated, weights, costs

    @partial(jax.jit, static_argnums=0)
    def evaluate_plan(self, state, plan):
        """Noiseless cost ``S(g(U))`` of a plan and its state trajectory."""
        applied = jnp.clip(plan, self.u_min, self.u_max)
        traj = self.rollout(state, applied)
        cost = self.rollout_cost.evaluate_rollout_cost(state, traj, applied)
        return cost, traj.ys[: self.planning_horizon + 1]

    @staticmethod
    def shift_plan(plan):
        """Receding-horizon warm start: drop the first control, hold the last."""
        return jnp.vstack([plan[1:], plan[-1:]])

    def warmup(self, state):
        """Compile ``update`` without advancing the plan or PRNG state."""
        _, key = jax.random.split(self._key)
        updated, _, _ = self.update(
            jnp.asarray(state, dtype=jnp.float32), self.plan, key
        )
        jax.block_until_ready(updated)

    def step(self, state):
        """Advance the plan one control step and return its first control."""
        self._key, key = jax.random.split(self._key)
        updated, weights, costs = self.update(
            jnp.asarray(state, dtype=jnp.float32), self.plan, key
        )
        control0 = np.asarray(jax.device_get(updated[0]))

        # Receding horizon: shift the plan and hold the last control.
        self.plan = self.shift_plan(updated)

        info = {
            "costs": np.asarray(jax.device_get(costs)),
            "weights": np.asarray(jax.device_get(weights)),
            "best_cost": float(jax.device_get(jnp.min(costs))),
            "ess": float(jax.device_get(1.0 / jnp.sum(weights**2))),
        }
        return control0, info

    def set_plan(self, controls):
        """Replace the nominal plan."""
        self.plan = jnp.asarray(controls, dtype=jnp.float32)


class TubeJaxMPPI(JaxMPPI):
    """Tube-MPPI nominal planner (Williams et al. 2018, Algorithm 2).

    Maintains a disturbance-free nominal state and two warm-start buffers: the
    nominal-state plan and the actual-state plan.  Each tick solves MPPI from
    both states and adopts the actual-state solution when its noiseless cost is
    within ``accept_margin`` of the nominal-state solution's cost.
    """

    def __init__(self, *args, accept_margin: float, **kwargs):
        kwargs.setdefault("bounded_plan", False)
        super().__init__(*args, **kwargs)
        self.accept_margin = float(accept_margin)
        self.nominal_state = None
        self.nominal_plan = self.plan
        self.actual_plan = self.plan
        self._accept_count = 0
        self._step_count = 0

    @property
    def accept_rate(self):
        return self._accept_count / max(self._step_count, 1)

    @partial(jax.jit, static_argnums=0)
    def _solve_pair(self, state, nominal_state, nominal_plan, actual_plan, key):
        k_nom, k_act = jax.random.split(key)
        u_nom, w_nom, c_nom = self.update(nominal_state, nominal_plan, k_nom)
        u_act, w_act, c_act = self.update(state, actual_plan, k_act)

        s_nom, x_nom = self.evaluate_plan(nominal_state, u_nom)
        s_act, x_act = self.evaluate_plan(state, u_act)

        accept = s_act <= s_nom + self.accept_margin
        u_sel = jnp.where(accept, u_act, u_nom)
        x_sel = jnp.where(accept, x_act, x_nom)
        x_next = x_sel[1]  # accepted plan propagated one step
        return (
            u_sel,
            u_act,
            x_sel,
            x_next,
            accept,
            s_nom,
            s_act,
            1.0 / jnp.sum(w_nom**2),
            1.0 / jnp.sum(w_act**2),
            jnp.min(c_nom),
            jnp.min(c_act),
        )

    def step(self, state):
        """Run one nominal-state selection cycle and return the nominal control."""
        state = jnp.asarray(state, dtype=jnp.float32)
        if self.nominal_state is None:
            self.nominal_state = state
        self._key, key = jax.random.split(self._key)
        out = self._solve_pair(
            state, self.nominal_state, self.nominal_plan, self.actual_plan, key
        )
        (
            u_sel,
            u_act,
            x_sel,
            x_next,
            accept,
            s_nom,
            s_act,
            ess_nom,
            ess_act,
            best_nom,
            best_act,
        ) = out

        control0 = jnp.clip(u_sel[0], self.u_min, self.u_max)
        (
            control0_np,
            accept_np,
            s_nom_np,
            s_act_np,
            ess_nom_np,
            ess_act_np,
            best_nom_np,
            best_act_np,
        ) = jax.device_get(
            (control0, accept, s_nom, s_act, ess_nom, ess_act, best_nom, best_act)
        )

        self.plan = self.shift_plan(u_sel)
        self.nominal_plan = self.plan
        self.actual_plan = self.shift_plan(u_act)
        self.nominal_state = x_next
        self._accept_count += float(accept_np)
        self._step_count += 1

        info = {
            "accepted": bool(accept_np),
            "S_nom": float(s_nom_np),
            "S_act": float(s_act_np),
            "accept_margin": self.accept_margin,
            "ess_nom": float(ess_nom_np),
            "ess_act": float(ess_act_np),
            "best_cost_nom": float(best_nom_np),
            "best_cost_act": float(best_act_np),
            "nominal_trajectory": x_sel,
            "nominal_controls": jnp.clip(u_sel, self.u_min, self.u_max),
        }
        return np.asarray(control0_np), info

    def warmup(self, state):
        """Compile the two-solve kernel without advancing plans or PRNG state."""
        state = jnp.asarray(state, dtype=jnp.float32)
        _, key = jax.random.split(self._key)
        out = self._solve_pair(state, state, self.plan, self.plan, key)
        jax.block_until_ready(out)

    def set_plan(self, controls):
        """Replace the nominal and actual plans."""
        self.plan = jnp.asarray(controls, dtype=jnp.float32)
        self.nominal_plan = self.plan
        self.actual_plan = self.plan
