"""Dimension-agnostic reachset-aware sampling planner.

This is the generalized core extracted from ``ImmraxController``: an MPPI-style
sampler that draws control perturbations, rolls each candidate out through an
``immrax`` ``System`` (either as a point trajectory or an interval reachable set
via ``natemb``), scores it with a pluggable cost evaluator, and returns the
best (lowest-cost) control.

Unlike ``ImmraxController`` it has **no Buzzracer coupling** and **no hard-coded
state/control/disturbance dimensions** — all three come from the supplied
``predictor`` (state dim), ``control_bounds`` (control dim), and
``disturbance_widths`` (disturbance dim).  ``ImmraxController`` wraps one of
these for the 6-state dynamic-bicycle car; the DualGuard RC experiment wraps one
for the 3-state Dubins car.

The two-channel [steering, throttle] behavior of the original controller is
preserved byte-for-byte: for ``control_dim == 2`` the PRNG split
(``split(key, 3)`` -> ``[ch0, ch1, next]``) and per-channel noise are identical
to the original ``sample_controls``.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from immrax import i2ut, icentpert, interval, natemb, ut2i
from immrax.system.trajectory import RawTrajectory
from immrax.utils import timed

# Costs at or above this encode a predicted crash: every cost evaluator adds a
# collision term of 1e9 (or infinity) and bounds the collision-free terms far
# below it.  Also used by ``step`` to flag a control as unsafe.
UNSAFE_COST_THRESHOLD = 1e8

# Number of equal slices of the planning horizon whose reachset width growth is
# tracked.
GROWTH_CHUNKS = 5


def _get_control_index(t, planning_dt, planning_horizon):
    """Map continuous integration time ``t`` to a control index."""
    return (
        jnp.floor((t + 0.5 * planning_dt) / planning_dt).astype(int) % planning_horizon
    )


@dataclass
class SampleBounds:
    """Sampling bounds for one control channel (or its time-derivative)."""

    min: float
    std: float
    max: float

    def ensure_bounded(self, val):
        return jnp.clip(val, self.min, self.max)


class ImmraxPlanner:
    """Reachset-aware sampling planner over an arbitrary ``immrax`` system.

    Parameters
    ----------
    predictor : immrax.System
        Open-loop system ``f(t, x, u, w)``.  ``predictor.xlen`` sets the state
        dimension.
    control_bounds, control_dt_bounds : list[SampleBounds]
        Per-channel value / time-derivative sampling bounds.  Their length sets
        the control dimension.  Order is significant (channel ``i`` of the
        emitted control vector uses ``control_bounds[i]``).
    rollout_cost : RolloutCostEvaluator
        Cost evaluator exposing ``evaluate_rollout_cost(state_center, traj, u)``.
    disturbance : Disturbance
        The zero-centered box disturbance the reachsets propagate.  Its dimension
        must match ``predictor.wlen``.  Pass the *same* object to the plant so the
        two cannot disagree about what the car is being subjected to.
    initial_controls : array-like, shape (planning_horizon, control_dim)
        Warm-start control sequence.
    use_reachsets : bool
        If True, roll out interval reachable sets; else point trajectories.
    """

    def __init__(
        self,
        predictor,
        *,
        control_bounds,
        control_dt_bounds,
        rollout_cost,
        disturbance,
        initial_controls,
        num_samples: int = 1024,
        planning_horizon: int = 30,
        planning_dt: float = 0.02,
        prng_seed: int = 0,
        use_reachsets: bool = True,
    ):
        self.predictor = predictor
        self.state_dim = int(predictor.xlen)

        self.control_bounds = list(control_bounds)
        self.control_dt_bounds = list(control_dt_bounds)
        self.control_dim = len(self.control_bounds)
        assert len(self.control_dt_bounds) == self.control_dim, (
            "control_bounds and control_dt_bounds must have the same length"
        )

        self.rollout_cost = rollout_cost
        self.num_samples = num_samples
        self.planning_horizon = int(planning_horizon)
        self.planning_dt = planning_dt
        self.use_reachsets = use_reachsets

        self.prng_key = jax.random.key(jax.device_put(prng_seed))

        # Interval embedding of the predictor for reachable-set rollouts.
        self.reach_predictor = natemb(predictor)

        # Point rollouts propagate the zero disturbance; reachset rollouts propagate
        # the whole box.  A dimension mismatch here otherwise surfaces as an opaque
        # unpack error inside the predictor's f().
        predictor_wlen = getattr(predictor, "wlen", None)
        assert predictor_wlen is None or disturbance.dim == predictor_wlen, (
            f"disturbance has {disturbance.dim} channels but {predictor.name} takes "
            f"{predictor_wlen}"
        )
        self.disturbance_spec = disturbance
        self.disturbance_dim = disturbance.dim
        self._zero_disturbance = disturbance.zero()
        self.disturbance = lambda t, x: self._zero_disturbance
        self._zero_disturbance_interval = disturbance.interval()
        self.disturbance_int = lambda t, x: self._zero_disturbance_interval

        # Integration constants (pre-placed on device to avoid runtime transfers).
        self._t_start = jax.device_put(np.array(0.0))
        self._t_end = jax.device_put(np.array(self.planning_horizon * self.planning_dt))
        self._planning_dt_jax = jax.device_put(np.array(self.planning_dt))

        # Warm-start plan and its (finite-difference) time-derivative.
        initial_controls = jnp.asarray(initial_controls)
        assert initial_controls.shape == (self.planning_horizon, self.control_dim), (
            f"initial_controls must be (planning_horizon, control_dim) = "
            f"({self.planning_horizon}, {self.control_dim}), got {initial_controls.shape}"
        )
        self.planned_controls = initial_controls
        self.planned_controls_dt = (
            jnp.vstack(
                [
                    jnp.zeros((1, self.control_dim)),
                    jnp.diff(self.planned_controls, axis=0),
                ]
            )
            / self.planning_dt
        )

        self._rollout = (
            self._rollout_reachset if use_reachsets else self._rollout_trajectory
        )

        # Unsafe-fraction tracking (best plan predicts a collision).
        self._total_control_steps = 0
        self._unsafe_control_steps = set()

        # Lifetime running means of the per-step sample-safety fraction and of
        # the reachset widths at the end of each horizon chunk.  Widths stay at
        # 1 for point-trajectory rollouts, which carry no interval bounds.
        self._safe_sample_fraction = 0.0
        self._reachset_width_mean = np.ones(GROWTH_CHUNKS)
        self._reachset_width_steps = 0
        self._growth_chunk_indices = jax.device_put(
            np.array(
                [
                    (i + 1) * self.planning_horizon // GROWTH_CHUNKS
                    for i in range(GROWTH_CHUNKS)
                ]
            )
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample_controls(self, planned_controls_dt, planned_controls, prng_key):
        """Draw ``num_samples`` control sequences by perturbing the shifted plan.

        For each channel: perturb the (receding-horizon shifted) control
        derivative with Gaussian noise, clip, integrate (cumsum) to controls,
        and clip to value bounds.
        """
        keys = jax.random.split(prng_key, self.control_dim + 1)
        next_key = keys[-1]

        # Receding horizon: roll the plan up by one and zero the new last row.
        shifted_controls_dt = (
            jnp.roll(planned_controls_dt, -1, axis=0).at[-1, :].set(0.0)
        )

        noise_shape = (self.num_samples, self.planning_horizon)

        sampled_dt_channels = []
        sampled_channels = []
        for i in range(self.control_dim):
            dt_bounds = self.control_dt_bounds[i]
            val_bounds = self.control_bounds[i]

            noise = dt_bounds.std * jax.random.normal(keys[i], shape=noise_shape)
            sampled_dt = dt_bounds.ensure_bounded(shifted_controls_dt[:, i] + noise)
            sampled = val_bounds.ensure_bounded(
                planned_controls[0, i]
                + self.planning_dt * jnp.cumsum(sampled_dt, axis=1)
            )
            sampled_dt_channels.append(sampled_dt)
            sampled_channels.append(sampled)

        return (
            jnp.stack(sampled_dt_channels, axis=-1),
            jnp.stack(sampled_channels, axis=-1),
            next_key,
        )

    # ------------------------------------------------------------------
    # Rollouts
    # ------------------------------------------------------------------
    def rollout_sampled_trajectory(self, x0, control_traj) -> RawTrajectory:
        def control_action(t, x):
            idx = _get_control_index(t, self.planning_dt, self.planning_horizon)
            return control_traj[idx]

        return self.predictor.compute_trajectory(
            self._t_start,
            self._t_end,
            x0,
            (control_action, self.disturbance),
            dt=self._planning_dt_jax,
            solver="euler",
        )

    @partial(jax.jit, static_argnums=0)
    def rollout_reachset(self, x0_2n, planned_controls):
        def ff_control(t, x):
            idx = _get_control_index(t, self.planning_dt, self.planning_horizon)
            return interval(planned_controls[idx])

        return self.reach_predictor.compute_trajectory(
            self._t_start,
            self._t_end,
            x0_2n,
            (ff_control, self.disturbance_int),
            dt=self._planning_dt_jax,
            solver="euler",
        )

    def _rollout_reachset(self, state_jax, controls):
        x0_2n = i2ut(icentpert(state_jax, jnp.zeros(self.state_dim)))
        return self.rollout_reachset(x0_2n, controls)

    def _rollout_trajectory(self, state_jax, controls):
        return self.rollout_sampled_trajectory(state_jax, controls)

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------
    @timed
    @partial(jax.jit, static_argnums=0)
    def update_planned_controls(
        self, state, planned_controls_dt, planned_controls, prng_key
    ):
        state_jax = jnp.asarray(state[0 : self.state_dim])
        sampled_controls_dt, sampled_controls, next_key = self.sample_controls(
            planned_controls_dt, planned_controls, prng_key
        )

        rollouts = jax.vmap(self._rollout, in_axes=(None, 0))(
            state_jax, sampled_controls
        )
        costs = jax.vmap(
            lambda rollout, control: self.rollout_cost.evaluate_rollout_cost(
                state_jax, rollout, control
            )
        )(rollouts, sampled_controls)

        best_idx = jnp.nanargmin(costs)
        best_cost = costs[best_idx]
        best_rollout = jax.tree.map(lambda a: a[best_idx], rollouts)
        log_sample_controls = sampled_controls[:20]

        # Fraction of samples whose predicted crash cost is zero, i.e. whose
        # total cost stays below the crash threshold.
        safe_sample_fraction = jnp.mean(
            jnp.isfinite(costs) & (costs < UNSAFE_COST_THRESHOLD)
        )

        # Widest reachset dimension at the end of each horizon chunk, taken from
        # the reachset prediction for the selected control.  Point rollouts have
        # no bounds, so they report unit widths.
        if self.use_reachsets:
            traj_int = jax.vmap(ut2i)(best_rollout.ys)
            chunk_widths = traj_int.width[self._growth_chunk_indices]
            reachset_width_growth = jnp.max(chunk_widths, axis=1)
        else:
            reachset_width_growth = jnp.ones(GROWTH_CHUNKS)

        return (
            sampled_controls_dt[best_idx],
            sampled_controls[best_idx],
            next_key,
            best_rollout,
            log_sample_controls,
            best_cost,
            safe_sample_fraction,
            reachset_width_growth,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def warmup(self, state):
        """Trigger JIT compilation without advancing the plan or PRNG state."""
        state_dev = jax.device_put(np.asarray(state)[0 : self.state_dim])
        res, _ = self.update_planned_controls(
            state_dev, self.planned_controls_dt, self.planned_controls, self.prng_key
        )
        jax.block_until_ready(res[0])

    def step(self, state):
        """Advance one planning cycle.

        Parameters
        ----------
        state : array-like
            Current state; the first ``state_dim`` entries are used.

        Returns
        -------
        control0 : np.ndarray, shape (control_dim,)
            First control of the newly optimized plan.
        info : dict
            ``compute_time`` (s), ``best_cost``, ``unsafe`` (bool),
            ``best_rollout`` (device), ``sample_controls`` (device, first 20),
            ``safe_sample_fraction`` (fraction of this step's samples with zero
            predicted crash cost), ``reachset_width_growth`` (device, widest
            bound per horizon chunk for the selected control).
        """
        state_dev = jax.device_put(np.asarray(state))
        res, compute_time = self.update_planned_controls(
            state_dev,
            self.planned_controls_dt,
            self.planned_controls,
            self.prng_key,
        )
        (
            self.planned_controls_dt,
            self.planned_controls,
            self.prng_key,
            best_rollout,
            log_sample_controls,
            best_cost,
            safe_sample_fraction,
            reachset_width_growth,
        ) = res

        self._total_control_steps += 1
        best_cost_host = float(jax.device_get(best_cost))
        safe_fraction_host, growth_host = jax.device_get(
            (safe_sample_fraction, reachset_width_growth)
        )
        unsafe = best_cost_host > UNSAFE_COST_THRESHOLD
        if unsafe:
            self._unsafe_control_steps.add(self._total_control_steps)

        # Exponential-form running means (exact for a fixed sample count per step).
        n = self._total_control_steps
        self._safe_sample_fraction += (
            float(safe_fraction_host) - self._safe_sample_fraction
        ) / n
        if self.use_reachsets:
            growth = np.asarray(growth_host)
            # Overflowed bounds carry no usable width; leave them out of the mean.
            if np.isfinite(growth).all():
                self._reachset_width_steps += 1
                self._reachset_width_mean += (
                    growth - self._reachset_width_mean
                ) / self._reachset_width_steps

        control0 = jax.device_get(self.planned_controls)[0]
        info = {
            "compute_time": float(compute_time),
            "best_cost": best_cost_host,
            "unsafe": unsafe,
            "best_rollout": best_rollout,
            "sample_controls": log_sample_controls,
            "safe_sample_fraction": float(safe_fraction_host),
            "reachset_width_growth": growth_host,
        }
        return control0, info

    @property
    def unsafe_fraction(self):
        """Fraction of control steps whose best plan predicted a collision."""
        if self._total_control_steps == 0:
            return 0.0
        return len(self._unsafe_control_steps) / self._total_control_steps

    @property
    def safe_sample_fraction(self):
        """Lifetime running mean of the fraction of samples predicted safe."""
        return self._safe_sample_fraction

    @property
    def reachset_width_growth(self):
        """Lifetime mean, per horizon chunk, of the widest reachset dimension."""
        return self._reachset_width_mean.copy()
