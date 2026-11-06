from abc import ABC, abstractmethod
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from immrax import RawDiscreteTrajectory, natif, ut2i
from immrax.system.trajectory import RawTrajectory

# Weights of the two safety-critical state terms: the sparse collision indicator
# and the soft track-boundary barrier.  Tube-MPPI uses their minimum as the
# nominal-state acceptance margin (Williams et al. 2018, Algorithm 2).
COLLISION_COST_WEIGHT = 1e9
BOUNDARY_COST_WEIGHT = 10.0


class RolloutCostEvaluator(ABC):
    """Base class for rollout cost evaluation strategies.

    Subclasses implement `evaluate_rollout_cost` to define how the cost
    of a rollout is computed. Shared trajectory cost infrastructure
    (Frenet projection, cost components) lives here.
    """

    def __init__(
        self,
        track,
        planning_horizon: int,
        planning_dt: float,
        centering_weight_factor: float = 1.0,
        velocity_scale_factor: float = 1.0,
    ):
        self.planning_horizon = planning_horizon
        self.planning_dt = planning_dt
        self.track = track
        self.centering_weight_factor = centering_weight_factor
        self.velocity_scale_factor = velocity_scale_factor

        self._jax_ss = jax.device_put(np.asarray(track.ss))
        self._jax_raceline_velocity = jax.device_put(
            np.asarray(track.raceline_velocity)
        )
        self._jax_raceline_left_boundary = jax.device_put(
            np.asarray(track.raceline_left_boundary)
        )
        self._jax_raceline_right_boundary = jax.device_put(
            np.asarray(track.raceline_right_boundary)
        )
        self._jax_raceline_x = jax.device_put(np.asarray(track.raceline_points[0]))
        self._jax_raceline_y = jax.device_put(np.asarray(track.raceline_points[1]))
        self._jax_raceline_headings = jax.device_put(
            np.asarray(track.raceline_headings)
        )

    @abstractmethod
    def evaluate_rollout_cost(
        self, state_center: jax.Array, traj: RawTrajectory, u: jax.Array
    ): ...

    def evaluate_trajectory_cost(self, state: jax.Array, traj: RawTrajectory):
        components = self._compute_cost_components(state, traj)
        return sum(components.values())

    def _compute_cost_components(self, state: jax.Array, traj: RawTrajectory):
        traj_slice = traj.ys[: self.planning_horizon + 1]

        xy_vals = traj_slice[:, :2]
        vel = traj_slice[:, 3]

        progress_vals, lateral_err, index = self._project_to_frenet_batch(xy_vals)
        abs_lateral_err = jnp.abs(lateral_err)
        ref_vel = self.velocity_scale_factor * self._jax_raceline_velocity[index]
        bounds_left = self._jax_raceline_left_boundary[index]
        bounds_right = self._jax_raceline_right_boundary[index]

        vel_diff_sq = (vel - ref_vel) ** 2
        step_cost = jnp.sum(
            self.centering_weight_factor * 5.0 * abs_lateral_err
            + 2.5 * vel_diff_sq
            - 10.0 * jnp.minimum(vel, 0.0)
        )

        bound = jnp.where(lateral_err > 0, bounds_left, bounds_right)
        boundary_arg = -100.0 * (bound - (abs_lateral_err + 0.05))
        boundary_cost = (
            self.centering_weight_factor
            * BOUNDARY_COST_WEIGHT
            * jnp.sum(jnp.maximum(0.0, (2.0 / jnp.pi) * jnp.arctan(boundary_arg) + 1.0))
        )

        collision = jnp.logical_or(
            lateral_err > bounds_left, lateral_err < -bounds_right
        )
        collision_cost = jnp.where(jnp.any(collision), COLLISION_COST_WEIGHT, 0.0)

        progress_made = progress_vals[-1] - progress_vals[0]
        track_len = self.track.raceline_len_m
        progress_made = jnp.where(
            progress_made < -track_len / 2,
            progress_made + track_len,
            jnp.where(
                progress_made > track_len / 2,
                progress_made - track_len,
                progress_made,
            ),
        )
        terminal_cost = -2.0 * progress_made

        return {
            "step_cost": step_cost,
            "boundary_cost": boundary_cost,
            "collision_cost": collision_cost,
            "terminal_cost": terminal_cost,
        }

    def _project_to_frenet_batch(
        self, xy_vals: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        dx = xy_vals[:, 0:1] - self._jax_raceline_x[None, :]
        dy = xy_vals[:, 1:2] - self._jax_raceline_y[None, :]
        dists_sq = dx**2 + dy**2

        closest_idx = jnp.argmin(dists_sq, axis=1)

        progress = self._jax_ss[closest_idx]

        raceline_heading = self._jax_raceline_headings[closest_idx]
        dx_closest = xy_vals[:, 0] - self._jax_raceline_x[closest_idx]
        dy_closest = xy_vals[:, 1] - self._jax_raceline_y[closest_idx]

        lateral_err = -dx_closest * jnp.sin(raceline_heading) + dy_closest * jnp.cos(
            raceline_heading
        )

        return progress, lateral_err, closest_idx


class CenterReachsetCost(RolloutCostEvaluator):
    """Evaluate cost on center trajectory only (ignores reachset width)."""

    def evaluate_rollout_cost(
        self, state_center: jax.Array, traj: RawTrajectory, u: jax.Array
    ):
        ts = traj.ts
        traj_int = jax.vmap(ut2i)(traj.ys)
        center_traj = RawDiscreteTrajectory(ts, traj_int.center)
        return self.evaluate_trajectory_cost(state_center, center_traj)


class VerticesReachsetCost(RolloutCostEvaluator):
    """Evaluate cost on 4 corner trajectories in x-y, sum the costs."""

    @partial(jax.jit, static_argnums=0)
    def evaluate_rollout_cost(
        self, state_center: jax.Array, traj: RawTrajectory, u: jax.Array
    ):
        ts = traj.ts
        traj_int = jax.vmap(ut2i)(traj.ys)
        center = traj_int.center

        # Build 4 trajectories: each corner of x-y interval, other states use center
        # FIXME: using midpoints for non-x-y states is dangerous and may misrepresent costs
        x_lo = traj_int[:, 0].lower
        x_hi = traj_int[:, 0].upper
        y_lo = traj_int[:, 1].lower
        y_hi = traj_int[:, 1].upper

        def make_corner_traj(x_vals, y_vals):
            return center.at[:, 0].set(x_vals).at[:, 1].set(y_vals)

        corners = jnp.stack(
            [
                make_corner_traj(x_lo, y_lo),
                make_corner_traj(x_lo, y_hi),
                make_corner_traj(x_hi, y_lo),
                make_corner_traj(x_hi, y_hi),
            ],
            axis=0,
        )

        def eval_corner(corner_ys):
            corner_traj = RawDiscreteTrajectory(ts, corner_ys)
            return self.evaluate_trajectory_cost(state_center, corner_traj)

        state_costs = jax.vmap(eval_corner)(corners)

        control_cost = 1e1 * jnp.sum(jnp.square(jnp.diff(u, axis=0))) + 5e0 * jnp.sum(
            jnp.square(u)
        )

        return jnp.sum(state_costs) + control_cost


class ExhaustiveCollisionReachsetCost(RolloutCostEvaluator):
    """Like ``VerticesReachsetCost``, but the collision component is checked
    over a dense uniform 10x10 cartesian grid spanning each step's x-y
    interval instead of only at the 4 corner vertices.

    All other cost components (step, speed, boundary, terminal, control) are
    computed identically to ``VerticesReachsetCost`` (summed over the 4
    corners). If any sampled grid point would be a collision after Frenet
    projection, the total cost is ``jnp.inf``.
    """

    GRID_RESOLUTION = 10

    @partial(jax.jit, static_argnums=0)
    def evaluate_rollout_cost(
        self, state_center: jax.Array, traj: RawTrajectory, u: jax.Array
    ):
        ts = traj.ts
        traj_int = jax.vmap(ut2i)(traj.ys)
        center = traj_int.center

        # Build 4 trajectories: each corner of x-y interval, other states use center
        # FIXME: using midpoints for non-x-y states is dangerous and may misrepresent costs
        x_lo = traj_int[:, 0].lower
        x_hi = traj_int[:, 0].upper
        y_lo = traj_int[:, 1].lower
        y_hi = traj_int[:, 1].upper

        def make_corner_traj(x_vals, y_vals):
            return center.at[:, 0].set(x_vals).at[:, 1].set(y_vals)

        corners = jnp.stack(
            [
                make_corner_traj(x_lo, y_lo),
                make_corner_traj(x_lo, y_hi),
                make_corner_traj(x_hi, y_lo),
                make_corner_traj(x_hi, y_hi),
            ],
            axis=0,
        )

        def eval_corner(corner_ys):
            corner_traj = RawDiscreteTrajectory(ts, corner_ys)
            components = self._compute_cost_components(state_center, corner_traj)
            # Collision is handled separately over a dense grid; exclude it here.
            return sum(v for k, v in components.items() if k != "collision_cost")

        state_costs = jax.vmap(eval_corner)(corners)

        collision_cost = self._evaluate_grid_collision_cost(x_lo, x_hi, y_lo, y_hi)

        control_cost = 1e1 * jnp.sum(jnp.square(jnp.diff(u, axis=0))) + 5e0 * jnp.sum(
            jnp.square(u)
        )

        return jnp.sum(state_costs) + collision_cost + control_cost

    def _evaluate_grid_collision_cost(
        self,
        x_lo: jax.Array,
        x_hi: jax.Array,
        y_lo: jax.Array,
        y_hi: jax.Array,
    ) -> jax.Array:
        res = self.GRID_RESOLUTION

        x_lo = x_lo[: self.planning_horizon]
        x_hi = x_hi[: self.planning_horizon]
        y_lo = y_lo[: self.planning_horizon]
        y_hi = y_hi[: self.planning_horizon]

        # Uniform sampling fractions across each interval (endpoints included).
        fracs = jnp.linspace(0.0, 1.0, res)

        # Per-step 10x10 cartesian grid: shape (T, res, res).
        xs = x_lo[:, None, None] + (x_hi - x_lo)[:, None, None] * fracs[None, :, None]
        ys = y_lo[:, None, None] + (y_hi - y_lo)[:, None, None] * fracs[None, None, :]
        xs = jnp.broadcast_to(xs, (x_lo.shape[0], res, res))
        ys = jnp.broadcast_to(ys, (y_lo.shape[0], res, res))

        xy_vals = jnp.stack([xs.reshape(-1), ys.reshape(-1)], axis=-1)

        _, lateral_err, index = self._project_to_frenet_batch(xy_vals)
        bounds_left = self._jax_raceline_left_boundary[index]
        bounds_right = self._jax_raceline_right_boundary[index]

        collision = jnp.logical_or(
            lateral_err > bounds_left, lateral_err < -bounds_right
        )
        return jnp.where(jnp.any(collision), jnp.inf, 0.0)


class NatifUpperReachsetCost(RolloutCostEvaluator):
    """Evaluate cost upper bound via interval arithmetic (natif)."""

    def evaluate_rollout_cost(
        self, state_center: jax.Array, traj: RawTrajectory, u: jax.Array
    ):
        ts = traj.ts
        traj_int = jax.vmap(ut2i)(traj.ys)

        jax.debug.print("AAAAA")

        def cost_fn(traj_ys):
            traj_obj = RawDiscreteTrajectory(ts, traj_ys)
            return self.evaluate_trajectory_cost(state_center, traj_obj)

        return natif(cost_fn)(traj_int).upper


class NatifLowerReachsetCost(RolloutCostEvaluator):
    """Evaluate cost lower bound via interval arithmetic (natif)."""

    def evaluate_rollout_cost(
        self, state_center: jax.Array, traj: RawTrajectory, u: jax.Array
    ):
        ts = traj.ts
        traj_int = jax.vmap(ut2i)(traj.ys)

        def cost_fn(traj_ys):
            traj_obj = RawDiscreteTrajectory(ts, traj_ys)
            return self.evaluate_trajectory_cost(state_center, traj_obj)

        return natif(cost_fn)(traj_int).lower


class RegularizedReachsetCost(RolloutCostEvaluator):
    """Center cost plus penalty on interval width (reachset size)."""

    def evaluate_rollout_cost(
        self, state_center: jax.Array, traj: RawTrajectory, u: jax.Array
    ):
        ts = traj.ts
        traj_int = jax.vmap(ut2i)(traj.ys)

        center_traj = RawDiscreteTrajectory(ts, traj_int.center)
        center_cost = self.evaluate_trajectory_cost(state_center, center_traj)

        widths = traj_int.width
        size_costs = jnp.sum(
            jax.vmap(
                lambda w: jnp.where(jnp.isfinite(w).all(), jnp.abs(jnp.prod(w)), 1e4)
            )(widths[: self.planning_horizon, :6])
        )

        return center_cost + size_costs


class PointTrajectoryCost(RolloutCostEvaluator):
    """Evaluate cost on a single point trajectory plus control regularization.

    ``control_diff_weight`` scales the finite-difference penalty; 0 omits it.
    """

    def __init__(
        self,
        track,
        planning_horizon: int,
        planning_dt: float,
        centering_weight_factor: float = 1.0,
        velocity_scale_factor: float = 1.0,
        control_diff_weight: float = 1e1,
        control_weight: float = 5e0,
    ):
        super().__init__(
            track,
            planning_horizon,
            planning_dt,
            centering_weight_factor=centering_weight_factor,
            velocity_scale_factor=velocity_scale_factor,
        )
        self.control_diff_weight = control_diff_weight
        self.control_weight = control_weight

    @partial(jax.jit, static_argnums=0)
    def evaluate_rollout_cost(self, state_center, traj, u):
        state_cost = self.evaluate_trajectory_cost(state_center, traj)
        control_cost = self.control_diff_weight * jnp.sum(
            jnp.square(jnp.diff(u, axis=0))
        ) + self.control_weight * jnp.sum(jnp.square(u))
        return state_cost + control_cost


class SigmaWeightedControlCost(PointTrajectoryCost):
    """Paper cost ``L(x, u) = C(x) + lambda * u^T Sigma^-1 u``.

    ``C(x)`` is inherited unchanged from ``PointTrajectoryCost`` (the state terms
    plus the sparse collision indicator).  The control term uses the sampling
    covariance ``Sigma``: ``control_sigma`` must be the planner's per-step control
    std, so the same ``Sigma`` is used for sampling and for the control cost.
    """

    def __init__(
        self,
        track,
        planning_horizon: int,
        planning_dt: float,
        *,
        control_sigma,
        control_lambda: float = 1.0,
        centering_weight_factor: float = 1.0,
        velocity_scale_factor: float = 1.0,
    ):
        super().__init__(
            track,
            planning_horizon,
            planning_dt,
            centering_weight_factor=centering_weight_factor,
            velocity_scale_factor=velocity_scale_factor,
            control_diff_weight=0.0,
            control_weight=0.0,
        )
        self.control_lambda = float(control_lambda)
        self.inv_sigma_sq = jax.device_put(
            1.0 / jnp.asarray(control_sigma, dtype=jnp.float32) ** 2
        )

    @partial(jax.jit, static_argnums=0)
    def evaluate_rollout_cost(self, state_center, traj, u):
        state_cost = self.evaluate_trajectory_cost(state_center, traj)
        control_cost = self.control_lambda * jnp.sum(
            jnp.square(u) * self.inv_sigma_sq
        )
        return state_cost + control_cost


# ---------------------------------------------------------------------------
# DualGuard-MPPI RC-car cost (arXiv 2502.01924, Section VI, eq. 16)
#
#   S = (V_max - V)^2 + K_c * (l_center - l(x)) + P(x)
#
# on the 3-state Dubins layout ``[x, y, heading]`` with control ``[V, delta]``:
#   * velocity term      -> from the control channel ``V = u[:, 0]`` (V is a
#     control here, not a state, so it does NOT come from ``traj.ys``);
#   * centerline term    -> ``K_c * |lateral_err|`` via the same Frenet
#     projection used by the 6-state costs (``centering_weight_factor`` = K_c);
#   * safety penalty P(x)-> the reachset boundary barrier + hard collision when
#     a reachable-set corner exits the track boundary (this replaces DualGuard's
#     precomputed HJ BRT).
#
# Only x/y (indices 0, 1) and heading are referenced, so these classes are
# state-dimension-agnostic and do not disturb the 6-state cost path above.
# ---------------------------------------------------------------------------


class DubinsRacingCost(RolloutCostEvaluator):
    """Reachset (4-corner) evaluation of eq. 16 for the 3-state Dubins car."""

    def __init__(
        self,
        track,
        planning_horizon: int,
        planning_dt: float,
        v_max: float = 1.4,
        centering_weight_factor: float = 1.0,
        velocity_scale_factor: float = 1.0,
        control_smoothness_weight: float = 0.0,
        terminal_progress_weight: float = 0.0,
    ):
        super().__init__(
            track,
            planning_horizon,
            planning_dt,
            centering_weight_factor=centering_weight_factor,
            velocity_scale_factor=velocity_scale_factor,
        )
        self.v_max = v_max
        # eq. 16 has no explicit control-regularization term; both default to 0.
        self.control_smoothness_weight = control_smoothness_weight
        self.terminal_progress_weight = terminal_progress_weight

    def _trajectory_state_cost(self, traj_ys: jax.Array) -> jax.Array:
        """Centerline (K_c*|lat|) + boundary barrier + hard collision (P(x)).

        ``traj_ys`` is a point trajectory of shape (T+1, state_dim); only the
        x/y columns (0, 1) are used.
        """
        traj_slice = traj_ys[: self.planning_horizon + 1]
        xy_vals = traj_slice[:, :2]

        progress_vals, lateral_err, index = self._project_to_frenet_batch(xy_vals)
        abs_lateral_err = jnp.abs(lateral_err)

        # Centerline-following term  K_c * |l_center - l(x)|.
        centerline_cost = jnp.sum(self.centering_weight_factor * abs_lateral_err)

        # Safety penalty P(x): soft arctan boundary barrier + hard collision.
        bounds_left = self._jax_raceline_left_boundary[index]
        bounds_right = self._jax_raceline_right_boundary[index]
        bound = jnp.where(lateral_err > 0, bounds_left, bounds_right)
        boundary_arg = -100.0 * (bound - (abs_lateral_err + 0.05))
        boundary_cost = 10.0 * jnp.sum(
            jnp.maximum(0.0, (2.0 / jnp.pi) * jnp.arctan(boundary_arg) + 1.0)
        )
        collision = jnp.logical_or(
            lateral_err > bounds_left, lateral_err < -bounds_right
        )
        collision_cost = jnp.where(jnp.any(collision), 1e9, 0.0)

        cost = centerline_cost + boundary_cost + collision_cost
        return cost + self._progress_cost(progress_vals)

    def _progress_cost(self, progress_vals: jax.Array) -> jax.Array:
        """Cost contribution of the optional lap-progress reward (0 => strict eq. 16).

        ``progress_vals`` is arc length along the raceline, so the net progress over
        the horizon needs unwrapping at the start/finish line.
        """
        if self.terminal_progress_weight == 0.0:
            return jnp.asarray(0.0)

        progress_made = progress_vals[-1] - progress_vals[0]
        track_len = self.track.raceline_len_m
        progress_made = jnp.where(
            progress_made < -track_len / 2,
            progress_made + track_len,
            jnp.where(
                progress_made > track_len / 2,
                progress_made - track_len,
                progress_made,
            ),
        )
        return -self.terminal_progress_weight * progress_made

    def _velocity_and_smoothness_cost(self, u: jax.Array) -> jax.Array:
        # (V_max - V)^2 summed over the horizon; V is control channel 0.
        vel_cost = jnp.sum((self.v_max - u[:, 0]) ** 2)
        smooth_cost = self.control_smoothness_weight * jnp.sum(
            jnp.square(jnp.diff(u, axis=0))
        )
        return vel_cost + smooth_cost

    @partial(jax.jit, static_argnums=0)
    def evaluate_rollout_cost(self, state_center, traj, u):
        traj_int = jax.vmap(ut2i)(traj.ys)
        center = traj_int.center

        x_lo = traj_int[:, 0].lower
        x_hi = traj_int[:, 0].upper
        y_lo = traj_int[:, 1].lower
        y_hi = traj_int[:, 1].upper

        def make_corner_traj(x_vals, y_vals):
            return center.at[:, 0].set(x_vals).at[:, 1].set(y_vals)

        corners = jnp.stack(
            [
                make_corner_traj(x_lo, y_lo),
                make_corner_traj(x_lo, y_hi),
                make_corner_traj(x_hi, y_lo),
                make_corner_traj(x_hi, y_hi),
            ],
            axis=0,
        )

        state_costs = jax.vmap(self._trajectory_state_cost)(corners)

        return jnp.sum(state_costs) + self._velocity_and_smoothness_cost(u)


class PointDubinsRacingCost(DubinsRacingCost):
    """Single-point (non-reachset) evaluation of eq. 16 for the Dubins car."""

    @partial(jax.jit, static_argnums=0)
    def evaluate_rollout_cost(self, state_center, traj, u):
        state_cost = self._trajectory_state_cost(traj.ys)
        return state_cost + self._velocity_and_smoothness_cost(u)


# ---------------------------------------------------------------------------
# eq. 16 as the paper actually writes it, over the l(x) field.
#
# DubinsRacingCost above measures "distance from the track center" as Frenet lateral
# error against a *minimum-curvature racing line* -- which is not the centerline, and
# is not what l(x) means.  The paper's l(x) is the distance-to-nearest-obstacle field
# solved by the Level-Set Toolbox and shipped in brt_rc_wh_fine_v3.mat; l_center is its
# maximum, attained on the true centerline.  So
#
#     K_c * (l_center - l(x))
#
# is distance from the *centerline*, and the safety penalty P(x) keys off the sign of
# l(x) directly rather than off a raceline-relative corridor.  These classes read the
# field straight off the grid, which is both the faithful formulation and cheaper (no
# 1024-point nearest-neighbour scan per state).
# ---------------------------------------------------------------------------


class LxDubinsRacingCost(DubinsRacingCost):
    """Reachset (4-corner) evaluation of eq. 16 over the l(x) grid.

    Requires a track exposing ``lx_grid`` / ``lx_dx`` / ``l_center`` (i.e.
    ``WarehouseTrack``); an RCP track has no l(x) field.
    """

    def __init__(
        self,
        track,
        planning_horizon: int,
        planning_dt: float,
        v_max: float = 1.4,
        centering_weight_factor: float = 120.0,
        velocity_scale_factor: float = 1.0,
        control_smoothness_weight: float = 0.0,
        terminal_progress_weight: float = 0.0,
        collision_weight: float = 50.0,
        hard_collision: bool = True,
    ):
        super().__init__(
            track,
            planning_horizon,
            planning_dt,
            v_max=v_max,
            centering_weight_factor=centering_weight_factor,
            velocity_scale_factor=velocity_scale_factor,
            control_smoothness_weight=control_smoothness_weight,
            terminal_progress_weight=terminal_progress_weight,
        )
        if not hasattr(track, "lx_grid"):
            raise TypeError(
                f"{type(track).__name__} has no l(x) field; the eq.-16 lx cost needs a "
                "track reconstructed from the HJB artifact (WarehouseTrack)"
            )
        self._jax_lx = jax.device_put(jnp.asarray(track.lx_grid, dtype=jnp.float32))
        self._lx_dx = float(track.lx_dx)
        self.l_center = float(track.l_center)
        self.collision_weight = collision_weight
        self.hard_collision = hard_collision

    def _lookup_lx(self, xy_vals: jax.Array) -> jax.Array:
        """l(x) at each (x, y), by nearest grid cell.  Off-grid counts as wall."""
        ny, nx = self._jax_lx.shape
        xi = jnp.round(xy_vals[:, 0] / self._lx_dx).astype(int)
        yi = jnp.round(xy_vals[:, 1] / self._lx_dx).astype(int)
        in_bounds = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
        lx = self._jax_lx[jnp.clip(yi, 0, ny - 1), jnp.clip(xi, 0, nx - 1)]
        # Matches the reconstruction's RegularGridInterpolator(fill_value=-1.0): there
        # is no track outside the grid, so leaving it is a collision.
        return jnp.where(in_bounds, lx, -1.0)

    def _trajectory_state_cost(self, traj_ys: jax.Array) -> jax.Array:
        traj_slice = traj_ys[: self.planning_horizon + 1]
        xy_vals = traj_slice[:, :2]
        lx = self._lookup_lx(xy_vals)

        # K_c * (l_center - l(x)):  distance from the centerline.
        centerline_cost = self.centering_weight_factor * jnp.sum(self.l_center - lx)

        # P(x):  graded, in the number of colliding steps.
        collided = lx < 0.0
        collision_cost = self.collision_weight * jnp.sum(collided)
        if self.hard_collision:
            # ImmraxPlanner.step reads its `unsafe` flag off `best_cost > 1e8`, so a
            # collision must be unambiguously separated from any collision-free cost.
            # Being constant across colliding rollouts, it leaves the graded term above
            # free to still discriminate *how badly* they collide -- which is what the
            # reachset arm needs when an over-approximated tube clips the wall on every
            # sample. Set hard_collision=False for the paper's formulation verbatim.
            collision_cost = collision_cost + jnp.where(jnp.any(collided), 1e9, 0.0)

        cost = centerline_cost + collision_cost
        if self.terminal_progress_weight != 0.0:
            progress_vals, _, _ = self._project_to_frenet_batch(xy_vals)
            cost = cost + self._progress_cost(progress_vals)
        return cost


class PointLxDubinsRacingCost(LxDubinsRacingCost):
    """Single-point (non-reachset) evaluation of eq. 16 over the l(x) grid."""

    @partial(jax.jit, static_argnums=0)
    def evaluate_rollout_cost(self, state_center, traj, u):
        state_cost = self._trajectory_state_cost(traj.ys)
        return state_cost + self._velocity_and_smoothness_cost(u)
