import time
from dataclasses import dataclass
from functools import partial
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from immrax import (
    RawDiscreteTrajectory,
    i2ut,
    icentpert,
    interval,
    natemb,
    jacemb,
    natif,
    ut2i,
)
from immrax.system.trajectory import RawTrajectory
from immrax.utils import timed

from buzzracer.controllers.car_controller import CarController
from buzzracer.controllers.stanley_car_controller import StanleyCarController
from immrax_dynamic_bicycle_cartesian import (
    DynamicBicycleCartesian,
)
from buzzracer.sysid.dynamic_bicycle_model import DynamicBicycleModelCartesian
from buzzracer.types import CartesianState, Control
from reachset_cost import (
    PointTrajectoryCost,
    VerticesReachsetCost,
    ExhaustiveCollisionReachsetCost
)
from immrax_planner import ImmraxPlanner, SampleBounds, _get_control_index
from disturbance import Disturbance

# jax.config.update("jax_debug_nans", True)

PRNG_SEED = 0


class ImmraxController(CarController):
    def __init__(self, car, config):
        # Allow transfers during initialization (reset any previous guard state)
        jax.config.update("jax_transfer_guard", "allow")

        # Default: use reachable sets. Override via XML: use_reachsets='False'
        self.use_reachsets = True

        # Configurable PRNG seed (overridable via XML: prng_seed='42')
        self.prng_seed = PRNG_SEED

        # Cost weight factors (overridable via XML)
        self.centering_weight_factor = 1.0
        self.velocity_scale_factor = 1.0

        # Planning horizon in time steps (overridable via XML: horizon='30')
        self.horizon = 30

        # load config etc
        super().__init__(car, config)
        self.print_debug_enable()

        self.debug_dict = {}

        # Sampler settings
        # ============================================================
        # NOTE: the sampling/rollout/optimization core now lives in the
        # dimension-agnostic ``ImmraxPlanner`` (immrax_planner.py); this
        # controller is the Buzzracer adapter that wires up the 6-state
        # dynamic-bicycle car and warm-starts the plan with a Stanley
        # controller. Behavior is byte-identical to the previous inline core.
        self.planning_dt = 0.02
        self.planning_horizon = int(self.horizon)  # time steps (XML: horizon=...)
        self.num_samples = 1024
        throttle_range = car.max_throttle - car.min_throttle
        self.throttle_dt_bounds = SampleBounds(
            min=-4 * throttle_range,  # allow slowing down very quickly
            std=throttle_range / 2,
            max=throttle_range,
        )
        self.steering_dt_bounds = SampleBounds(
            -3 / 2 * jnp.pi,
            3 / 4 * jnp.pi,
            3 / 2 * jnp.pi,
        )
        self.throttle_bounds = SampleBounds(
            min=car.min_throttle, std=car.max_throttle / 2, max=car.max_throttle
        )
        self.steering_bounds = SampleBounds(
            -car.max_steering_left,
            car.max_steering_right / 1,
            car.max_steering_right,
        )

        # Trajectory rollout settings
        # ============================================================
        self.predictor = DynamicBicycleCartesian(car)
        self.track = car.main.track

        # Mode-specific cost (control [steering, throttle], 6-state car).
        if self.use_reachsets:
            self.rollout_cost = VerticesReachsetCost(
            # self.rollout_cost = ExhaustiveCollisionReachsetCost(
                self.track,
                self.planning_horizon,
                self.planning_dt,
                centering_weight_factor=self.centering_weight_factor,
                velocity_scale_factor=self.velocity_scale_factor,
            )
        else:
            self.rollout_cost = PointTrajectoryCost(
                self.track,
                self.planning_horizon,
                self.planning_dt,
                centering_weight_factor=self.centering_weight_factor,
                velocity_scale_factor=self.velocity_scale_factor,
            )

        # Warm-start the plan with the Stanley controller.
        self.stanley_controller = StanleyCarController(car, config)
        initial = jax.device_put(
            np.array(self.car.state[:6])
        )  # [x, y, heading, vf, vs, omega]
        _, initial_controls = self.__plan_stanley_traj(
            initial
        )  # (steering, throttle) for each timestep over planning horizon

        # Generalized planner. Channel order [steering, throttle] and disturbance
        # widths [1e-1, 2e-1] match the previous inline implementation exactly.
        self.planner = ImmraxPlanner(
            self.predictor,
            control_bounds=[self.steering_bounds, self.throttle_bounds],
            control_dt_bounds=[self.steering_dt_bounds, self.throttle_dt_bounds],
            rollout_cost=self.rollout_cost,
            # Matches the simulator's lateral_only noise, which draws
            # [w_lat, w_heading] from [1.0, 2.0] * state_noise_magnitude (=0.1).
            disturbance=Disturbance((1e-1, 2e-1)),
            initial_controls=initial_controls,
            num_samples=self.num_samples,
            planning_horizon=self.planning_horizon,
            planning_dt=self.planning_dt,
            prng_seed=self.prng_seed,
            use_reachsets=self.use_reachsets,
        )
        self.controls = jax.device_get(self.planner.planned_controls)[0, :]

        # Pre-compute track arrays with explicit device transfer
        self._jax_ss = jax.device_put(np.asarray(self.track.ss))
        self._jax_raceline_velocity = jax.device_put(
            np.asarray(self.track.raceline_velocity)
        )
        self._jax_raceline_left_boundary = jax.device_put(
            np.asarray(self.track.raceline_left_boundary)
        )
        self._jax_raceline_right_boundary = jax.device_put(
            np.asarray(self.track.raceline_right_boundary)
        )
        # Raceline arrays for Cartesian-to-Frenet projection
        self._jax_raceline_x = jax.device_put(
            np.asarray(self.track.raceline_points[0])
        )  # (N,)
        self._jax_raceline_y = jax.device_put(
            np.asarray(self.track.raceline_points[1])
        )  # (N,)
        self._jax_raceline_headings = jax.device_put(
            np.asarray(self.track.raceline_headings)
        )  # (N,)

        # Unsafe-fraction tracking now lives on ``self.planner`` and is exposed
        # via read-only properties below.

        # Visualization settings
        # ============================================================
        self._viz_skip_count = 1  # Only visualize every N updates
        self._viz_counter = 0

        # Timing instrumentation
        self._timing_log = []
        self._last_update_time_ms = 0.0

        # JIT warmup
        self.print_info("Warming up JIT compilation...")
        _t0 = time.perf_counter()
        self.planner.warmup(initial)
        _t1 = time.perf_counter()
        self.print_info(f"JIT compilation time: {(_t1 - _t0) * 1000:.1f}ms")

        # Disallow implicit host-to-device transfers after initialization
        jax.config.update("jax_transfer_guard", "disallow")

    def control(self):
        # lookahead_dist = 46e-3  # optitrack uses rear axel, sim assumes CoM
        lookahead_dist = 0.0  # optitrack uses rear axel, sim assumes CoM
        x, y, heading, v_forward, v_sideway, omega = (*self.car.state,)
        x += lookahead_dist * np.cos(heading)
        y += lookahead_dist * np.sin(heading)
        adjusted_state = (x, y, heading, v_forward, v_sideway, omega)

        throttle, steering, valid, debug_dict = self.ctrl_car(
            adjusted_state, self.track
        )
        # Timing log
        self._timing_log.append(self._last_update_time_ms)
        if len(self._timing_log) % 100 == 0:
            self.print_info(
                f"Control loop: mean={np.mean(self._timing_log[-100:]):.2f}ms, std={np.std(self._timing_log[-100:]):.2f}ms"
            )
        self.debug_dict = debug_dict
        self.car.debug_dict.update(debug_dict)
        if valid:
            self.car.throttle = throttle
            self.car.steering = steering
        else:
            self.print_warning(" car %d invalid results from ctrl_car", self.car.id)
            self.car.throttle = 0.0
            self.car.steering = 0.0
        return valid

    def ctrl_car(self, state, track, v_override=None, reverse=False):
        """
        given state of the vehicle and an instance of track, provide throttle and steering output
        input:
          state: (x,y,heading,v_forward,v_sideway,omega)
          track: track object, can be RCPTrack or skidpad
          v_override: If specified, use this as target velocity instead of the optimal value provided by track object
          reverse: true if running in opposite direction of raceline init direction

        output:
          (throttle,steering,valid,debug)
        ranges for output:
          throttle -1.0,self.max_throttle
          steering as an angle in radians, TRIMMED to self.max_steering, left(+), right(-)
          valid: bool, if the car can be controlled here, if this is false, then throttle will also be set to 0
                  This typically happens when vehicle is off track, and track object cannot find a reasonable local raceline
        debug: a dictionary of objects to be debugged, e.g. {offset, error in v}
        """

        control0, info = self.planner.step(state)
        self._last_update_time_ms = info["compute_time"] * 1000

        # self.__plot_planned_trajectory(info["best_rollout"].ox)
        # self.__debug_compare_costs(state, info["best_rollout"].ox)
        # self.__plot_track_bounds(np.asarray(state))
        # Reachset-only debug (requires use_reachsets=True):
        # (
        #     overflow_timestep,
        #     valid_polytopes,
        #     final_widths,
        # ) = self.__analyze_reachset_validity(info["best_rollout"])
        # self.__plot_reachset(info["best_rollout"])
        # self.__log_reachability_results(overflow_timestep, 0, final_widths)

        # Necessary transfer to interface w/ Buzzracer sim
        controls = jax.device_get(self.planner.planned_controls)
        # self.controls = 0.1 * self.controls + 0.9 * controls
        log_data = jax.device_get({"sample_controls": info["sample_controls"]})
        log_data["best_controls"] = controls
        log_data["state"] = np.asarray(state)
        result = (
            float(control0[1]),  # throttle
            float(control0[0]),  # steering
            True,
            log_data,
        )

        return result

    # ------------------------------------------------------------------
    # Planner delegation
    # ------------------------------------------------------------------
    # The sampling / rollout / optimization core lives in ``self.planner``
    # (immrax_planner.ImmraxPlanner). These proxies expose the pieces the
    # reachset_sweep instrumentation and the debug helpers below still read.

    @property
    def unsafe_fraction(self):
        """Fraction of control steps where the best trajectory had infinite cost."""
        return self.planner.unsafe_fraction

    @property
    def _total_control_steps(self):
        return self.planner._total_control_steps

    @property
    def _unsafe_control_steps(self):
        return self.planner._unsafe_control_steps

    @property
    def planned_controls(self):
        return self.planner.planned_controls

    @property
    def planned_controls_dt(self):
        return self.planner.planned_controls_dt

    def rollout_sampled_trajectory(self, x0, control_traj) -> RawTrajectory:
        return self.planner.rollout_sampled_trajectory(x0, control_traj)

    # ============================================================
    # Debug functions
    # ============================================================

    def __analyze_reachset_validity(self, planned_reachset):
        """Analyze reachable set for overflow and track boundary violations.

        Args:
            planned_reachset: RawTrajectory with .ys as (T, 2n) upper-triangle array.

        Returns:
            overflow_timestep: Index of the first overflowed timestep (or num_timesteps if none).
            valid_intervals: Interval array of valid timesteps.
            final_widths: State widths at last valid timestep, or None if none valid.
        """
        traj_int = jax.vmap(ut2i)(planned_reachset.ys)

        final_widths = None
        overflow_timestep = 0
        for idx in range(self.planning_horizon + 1):
            widths = traj_int[idx].width

            # Check for overflow (NaN or Inf in widths, or width > 1e6)
            if (
                jnp.any(jnp.isnan(widths))
                or jnp.any(jnp.isinf(widths))
                or jnp.any(widths > 1e6)
            ):
                break

            overflow_timestep = idx + 1

            self.__check_reachset_track_collision(idx, traj_int.center[idx, :2])
            final_widths = widths

        valid_intervals = jax.tree.map(lambda a: a[:overflow_timestep], traj_int)

        return (
            overflow_timestep,
            valid_intervals,
            final_widths,
        )

    def __check_reachset_track_collision(self, timestep_idx, center_xy_1d):
        """Check if a reachable set center point is inside track boundaries and warn if not."""
        center_xy = center_xy_1d[None, :]  # (1, 2)
        _, center_lateral_err, bound_idx = self._project_to_frenet_batch(center_xy)
        left_bound = self._jax_raceline_left_boundary[bound_idx[0]]
        right_bound = self._jax_raceline_right_boundary[bound_idx[0]]

        center_collision = (center_lateral_err[0] > left_bound) | (
            center_lateral_err[0] < -right_bound
        )
        if center_collision:
            self.print_warning(
                f"Reachable set center exits track at timestep {timestep_idx}! "
                f"lateral_err={float(center_lateral_err[0]):.4f}, "
                f"bounds=[-{float(right_bound):.4f}, +{float(left_bound):.4f}]"
            )

    def __plot_reachset(self, reach_traj):
        """Draw the interval overapproximation of the reachable set trajectory via GL."""
        viz = self.car.main.visualization
        traj_int = jax.vmap(ut2i)(reach_traj.ys)
        traj_int_cpu = jax.device_get(traj_int)

        overflow_timestep = 0
        center_points = []

        # Draw interval boxes as filled quads + collect center trajectory
        for i in range(self.planning_horizon + 1):
            iover = traj_int_cpu[i]
            widths = iover.width

            if (
                jnp.any(jnp.isnan(widths))
                or jnp.any(jnp.isinf(widths))
                or jnp.any(widths > 1e6)
            ):
                break

            overflow_timestep = i + 1

            x_lo = float(iover[0].lower)
            x_hi = float(iover[0].upper)
            y_lo = float(iover[1].lower)
            y_hi = float(iover[1].upper)

            viz.draw_quad(x_lo, y_lo, x_hi, y_hi, (0.0, 0.2, 0.8, 0.25))

            center_points.append((float(traj_int.center[i, 0]),
                                  float(traj_int.center[i, 1])))

        if center_points:
            viz.draw_polyline(center_points, color=(1.0, 0.0, 0.0, 0.8), width=0.005)

    def __log_reachability_results(self, overflow_timestep, reach_time, final_widths):
        """Log reachability analysis results including valid steps and bound widths."""
        state_names = ["x", "y", "heading", "v_forward", "v_sideway", "omega"]
        reach_time_ms = reach_time * 1000
        self.print_info(
            f"Reachability: {overflow_timestep}/{self.planning_horizon + 1} steps valid, "
            f"compute time: {reach_time_ms:.2f}ms"
        )

        if final_widths is not None:
            # Compute relative widths (normalized by smallest non-zero width)
            min_width = jnp.maximum(jnp.min(final_widths), 1e-10)
            relative_widths = final_widths / min_width
            width_strs = [
                f"{state_names[i]}={final_widths[i]:.4g} ({relative_widths[i]:.1f}x)"
                for i in range(6)
            ]
            self.print_info(f"Final valid bound widths: {', '.join(width_strs)}")
        elif overflow_timestep == 0:
            self.print_warning("Reachable set overflowed at first timestep!")

    def __evaluate_trajectory_cost_breakdown(
        self, state: jax.Array, traj: RawTrajectory
    ):
        """Evaluate trajectory cost with component breakdown for debugging."""
        components = self.__compute_cost_components(state, traj)
        components["total"] = sum(components.values())
        return components

    def __debug_compare_costs(self, sim_state: jax.Array, best_sampled_traj: jax.Array):
        """Compare cost breakdowns between Stanley reference and best sampled trajectory."""
        state_arr = jnp.array(sim_state[:6])

        # Get Stanley trajectory
        _, stanley_controls = self.__plan_stanley_traj(state_arr)
        stanley_traj = self.rollout_sampled_trajectory(state_arr, stanley_controls)

        # Create RawTrajectory from best sampled traj for cost evaluation
        class FakeTraj:
            def __init__(self, ys):
                self.ys = ys

        best_traj_obj = FakeTraj(best_sampled_traj)

        # Get cost breakdowns
        stanley_costs = self.__compute_cost_components(state_arr, stanley_traj)
        sampled_costs = self.__compute_cost_components(state_arr, best_traj_obj)

        stanley_total = sum(stanley_costs.values())
        sampled_total = sum(sampled_costs.values())

        # Print comparison
        self.print_info("=" * 60)
        self.print_info("COST COMPARISON: Stanley vs Best Sampled")
        self.print_info("-" * 60)
        self.print_info(
            f"{'Component':<20} {'Stanley':>15} {'Sampled':>15} {'Diff':>15}"
        )
        self.print_info("-" * 60)
        for key in stanley_costs:
            s_val = float(stanley_costs[key])
            b_val = float(sampled_costs[key])
            diff = b_val - s_val
            self.print_info(f"{key:<20} {s_val:>15.2f} {b_val:>15.2f} {diff:>+15.2f}")
        self.print_info("-" * 60)
        self.print_info(
            f"{'total':<20} {float(stanley_total):>15.2f} {float(sampled_total):>15.2f} "
            f"{float(sampled_total - stanley_total):>+15.2f}"
        )

        # Additional debug: show lateral error and boundary values
        self.print_info("Lateral error debug (sampled trajectory):")
        traj_slice = best_sampled_traj[: self.planning_horizon]
        xy_vals = traj_slice[:, :2]
        _, lateral_err, index = self._project_to_frenet_batch(xy_vals)
        bounds_left = self._jax_raceline_left_boundary[index]
        bounds_right = self._jax_raceline_right_boundary[index]

        # Show first 3 and last 2 timesteps
        for i in [0, 1, 2, self.planning_horizon - 2, self.planning_horizon - 1]:
            collision_left = lateral_err[i] > bounds_left[i]
            collision_right = lateral_err[i] < -bounds_right[i]
            self.print_info(
                f"  t={i}: lat_err={float(lateral_err[i]):.4f}, "
                f"bounds=[-{float(bounds_right[i]):.4f}, +{float(bounds_left[i]):.4f}], "
                f"collision={'LEFT' if collision_left else ('RIGHT' if collision_right else 'OK')}"
            )
        self.print_info("=" * 60)

    def __plot_planned_trajectory(self, traj):
        self._viz_counter += 1
        if self._viz_counter >= self._viz_skip_count:
            self._viz_counter = 0
            traj_cart = [CartesianState(*state) for state in traj]
            self.plot_trajectory(traj_cart, color=(0, 0, 200))

    def __plot_track_bounds(self, sim_state: CartesianState):
        # Compute track bounds using Cartesian trajectory
        planned_traj = self.rollout_sampled_trajectory(
            jnp.array([*sim_state]), self.planned_controls
        )
        planned_ys = planned_traj.ys[: self.planning_horizon, :]
        xy_vals = planned_ys[:, :2]  # (x, y) positions

        # Project to get progress values and find boundary widths
        _, _, index = self._project_to_frenet_batch(xy_vals)
        bounds_left = jnp.take(self._jax_raceline_left_boundary, index)
        bounds_right = jnp.take(self._jax_raceline_right_boundary, index)

        # Get raceline headings at these progress values
        headings = self._jax_raceline_headings[index]

        # Compute boundary points in Cartesian coordinates
        # Left boundary: offset perpendicular (90 degrees left of heading)
        left_x = xy_vals[:, 0] + bounds_left * jnp.cos(headings + jnp.pi / 2)
        left_y = xy_vals[:, 1] + bounds_left * jnp.sin(headings + jnp.pi / 2)
        # Right boundary: offset perpendicular (90 degrees right of heading)
        right_x = xy_vals[:, 0] - bounds_right * jnp.cos(headings + jnp.pi / 2)
        right_y = xy_vals[:, 1] - bounds_right * jnp.sin(headings + jnp.pi / 2)

        # Convert to CartesianState for plotting (use trajectory velocities)
        bounds_left_cart = [
            CartesianState(
                left_x[i],
                left_y[i],
                headings[i],
                planned_ys[i, 3],
                planned_ys[i, 4],
                planned_ys[i, 5],
            )
            for i in range(self.planning_horizon)
        ]
        bounds_right_cart = [
            CartesianState(
                right_x[i],
                right_y[i],
                headings[i],
                planned_ys[i, 3],
                planned_ys[i, 4],
                planned_ys[i, 5],
            )
            for i in range(self.planning_horizon)
        ]

        self.plot_trajectory(bounds_left_cart, color=(0, 255, 0))
        self.plot_trajectory(bounds_right_cart, color=(0, 255, 0))

    def _project_to_frenet_batch(
        self, xy_vals: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Project Cartesian (x, y) points onto raceline.

        Args:
            xy_vals: (N, 2) array of [x, y] positions

        Returns:
            progress: (N,) progress values along raceline
            lateral_err: (N,) signed lateral errors (positive = left of raceline)
            closest_idx: (N,) indices of closest raceline points
        """
        # Compute squared distances to all raceline points
        # xy_vals: (N, 2), raceline: (M, 2) -> distances: (N, M)
        dx = xy_vals[:, 0:1] - self._jax_raceline_x[None, :]  # (N, M)
        dy = xy_vals[:, 1:2] - self._jax_raceline_y[None, :]  # (N, M)
        dists_sq = dx**2 + dy**2

        # Find closest raceline point index for each trajectory point
        closest_idx = jnp.argmin(dists_sq, axis=1)  # (N,)

        # Get progress at closest points
        progress = self._jax_ss[closest_idx]  # (N,)

        # Compute signed lateral error using cross product with tangent
        raceline_heading = self._jax_raceline_headings[closest_idx]  # (N,)
        dx_closest = xy_vals[:, 0] - self._jax_raceline_x[closest_idx]
        dy_closest = xy_vals[:, 1] - self._jax_raceline_y[closest_idx]

        # Dot product with left-pointing normal: D · N where N = (-sin(h), cos(h))
        # Positive when point is to the LEFT of raceline direction
        lateral_err = -dx_closest * jnp.sin(raceline_heading) + dy_closest * jnp.cos(
            raceline_heading
        )

        return progress, lateral_err, closest_idx

    def __plan_stanley_traj(
        self, sim_state: jax.Array
    ) -> Tuple[List[CartesianState], jax.Array]:
        """Plan a trajectory using the Stanley controller for warm-starting the sampler.

        Args:
            sim_state: Initial Cartesian state [x, y, heading, vf, vs, omega]

        Returns:
            cart_states: List of CartesianState along the trajectory
            control_traj: Array of shape (planning_horizon, 2) with [steering, throttle]
        """
        cart_state = CartesianState(*sim_state)
        cart_states = [cart_state]
        control_traj = jnp.zeros((self.planning_horizon, 2))

        for i in range(self.planning_horizon):
            throttle, steering, _, _ = self.stanley_controller.ctrl_car(
                cart_state, self.track
            )
            control_traj = control_traj.at[i, :].set(jnp.array([steering, throttle]))

            cart_state = DynamicBicycleModelCartesian.advance_dynamics(
                cart_state,
                Control(steering=steering, throttle=throttle),
                self.car,
                dt=self.planning_dt,
            )
            cart_states.append(cart_state)

        return cart_states, control_traj

    def _debug_log_best_cost_components(
        self, state: jax.Array, best_traj_ys: jax.Array
    ):
        """Re-evaluate cost on the best sampled trajectory and log individual components.

        Args:
            state: Current vehicle state (Cartesian) as a JAX array.
            best_traj_ys: State array of the best trajectory, shape (planning_horizon, state_dim).
        """
        ts = jnp.arange(0, self.planning_horizon * self.planning_dt, self.planning_dt)
        traj = RawDiscreteTrajectory(ts, best_traj_ys)
        components = self.__compute_cost_components(state, traj)
        total = sum(components.values())
        parts = " | ".join(f"{k}={float(v):.4g}" for k, v in components.items())
        self.print_info(f"Best cost: {float(total):.4g} ({parts})")
