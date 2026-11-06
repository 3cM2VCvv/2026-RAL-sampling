"""Buzzracer adapter for vanilla MPPI on the 6-state dynamic bicycle.

Reuses the JAX system (``DynamicBicycleCartesian``) and the JAX point-trajectory
cost (``PointTrajectoryCost``) that ``ImmraxController`` uses with reachsets
disabled. The cost's control-difference regularization is disabled; the
instantaneous control-magnitude term remains. ``JaxMPPI`` supplies only the MPPI
sampling/weighting update.

Sampling std: like ``ShieldMPPIController``, the per-channel std is derived from
``ImmraxController``'s derivative-space ``SampleBounds`` and converted to an
equivalent per-step control std.
"""

import math
import time

import numpy as np
from buzzracer.controllers.car_controller import CarController
from buzzracer.controllers.stanley_car_controller import StanleyCarController
from buzzracer.sysid.dynamic_bicycle_model import DynamicBicycleModelCartesian
from buzzracer.types import CartesianState, Control
from immrax_dynamic_bicycle_cartesian import DynamicBicycleCartesian
from immrax_planner import SampleBounds
from reachset_cost import PointTrajectoryCost
from tube_mppi import JaxMPPI

CONTROL_DIM = 2  # [steering, throttle]


class MPPI(CarController):
    """Vanilla MPPI over the car's dynamic-bicycle model and state cost."""

    def __init__(self, car, config, *, dynamics=None, cost=None):
        # Defaults are set before ``super().__init__`` so XML attributes win.
        self.horizon = 30
        self.prng_seed = 0
        self.centering_weight_factor = 1.0
        self.velocity_scale_factor = 1.0
        self.mppi_num_samples = 1024
        self.mppi_param_lambda = 1.0

        super().__init__(car, config)

        self.planning_dt = float(getattr(self.main, "dt", 0.02))
        self.planning_horizon = int(self.horizon)
        self.num_samples = int(self.mppi_num_samples)

        # Same sampling bounds ImmraxController uses (immrax_controller.py).
        throttle_range = car.max_throttle - car.min_throttle
        self.steering_dt_bounds = SampleBounds(
            -1.5 * math.pi, 0.75 * math.pi, 1.5 * math.pi
        )
        self.throttle_dt_bounds = SampleBounds(
            -4.0 * throttle_range, throttle_range / 2.0, throttle_range
        )
        self.steering_bounds = SampleBounds(
            -car.max_steering_left, car.max_steering_right, car.max_steering_right
        )
        self.throttle_bounds = SampleBounds(
            car.min_throttle, car.max_throttle / 2.0, car.max_throttle
        )

        self.dynamics = dynamics or DynamicBicycleCartesian(car)
        self.rollout_cost = cost or self._make_cost()

        self.planner = self._build_planner()

        self._timing_log = []
        self.planner.warmup(np.asarray(self.car.state[:6], dtype=np.float32))

    def _make_cost(self):
        return PointTrajectoryCost(
            self.track,
            self.planning_horizon,
            self.planning_dt,
            centering_weight_factor=self.centering_weight_factor,
            velocity_scale_factor=self.velocity_scale_factor,
            control_diff_weight=0.0,
        )

    def _build_planner(self):
        return JaxMPPI(
            self.dynamics,
            self.rollout_cost,
            control_bounds=[self.steering_bounds, self.throttle_bounds],
            control_std=self._input_std(),
            initial_controls=self._stanley_plan(),
            num_samples=self.num_samples,
            planning_horizon=self.planning_horizon,
            planning_dt=self.planning_dt,
            param_lambda=float(self.mppi_param_lambda),
            prng_seed=int(self.prng_seed),
        )

    def _input_std(self):
        """Per-channel std of the control input [steering, throttle]."""
        # Derivative-space samples integrate to a control variance
        # dt^2 * sigma_du^2 * (k + 1) at step k; match the horizon average.
        horizon_gain = math.sqrt((self.planning_horizon + 1) / 2.0)
        return [
            self.planning_dt * self.steering_dt_bounds.std * horizon_gain,
            self.planning_dt * self.throttle_dt_bounds.std * horizon_gain,
        ]

    def _stanley_plan(self):
        """Seed the plan with the same Stanley rollout ImmraxController uses."""
        self.stanley_controller = StanleyCarController(self.car, self.config)
        state = np.asarray(self.car.state[:6], dtype=float)
        cart_state = CartesianState(*state)
        controls = np.zeros((self.planning_horizon, CONTROL_DIM), dtype=np.float32)
        for i in range(self.planning_horizon):
            throttle, steering, _, _ = self.stanley_controller.ctrl_car(
                cart_state, self.track
            )
            controls[i, 0] = float(steering)
            controls[i, 1] = float(throttle)
            cart_state = DynamicBicycleModelCartesian.advance_dynamics(
                cart_state,
                Control(steering=float(steering), throttle=float(throttle)),
                self.car,
                dt=self.planning_dt,
            )
        return controls

    def _compute_control(self, state):
        control0, _ = self.planner.step(state)
        return control0

    def _apply_control(self, control0):
        self.car.steering = float(control0[0])
        self.car.throttle = float(control0[1])

    def control(self):
        start = time.perf_counter()
        control0 = self._compute_control(
            np.asarray(self.car.state[:6], dtype=np.float32)
        )
        self._apply_control(control0)

        self._timing_log.append((time.perf_counter() - start) * 1000.0)
        return True
