"""Buzzracer adapter that drives the Shield-MPPI submodule for the car.

The submodule supplies the MPPI weights/constraint utilities and the vendor
DCBF cost.  This module supplies the 6-state dynamic bicycle and a 2D control
input ``[steering, throttle]``.  Rollouts are deterministic; the DCBF penalty
is added by ``MPPIController.compute_cbf_cost``.

Sampling std: ImmraxController's ``SampleBounds`` are on the control
*derivative* (one Gaussian perturbation per step, integrated by cumsum).
Integrate that variance over the horizon and convert to an equivalent per-step
control std: ``sigma_u = dt * sigma_du * sqrt((N + 1) / 2)``.

The solver's own ``param_gamma`` free-energy term (``lambda * (1 - alpha)``
times ``z^T Sigma^-1 (z + eps)``) is the only input regularization; the rollout
cost contains no control terms.  ``shield_mppi_cost`` supplies only Immrax's
state terms.

After the MPPI mean is computed, the paper's local repair (BFGS over the first
``shield_repair_horizon`` controls) is applied and its first control is
executed.  The pre-repair mean, not the repaired sequence, warm-starts the next
planning step.
"""

import importlib.util
import math
import os
import sys
import time

import numpy as np
import scipy.optimize
import torch
from buzzracer.controllers.car_controller import CarController
from buzzracer.controllers.stanley_car_controller import StanleyCarController
from buzzracer.sysid.dynamic_bicycle_model import DynamicBicycleModelCartesian
from buzzracer.types import CartesianState, Control
from immrax_planner import SampleBounds
from shield_mppi_cost import ImmraxRolloutCost, cbf_half_width
from shield_mppi_dynamics import DynamicBicycleTorch

_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
)
_SOLVER_PATH = os.path.join(_ROOT, "vendor", "Shield-MPPI", "solver", "shieldmppi.py")

CONTROL_DIM = 2  # [steering, throttle]


def _load_solver_class():
    """Load ``MPPIController`` from the submodule by path.

    ``solver/`` has no package ``__init__`` and would shadow unrelated
    top-level names, so it is not added to ``sys.path``.
    """
    module_name = "shield_mppi_solver"
    if module_name in sys.modules:
        return sys.modules[module_name].MPPIController
    spec = importlib.util.spec_from_file_location(module_name, _SOLVER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.MPPIController


class ShieldMPPIController(CarController):
    """MPPI with Shield-MPPI's DCBF cost and local repair over the Immrax car.

    Keyword arguments let callers substitute the dynamics or cost evaluator;
    the defaults reproduce ``ImmraxController``'s model and state cost terms.
    """

    def __init__(self, car, config, *, dynamics=None, cost=None):
        # Defaults are set before ``super().__init__`` so XML attributes win.
        self.horizon = 30
        self.prng_seed = 0
        self.centering_weight_factor = 1.0
        self.velocity_scale_factor = 1.0
        self.shield_num_samples = 1024
        self.shield_param_lambda = 1.0
        self.shield_param_alpha = 0.0
        self.shield_param_beta = 0.5
        self.shield_cbf_cost_weight = 1000.0
        self.shield_w_T = 0.0
        self.shield_repair_horizon = 10
        self.shield_repair_steps = 20
        self.shield_device = ""

        super().__init__(car, config)

        self.planning_dt = float(getattr(self.main, "dt", 0.02))
        self.planning_horizon = int(self.horizon)
        self.num_samples = int(self.shield_num_samples)
        self.repair_horizon = int(self.shield_repair_horizon)
        self.repair_steps = int(self.shield_repair_steps)
        self.device = torch.device(
            self.shield_device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        torch.manual_seed(int(self.prng_seed))

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

        self.dynamics = dynamics or DynamicBicycleTorch(car)
        self.sigma_diag = torch.as_tensor(
            self._input_std(),
            dtype=torch.float32,
            device=self.device,
        )

        self.cost = cost or ImmraxRolloutCost(
            self.track,
            self.planning_horizon,
            self.planning_dt,
            centering_weight_factor=self.centering_weight_factor,
            velocity_scale_factor=self.velocity_scale_factor,
            device=self.device,
        )

        self.min_input = torch.tensor(
            [self.steering_bounds.min, self.throttle_bounds.min],
            dtype=torch.float32,
            device=self.device,
        )
        self.max_input = torch.tensor(
            [self.steering_bounds.max, self.throttle_bounds.max],
            dtype=torch.float32,
            device=self.device,
        )

        self.w_T = float(self.shield_w_T) or 2.0 * cbf_half_width(self.track)
        self.mppi = self._build_solver()
        self._warm_start()

        self._timing_log = []

    # ------------------------------------------------------------------
    # Sampling covariance
    # ------------------------------------------------------------------
    def _input_std(self):
        """Per-channel std of the control input [steering, throttle]."""
        # Derivative-space samples integrate to a control variance
        # dt^2 * sigma_du^2 * (k + 1) at step k; match the horizon average.
        horizon_gain = math.sqrt((self.planning_horizon + 1) / 2.0)
        return [
            self.planning_dt * self.steering_dt_bounds.std * horizon_gain,
            self.planning_dt * self.throttle_dt_bounds.std * horizon_gain,
        ]

    # ------------------------------------------------------------------
    # Solver setup
    # ------------------------------------------------------------------
    def _build_solver(self):
        solver_cls = _load_solver_class()
        sigma = torch.diag(self.sigma_diag)
        ref_track_path = torch.as_tensor(
            np.column_stack(
                [
                    self.track.raceline_points.T,
                    self.track.raceline_headings,
                    self.track.raceline_velocity,
                ]
            ),
            dtype=torch.float32,
        )
        solver = solver_cls(
            delta_t=self.planning_dt,
            time_step=self.planning_horizon,
            batch_size=self.num_samples,
            sigma=sigma,
            ref_track_path=ref_track_path,
            param_lambda=self.shield_param_lambda,
            param_alpha=self.shield_param_alpha,
            cbf_cost_weight=self.shield_cbf_cost_weight,
            w_T=self.w_T,
            beta=self.shield_param_beta,
            device=str(self.device),
        )
        # The adapter's rollouts are 6-state; only ``state`` needs resizing.
        solver.dim_x = 6
        solver.state = torch.zeros(6, device=self.device)
        solver.min_v = self.min_input
        solver.max_v = self.max_input
        return solver

    def _warm_start(self):
        """Seed the controls with the same Stanley rollout ImmraxController uses."""
        self.stanley_controller = StanleyCarController(self.car, self.config)
        state = np.asarray(self.car.state[:6], dtype=float)
        cart_state = CartesianState(*state)
        controls = torch.zeros(
            (self.planning_horizon, CONTROL_DIM), dtype=torch.float32
        )
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
        self.mppi.u_prev.copy_(controls.to(self.device))

    # ------------------------------------------------------------------
    # Planner step
    # ------------------------------------------------------------------
    def _sample_input_noise(self, batch, horizon):
        noise = torch.empty(
            batch, horizon, CONTROL_DIM, dtype=torch.float32, device=self.device
        )
        for channel in range(CONTROL_DIM):
            noise[..., channel].normal_(0.0, float(self.sigma_diag[channel]))
        return noise

    def predict_trajectory(self, initial_state, controls):
        return self.dynamics.rollout(initial_state, controls, self.planning_dt)

    def compute_total_cost(self, trajectory):
        return self.cost(self.mppi.state, trajectory)

    def compute_control_input(self):
        """One MPPI update over the sampled controls."""
        solver = self.mppi
        plan = solver.u_prev
        noise = self._sample_input_noise(self.num_samples, self.planning_horizon)
        sampled = solver.apply_constraint(plan.unsqueeze(0) + noise)
        noise = sampled - plan.unsqueeze(0)

        trajectory = self.predict_trajectory(solver.state, sampled)
        ref_path = solver.get_nearest_waypoints(trajectory, 200)
        costs = self.compute_total_cost(trajectory)
        costs = costs + solver.compute_cbf_cost(trajectory, ref_path)

        # Submodule free-energy term: lambda*(1-alpha) * z^T Sigma^-1 (z + eps).
        quad_term = (plan.unsqueeze(0) * solver.sigma_inv_diag * sampled).sum(dim=-1)
        costs = costs + solver.param_gamma * quad_term.sum(dim=1)

        weights = solver.compute_weights(costs)
        updated = solver.apply_constraint(
            plan + (weights.view(-1, 1, 1) * noise).sum(dim=0)
        )
        solver.u_prev.copy_(updated)
        return updated, trajectory

    # ------------------------------------------------------------------
    # Local repair
    # ------------------------------------------------------------------
    def _local_repair(self, mean, x0):
        """Paper's local repair: BFGS on the DCBF violation, first N controls."""
        n = min(self.repair_horizon, self.planning_horizon)
        if n <= 0:
            return mean

        solver = self.mppi
        x0 = torch.as_tensor(x0, dtype=torch.float32, device=self.device).reshape(-1)
        mean_traj = self.predict_trajectory(x0, mean.unsqueeze(0)).squeeze(0)
        ref_path = solver.get_nearest_waypoints(mean_traj.unsqueeze(0), 200).squeeze(0)
        ref_xy = ref_path[:n, :2]
        alpha = solver.alpha
        h = solver.h

        def violation(controls):
            controls = torch.cat([controls, mean[n:]], dim=0)
            traj = self.predict_trajectory(x0, controls.unsqueeze(0)).squeeze(0)
            xy = traj[:n, :2]
            h_next = h(xy, ref_xy)
            h_prev = torch.cat(
                [h(x0[:2].unsqueeze(0), ref_xy[:1]), h(xy[:-1], ref_xy[:-1])],
                dim=0,
            )
            return torch.clamp(-h_next + alpha * h_prev, min=0.0).sum()

        with torch.no_grad():
            before = float(violation(mean[:n].detach().clone()))
        if before <= 0.0:
            return mean

        def fun_and_grad(flat):
            controls = (
                torch.as_tensor(flat, dtype=torch.float32, device=self.device)
                .reshape(n, CONTROL_DIM)
                .requires_grad_(True)
            )
            loss = violation(controls)
            grad = torch.autograd.grad(loss, controls)[0]
            return (
                float(loss.detach()),
                grad.detach().cpu().numpy().astype(float).reshape(-1),
            )

        bounds = [
            (float(self.min_input[0]), float(self.max_input[0])),
            (float(self.min_input[1]), float(self.max_input[1])),
        ] * n
        result = scipy.optimize.minimize(
            fun_and_grad,
            mean[:n].detach().cpu().numpy().reshape(-1).astype(float),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.repair_steps},
        )

        safe = mean.clone()
        safe[:n] = torch.as_tensor(
            result.x.reshape(n, CONTROL_DIM),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            after = float(violation(safe[:n].detach().clone()))
        return safe if after <= before else mean

    # ------------------------------------------------------------------
    # CarController interface
    # ------------------------------------------------------------------
    def control(self):
        start = time.perf_counter()
        state = np.asarray(self.car.state[:6], dtype=np.float32)
        self.mppi.set_state(torch.as_tensor(state, device=self.device))
        mean, _ = self.compute_control_input()
        safe = self._local_repair(mean, state)

        self.car.steering = float(safe[0, 0])
        self.car.throttle = float(safe[0, 1])

        self._timing_log.append((time.perf_counter() - start) * 1000.0)
        return True
