''' Simulator for an Ackermann steering vehicle with dynamic bicycle model.

Uses the same RHS function as the immrax controller's motion predictor
(`DynamicBicycleCartesian.f`) integrated with a single forward Euler step per
simulator tick. This keeps the simulated dynamics and the controller's
prediction model in lockstep. Disturbances continue to be applied externally
by `Simulator.add_state_noise_*` after the dynamics step.
'''
from __future__ import annotations
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from buzzracer.types import CartesianState
from buzzracer.extensions.simulator import Simulator
from immrax_dynamic_bicycle_cartesian import DynamicBicycleCartesian

if TYPE_CHECKING:
    from buzzracer.cars.car import Car


class DynamicBicycleCartesianSimulator(Simulator):
    ''' Simulator for an Ackermann steering vehicle with dynamic bicycle model.

    advance_dynamics is one forward Euler step against
    `immrax_dynamic_bicycle_cartesian.DynamicBicycleCartesian.f`, JIT-compiled
    once per car at add_car time. Equivalent to
    `compute_trajectory(..., solver="euler", dt=dt)` for a single step.
    '''
    max_v = 3.0
    ''' Maximum speed a car can achieve '''
    using_kinematics = False
    ''' Use Kinematics model instead'''
    state_type = CartesianState

    def __init__(self):
        super().__init__()
        self._predictors: dict[int, DynamicBicycleCartesian] = {}
        self._step_fns: dict[int, callable] = {}

    def init(self):
        super().init()
        for car in self.main.cars:
            self.add_car(car)
        self.main.new_state_update.set()

    def add_car(self, car: Car):
        '''Add a car to use DynamicBicycleCartesianSimulator for state updates
            car needs to (x,y,heading,v_forward,v_sideway,omega)
        '''
        car.sim_state = CartesianState(*car.state)

        car.state_dim = 6
        car.control_dim = 2

        # not implemented: support for artificially added noise
        noise = False
        car.noise = noise
        noise_cov = np.diag([0.01]*6)
        if noise:
            car.noise_cov = noise_cov
            assert np.array(noise_cov).shape == (6, 6)

        # car.state_hist = []
        car.local_states_hist = []
        car.norm = []

        # Build per-car predictor (captures car's lf, lr, Iz, m as Python floats)
        # and a JIT'd Euler step closing over predictor + dt.
        predictor = DynamicBicycleCartesian(car)
        self._predictors[car.id] = predictor

        dt = float(self.main.dt)

        # Inline t=0.0 (Python scalar) and w=jnp.zeros(2) (XLA constant op)
        # rather than capturing pre-allocated device arrays — the latter would
        # trigger a device-to-host transfer during JIT compilation that the
        # ImmraxController's jax_transfer_guard="disallow" blocks.
        @jax.jit
        def step_fn(x, u):
            return x + dt * predictor.f(0.0, x, u, jnp.zeros(2))

        self._step_fns[car.id] = step_fn

        # Warm up the JIT so the first update() tick doesn't pay compile time.
        x_init = jax.device_put(np.asarray(car.state, dtype=np.float32))
        u_init = jax.device_put(np.zeros(2, dtype=np.float32))
        jax.block_until_ready(step_fn(x_init, u_init))

        super().add_car(car)

    def advance_dynamics(self,
                         state,
                         control,
                         car: Car,
                         dt: float,
                         curvature: float = None) -> np.ndarray:
        """Advance dynamics by one forward Euler step using the predictor RHS.

        x_{k+1} = x_k + dt * predictor.f(0.0, x_k, u_k, w=0)

        The dt argument is captured by the JIT'd step closure at add_car time
        (== self.main.dt). It is checked here against the runtime dt to catch
        any caller passing a different timestep.

        Args:
            state: (x, y, heading, v_forward, v_sideway, omega)
            control: (steering, throttle) — radians (left positive), throttle in [-1, 1]
            car: Car object; supplies kinematics/inertia parameters via the cached predictor
            dt: time step in seconds; must equal self.main.dt
            curvature: unused (only for curvilinear)

        Returns:
            Numpy ndarray of shape (6,), float64. Mutability is required by
            Simulator.add_state_noise_lateral_only and friends, which mutate
            car.state in place after this call.
        """
        assert dt == float(self.main.dt), (
            f"DynamicBicycleCartesianSimulator JIT'd step is closed over dt="
            f"{float(self.main.dt)} but advance_dynamics was called with dt={dt}."
        )
        # JIT operates in float32 by default (jax_enable_x64 is off in this
        # project). Pass float32 in to avoid implicit casts that the transfer
        # guard would block; coerce back to float64 on return so downstream
        # numpy code (incl. add_state_noise_*) sees the same dtype as before.
        x_jax = jax.device_put(np.asarray(state, dtype=np.float32))
        u_jax = jax.device_put(np.asarray(control, dtype=np.float32))
        x_next_jax = self._step_fns[car.id](x_jax, u_jax)
        return np.asarray(jax.device_get(x_next_jax), dtype=np.float64)
