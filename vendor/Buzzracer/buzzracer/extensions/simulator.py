''' Base class for all simulators '''
from __future__ import annotations
from typing import TYPE_CHECKING
from time import time, sleep
from abc import ABC, abstractmethod

import numpy as np

from buzzracer.common import ExperimentType
from buzzracer.extensions.extension import Extension
from buzzracer.sysid.vehicle_dynamics import VehicleDynamics
from buzzracer.types import CartesianState, CurvilinearState, Control
if TYPE_CHECKING:
    from buzzracer.cars.car import Car


class Simulator(Extension, ABC):
    '''
    Base class for simulators

    car.state = x,y,heading,v_forward,v_sideway,omega
    however simulator can establish a property car.sim_state
    that use different state representation for simulation
    '''
    state_type: type[CartesianState] | type[CurvilinearState] = CartesianState
    ''' State type used by this simulator, default cartesian'''

    def __init__(self):
        super().__init__(handle_name='simulator')
        self.print_debug_enable()
        self.match_time: bool = False
        ''' If True, attempt to match simulation with clock time. Pauses at each step.'''
        self.print_info('match_time: ' + str(self.match_time))
        self.dynamics_model: type[VehicleDynamics] = VehicleDynamics
        ''' Dynamics model to use for simulation, must be overridden in config
        possible values: KinematicBicycleModelFrenet, DynamicBicycleModelCartesian, etc.'''
        self.state_noise_enabled: bool = None
        ''' If True, enable state noise '''
        self.state_noise_magnitude: float = None
        ''' Noise magnitude, multipled to unit noise of the chosen noise type'''
        self.state_noise_type: str = None
        ''' Type of noise to add '''
        self.state_noise_direction: str = None
        ''' Direction strategy for lateral_only noise:
            uniform - sample w uniformly from within bounds
            adversarial_outside - push car towards outside of track turn
            adversarial_inside - push car towards inside of track turn
            adversarial_away - push car away from the raceline
            None - legacy constant disturbance '''
        self.impulse_state_noise_probability: float = None
        ''' Probability of the impulse state noise being added,(0,1), only used for impulse noise'''
        self.cars: list[Car] = []
        ''' List of all cars using this simulator. This may be a subset of main.cars'''

        self.t0 = None
        self.real_sim_time_ratio = 1.0
        ''' Real time / sim time. If larger than 1.0, simulation will be slowed down.
            This allow easier human interpretation of fast simulations.
            Only useful if match_time = True '''
        self.print_info('real/sim time ratio = %.1f ' %
                        (self.real_sim_time_ratio))

        self.sim_t = 0
        ''' Elapsed time in simulation'''
        self.cars: Car = []

        if self.main.experiment_type != ExperimentType.Simulation:
            self.print_error(
                'Experiment type is not Simulation but a Simulator is loaded')

    def init(self):
        if self.state_noise_enabled:
            assert self.state_noise_type is not None
            assert self.state_noise_magnitude is not None
            self.state_noise_magnitude = np.array(self.state_noise_magnitude)
            noise_type_to_fun = {
                'uniform': self.add_state_noise_uniform,
                'normal': self.add_state_noise_normal,
                'impulse': self.add_state_noise_impulse,
                'lateral_only': self.add_state_noise_lateral_only,
            }
            self.add_state_noise = noise_type_to_fun[self.state_noise_type]
            if self.state_noise_direction is not None:
                valid_directions = ('uniform', 'adversarial_outside', 'adversarial_inside', 'adversarial_away')
                assert self.state_noise_direction in valid_directions, \
                    f'state_noise_direction must be one of {valid_directions}, got {self.state_noise_direction}'

    def add_car(self, car):
        """register a car to use this simulation. """
        self.cars.append(car)

    # TODO use Replay.VehicleDynamics
    @staticmethod
    @abstractmethod
    def advance_dynamics(state: CurvilinearState | CartesianState,
                         control: Control,
                         car: Car,
                         dt: float,
                         curvature: float = None) -> CurvilinearState | CartesianState:
        """advance dynamics by dt.

        Args:
            state: state of the car, may be CartesianState or CurvilinearState
            control: (steering,throttle) steering in rad, left positive, throttle in [-1,1],
                    positive indicates acceleration
            car: Car object, contains information about the car's kinematics,
                also contains car.sim_state for simulators that do not use car.state for update
            dt: Time step to advance dynamics by, unit:seconds
            curvature: signed curvature
        Return:
            state at next time step.
        """
        return

    def update(self):
        for car in self.cars:
            if self.state_type == CartesianState:
                # NOTE cartesian state is passed directly as a tuple for now
                car.state = self.advance_dynamics(
                    car.state, (car.steering, car.throttle), car, self.main.dt)
            elif self.state_type == CurvilinearState:
                control = Control(steering=car.steering, throttle=car.throttle)
                curvature = self.main.track.curvature_s(car.sim_state.progress)
                car.sim_state = self.advance_dynamics(
                    car.sim_state, control, car, self.main.dt, curvature)
                car.state = self.main.track.curv_to_cart(car.sim_state)

        if self.state_noise_enabled:
            self.add_state_noise()
        self.main.new_state_update.set()
        self.sim_t += self.main.dt
        self.match_real_time()

    def match_real_time(self):
        """Sleep to match simulation time to clock time only works when then
        entire simulation loop runs faster than realtime."""
        if not self.match_time:
            return
        if self.t0 is None:
            self.t0 = time()
        time_to_reach = self.sim_t * self.real_sim_time_ratio + self.t0
        self.print_debug('sim_t = %.3f, time = %.3f, expected= %.3f, delta = %.3f' % (
            self.sim_t, time()-self.t0, self.sim_t*self.real_sim_time_ratio, time_to_reach-time()))
        if time_to_reach-time() < 0:
            lag_time = time()-time_to_reach
            self.print_debug("Simulation loop can't keep up ..... lagging %.3f s" % lag_time)

        sleep(max(0, time_to_reach - time()))

    def add_state_noise_normal(self):
        for car in self.cars:
            car.state += np.random.normal(size=car.state.shape) * \
                self.state_noise_magnitude * self.main.dt

    def add_state_noise_uniform(self):
        for car in self.cars:
            car.state += np.random.uniform(low=-1.0, high=1.0, size=car.state.shape) * \
                self.state_noise_magnitude * self.main.dt

    def add_state_noise_impulse(self):
        for car in self.cars:
            val = np.random.uniform()
            if val < self.impulse_state_noise_probability:
                car.state += self.state_noise_magnitude * self.main.dt

    def add_state_noise_lateral_only(self):
        """Reproduce the immrax reachability disturbance model.

        Applies body-frame lateral velocity and heading rate perturbations.
        The disturbance vector w = [w_lat, w_heading] is bounded by
        [1.0, 2.0] * state_noise_magnitude.

        The direction is controlled by state_noise_direction:
        - 'uniform': w sampled uniformly from within bounds
        - 'adversarial_outside': w pushes car towards outside of track turn
        - 'adversarial_inside': w pushes car towards inside of track turn
        - None: legacy constant maximum disturbance
        """
        for car in self.cars:
            heading = car.state[2]
            w = self._compute_lateral_w(car)
            car.state[0] += -w[0] * np.sin(heading) * self.main.dt
            car.state[1] +=  w[0] * np.cos(heading) * self.main.dt
            car.state[2] +=  w[1] * self.main.dt

    def _compute_lateral_w(self, car):
        """Compute lateral disturbance vector [w_lat, w_heading].

        Returns:
            np.ndarray of shape (2,): disturbance magnitudes for lateral
            velocity and heading rate.
        """
        mag = np.array([1.0, 2.0]) * self.state_noise_magnitude

        if self.state_noise_direction == 'uniform':
            return np.random.uniform(-1.0, 1.0, size=2) * mag

        if self.state_noise_direction in ('adversarial_outside', 'adversarial_inside'):
            direction = self._adversarial_lateral_direction(car)
            return direction * mag

        if self.state_noise_direction == 'adversarial_away':
            direction = self._away_from_raceline_direction(car)
            return direction * mag

        return mag

    def _adversarial_lateral_direction(self, car):
        """Determine adversarial direction sign from track curvature.

        Uses signed curvature to identify inside/outside of the current turn.
        On straight sections (near-zero curvature), falls back to the car's
        lateral offset from the raceline.

        Convention (positive curvature = turning left):
        - Outside of a left turn is to the right → direction = -1
        - Outside of a right turn is to the left → direction = +1

        Returns:
            float: +1.0 or -1.0 direction multiplier for the disturbance.
        """
        track = self.main.track
        xy = np.array([car.state[0], car.state[1]])

        # Find nearest raceline point
        dists = np.sum((track.raceline_points - xy.reshape(2, 1))**2, axis=0)
        nearest_idx = np.argmin(dists)
        s = float(track.ss[nearest_idx])

        # Signed curvature: positive = turning left
        kappa = track.curvature_s(s)
        if isinstance(kappa, np.ndarray):
            kappa = float(kappa.flat[0])

        towards_outside = (self.state_noise_direction == 'adversarial_outside')

        if abs(kappa) > 1e-3:
            # Outside is away from center of curvature
            outside_sign = -np.sign(kappa)
            return outside_sign if towards_outside else -outside_sign

        # Straight section: use lateral error relative to raceline
        track_heading = track.raceline_headings[nearest_idx]
        raceline_pt = track.raceline_points[:, nearest_idx]
        tangent = np.array([np.cos(track_heading), np.sin(track_heading)])
        lateral_err = float(np.cross(tangent, xy - raceline_pt))

        if abs(lateral_err) > 1e-3:
            # Push away from raceline for outside, towards for inside
            away_sign = np.sign(lateral_err)
            return away_sign if towards_outside else -away_sign

        # Near center of a straight: default direction
        return 1.0 if towards_outside else -1.0

    def _away_from_raceline_direction(self, car):
        """Determine direction sign that pushes the car away from the raceline.

        Computes the car's lateral offset from the nearest raceline point and
        returns the sign that increases that offset (i.e. pushes the car
        further from the raceline regardless of turn direction).

        Returns:
            float: +1.0 or -1.0 direction multiplier for the disturbance.
        """
        track = self.main.track
        xy = np.array([car.state[0], car.state[1]])

        # Find nearest raceline point
        dists = np.sum((track.raceline_points - xy.reshape(2, 1))**2, axis=0)
        nearest_idx = np.argmin(dists)

        # Lateral error: positive = car is to the left of raceline
        track_heading = track.raceline_headings[nearest_idx]
        raceline_pt = track.raceline_points[:, nearest_idx]
        tangent = np.array([np.cos(track_heading), np.sin(track_heading)])
        lateral_err = float(np.cross(tangent, xy - raceline_pt))

        if abs(lateral_err) > 1e-3:
            return np.sign(lateral_err)

        # On the raceline: default direction
        return 1.0
