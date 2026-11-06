"""Simulator for an Ackermann steering vehicle with dynamic bicycle model"""
# page 30 of book Vehicle Dynamics and Control
# "forward" states are longitudinal, "sideway" states are lateral

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from buzzracer.cars.car import Car

import jax
import jax.numpy as jnp
from immrax import System, lt
from immrax.comparison import IntervalRelation


# NOTE: duplicated from `tire.py`, modified to use jax.numpy
def tire_curve(slip):
    C = 1.6
    B = 2.3
    D = 1.1
    # C: tail shape
    retval = D * jnp.sin(C * jnp.arctan(B * slip))
    return retval


class DynamicBicycleCartesian(System):
    def __init__(self, car: Car) -> None:
        self.evolution = "continuous"
        self.xlen = 6  # car_states = x, y, heading, x_vel, y_vel, heading_vel
        self.wlen = 2  # disturbances = w_v_sideway, w_v_heading
        self.name = "Dynamic Bicycle Model"

        car.init_param()  # FIXME: This is a horrible hack
        self.lf = car.params.lf
        self.lr = car.params.lr
        self.L = car.params.lf + car.params.lr
        self.Iz = car.params.Iz
        self.m = car.params.m

        print(f"Car body params: lf={self.lf}, lr={self.lr}")

    def f(self, t, x: jnp.ndarray, u: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        x, y, heading, v_forward, v_sideway, omega = x
        a_forward, a_sideway, a_heading = 0.0, 0.0, 0.0
        steering, throttle = u
        w_v_sideway, w_v_heading = w

        def kinematic_model():
            beta = jnp.atan(self.lr / self.L * jnp.tan(steering))
            v_total = v_forward  # kinematic model is only 4 states, stores speed scalar in v_forward

            # motor model
            a_total = (
                6.17
                * (throttle - v_total / 15.2 - 0.333)
                * jnp.where(lt(v_total, jnp.array(0.0)), 0.0, 1.0)
            )
            omega_ref = v_total * jnp.cos(beta) / self.L * jnp.tan(steering)

            return (
                v_total * jnp.cos(heading + beta),
                v_total * jnp.sin(heading + beta) + w_v_sideway,
                omega_ref + w_v_heading,
                a_total,
                -v_sideway,
                -(omega - omega_ref),
            )

        def dynamic_model():
            nonlocal omega
            nonlocal v_sideway
            slip_f = -jnp.atan((omega * self.lf + v_sideway) / v_forward) + steering
            slip_r = jnp.atan((omega * self.lr - v_sideway) / v_forward)

            mass_ratio = self.m * 9.8 / (self.L)
            Ffy = tire_curve(slip_f) * mass_ratio * self.lr
            Fry = 1.15 * tire_curve(slip_r) * mass_ratio * self.lf

            # Dynamics
            a_forward = (
                6.17
                * (throttle - v_forward / 15.2 - 0.333)
                * jnp.where(lt(v_forward, jnp.array(0.0)), 0.0, 1.0)
            )
            a_sideway = 1.0 / self.m * (Fry + Ffy - self.m * v_forward * omega)
            a_heading = 1.0 / self.Iz * (Ffy * self.lf - Fry * self.lr)

            # Conversion to global frame
            v_sideway += w_v_sideway
            omega += w_v_heading

            cos_heading = jnp.cos(heading)
            sin_heading = jnp.sin(heading)

            v_x = v_forward * cos_heading - v_sideway * sin_heading
            v_y = v_forward * sin_heading + v_sideway * cos_heading

            return v_x, v_y, omega, a_forward, a_sideway, a_heading

        # for small longitudinal velocity use kinematic model
        # for tire slip, ratio between lateral and longitudinal speed matters, avoid singularities
        v_x, v_y, omega, a_forward, a_sideway, a_heading = jax.lax.cond(
            lt(
                v_forward,
                0.1 * jnp.ones_like(v_forward),
                # IntervalRelation.ALL,
                IntervalRelation.PRECEDES,
                # | IntervalRelation.FINISHED_BY
                # | IntervalRelation.CONTAINS
                # | IntervalRelation.STARTED_BY,
            ),
            # kinematic_model,
            kinematic_model,
            dynamic_model,
            # dynamic_model,
        )

        return jnp.array(
            [
                v_x,
                v_y,
                omega,
                a_forward,
                a_sideway,
                a_heading,
            ]
        )
