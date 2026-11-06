"""Fixed-length kinematic-bicycle ("Dubins") car for the DualGuard-MPPI RC-car
experiment (arXiv 2502.01924, Section VI, eq. 15).

State  x = [x, y, heading]            (3 states)
Control u = [V, delta]                (speed, steering angle)
Disturb w = [d_x, d_y]                (additive position-rate disturbance)

    x_dot       = V * cos(heading) + d_x
    y_dot       = V * sin(heading) + d_y
    heading_dot = V * tan(delta) / L

Unlike ``DynamicBicycleCartesian`` this model has no low-/high-speed branch, so
there is no ``jax.lax.cond`` / interval comparison in ``f``.  That makes it
cleanly embeddable via ``immrax.natemb`` for interval reachability with no
comparison hazards.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import radians
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from buzzracer.cars.car import Car

import jax.numpy as jnp
from immrax import System

# Paper Section VI: RC car wheelbase L = 23.5 cm.
DEFAULT_WHEELBASE = 0.235


@dataclass(frozen=True)
class DubinsCarParams:
    """The vehicle constants of a Dubins car: wheelbase plus its control bounds.

    Bundled because eq. 15's dynamics and eq. 15's admissible control set are one
    physical vehicle -- letting `L` and the `V`/`delta` limits be set independently
    at each call site is how a "paper-exact" car quietly stops being paper-exact.
    """

    L: float = DEFAULT_WHEELBASE  # wheelbase (m)
    v_min: float = 0.7  # V lower bound (m/s)
    v_max: float = 1.4  # V upper bound (m/s)
    delta_max: float = radians(25.0)  # steering bound (rad), symmetric


# arXiv 2502.01924, Section VI: L = 23.5 cm, V in [0.7, 1.4] m/s, delta in [-25, 25] deg.
DUALGUARD_RC_CAR = DubinsCarParams()


class DubinsCar(System):
    """3-state Dubins/kinematic-bicycle car with [V, delta] control.

    Parameters
    ----------
    L : float, optional
        Wheelbase in meters (default ``0.235`` from the paper).
    car : Car, optional
        If supplied, the wheelbase is read from the Buzzracer car body params
        (``lf + lr``) instead of ``L``, mirroring ``DynamicBicycleCartesian``.
    """

    def __init__(
        self, L: float = DEFAULT_WHEELBASE, car: Car | None = None
    ) -> None:
        self.evolution = "continuous"
        self.xlen = 3  # car_states = x, y, heading
        self.wlen = 2  # disturbances = d_x, d_y
        self.name = "Dubins Car"

        if car is not None:
            car.init_param()  # matches DynamicBicycleCartesian's param access
            self.L = car.params.lf + car.params.lr
        else:
            self.L = L

    def f(self, t, x: jnp.ndarray, u: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        _x, _y, heading = x
        V, delta = u
        d_x, d_y = w

        return jnp.array(
            [
                V * jnp.cos(heading) + d_x,
                V * jnp.sin(heading) + d_y,
                V * jnp.tan(delta) / self.L,
            ]
        )
