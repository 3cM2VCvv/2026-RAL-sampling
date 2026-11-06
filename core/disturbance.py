"""Shared description of the disturbance acting on a system.

The planner and the plant must agree on the disturbance, but they consume it very
differently: the planner needs a zero-centered *interval* to propagate through the
interval embedding (``icentpert``), while the plant needs a concrete *sample* to add
to its state derivative.  Keeping those as two independent literals -- a list of
half-widths on one side and an ``rng.uniform(-d, d)`` on the other -- lets them
silently drift apart.  ``Disturbance`` is the one object both sides read.

The disturbance is a zero-centered box: ``w in [-half_widths, +half_widths]``, constant
in time and independent of the state.  The interval form is the tightest sound
over-approximation of the sample form, which is what makes the reachsets valid.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from immrax import icentpert


@dataclass(frozen=True)
class Disturbance:
    """A zero-centered box disturbance ``w in [-half_widths, +half_widths]``.

    Parameters
    ----------
    half_widths : sequence of float
        Per-channel half-width.  Its length is the disturbance dimension, and must
        match the ``w`` arity of the system's ``f(t, x, u, w)`` (see ``System.wlen``).
    """

    half_widths: tuple[float, ...]

    def __init__(self, half_widths: Sequence[float]) -> None:
        object.__setattr__(self, "half_widths", tuple(float(w) for w in half_widths))
        if any(w < 0.0 for w in self.half_widths):
            raise ValueError(
                f"half_widths must be non-negative, got {self.half_widths}"
            )

    @property
    def dim(self) -> int:
        return len(self.half_widths)

    def widths(self) -> jax.Array:
        """Half-widths as a device array."""
        return jax.device_put(jnp.asarray(self.half_widths, dtype=float))

    def zero(self) -> jax.Array:
        """The zero disturbance -- what point (non-reachset) rollouts propagate."""
        return jax.device_put(jnp.zeros(self.dim))

    def interval(self):
        """The interval ``[-half_widths, +half_widths]`` -- what reachsets propagate."""
        return icentpert(self.zero(), self.widths())

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Draw one uniform realization from the box -- what the plant applies."""
        hw = np.asarray(self.half_widths)
        return rng.uniform(-hw, hw)


# arXiv 2502.01924, Section VI (eq. 15): the RC car's position-rate disturbance,
# d_x, d_y in [-0.1, 0.1], modelling model mismatch + state-estimation error.
DUALGUARD_RC_DISTURBANCE = Disturbance((0.1, 0.1))
