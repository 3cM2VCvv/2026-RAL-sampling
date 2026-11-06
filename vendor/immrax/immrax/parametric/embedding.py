import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, List, Literal, Union

import jax
from diffrax import AbstractSolver, Dopri5, Euler, ODETerm, SaveAt, Tsit5, diffeqsolve
from immutabledict import immutabledict
from jaxtyping import ArrayLike, Float, Integer

from ..system import System
from .parametope import Parametope


class ParametricEmbedding(ABC):
    sys: System

    def __init__(self, sys: System):
        self.sys = sys

    @abstractmethod
    def _initialize(self, pt0: Parametope) -> ArrayLike:
        """Initialize the Embedding System for a particular initial set pt0

        Parameters
        ----------
        pt0 : hParametope
            _description_

        Returns
        -------
        ArrayLike
            aux0: Auxilliary states to evolve with the embedding system
        """

    @abstractmethod
    def _dynamics(self, t, state, *args):
        """Embedding dynamics

        Parameters
        ----------
        t : _type_
            _description_
        state : _type_
            _description_
        """

    @partial(
        jax.jit, static_argnums=(0, 4), static_argnames=("solver", "f_kwargs", "inputs")
    )
    def compute_reachset(
        self,
        t0: Union[Integer, Float],
        tf: Union[Integer, Float],
        pt0: Parametope,
        inputs: List[Callable[[int, jax.Array], jax.Array]] = [],
        dt: float = 0.01,
        *,
        solver: Union[Literal["euler", "rk45", "tsit5"], AbstractSolver] = "tsit5",
        f_kwargs: immutabledict = immutabledict({}),
        **kwargs,
    ):
        # print("Recompiling reachset computation")
        def func(t, x, args):
            # Unpack the inputs
            return self._dynamics(t, x, *[u(t, x) for u in inputs], **f_kwargs)

        term = ODETerm(func)
        if solver == "euler":
            solver = Euler()
        elif solver == "rk45":
            solver = Dopri5()
        elif solver == "tsit5":
            solver = Tsit5()
        elif isinstance(solver, AbstractSolver):
            pass
        else:
            raise Exception(f"{solver=} is not a valid solver")

        aux0 = self._initialize(pt0)

        saveat = SaveAt(t0=True, t1=True, steps=True)
        # Save current guard state and temporarily allow transfers for diffrax
        _prev_guard = jax.config.jax_transfer_guard
        jax.config.update("jax_transfer_guard", "allow")
        sol = diffeqsolve(
            term, solver, t0, tf, dt, (pt0, aux0), saveat=saveat, **kwargs
        )
        jax.config.update("jax_transfer_guard", _prev_guard)
        return sol


class ParametopeEmbedding(ParametricEmbedding):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Class 'ParametopeEmbedding' is deprecated. Use 'ParametricEmbedding' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
