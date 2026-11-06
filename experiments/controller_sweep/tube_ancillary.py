"""iLQG ancillary controller for Tube-MPPI.

Solves a finite-horizon tracking problem toward the nominal state/control
trajectory with iterative LQG (Todorov & Li, 2005): linearize the explicit-Euler
step, run a regularized backward Riccati pass, then a forward pass with line
search.  Only the first control is consumed, but the full solve provides the
feedback correction for the current state deviation.
"""

from functools import partial

import jax
import jax.numpy as jnp


class ILQGTracker:
    """Tracking controller on a continuous ``immrax`` system.

    Parameters
    ----------
    dynamics : immrax.System
        System whose ``f(t, x, u, w)`` defines one Euler step at ``dt``.
    control_bounds : list[SampleBounds]
        Per-channel control limits; length sets the control dimension.
    horizon : int
        Number of control steps in the ancillary problem.
    dt : float
        Step size; must match the nominal planner's ``planning_dt``.
    track_weights : sequence[float]
        Diagonal state-tracking weights (positions before velocities).
    control_weights : sequence[float]
        Diagonal control-tracking weights.
    """

    def __init__(
        self,
        dynamics,
        control_bounds,
        *,
        horizon: int,
        dt: float,
        track_weights=(5.0, 5.0, 2.0, 1.0, 1.0, 0.5),
        control_weights=(1.0, 5.0),
        q_scale: float = 1.0,
        r_scale: float = 1.0,
        terminal_scale: float = 10.0,
        iterations: int = 5,
        mu_init: float = 1e-2,
        mu_factor: float = 5.0,
        mu_min: float = 1e-6,
        mu_max: float = 1e6,
        alphas=(1.0, 0.5, 0.25, 0.125),
    ):
        self.dynamics = dynamics
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.state_dim = int(dynamics.xlen)
        self.control_dim = len(control_bounds)

        self.u_min = jnp.asarray([b.min for b in control_bounds], dtype=jnp.float32)
        self.u_max = jnp.asarray([b.max for b in control_bounds], dtype=jnp.float32)
        self.zero_w = jnp.zeros(int(dynamics.wlen), dtype=jnp.float32)

        self.q = q_scale * jnp.asarray(track_weights, dtype=jnp.float32)
        self.r = r_scale * jnp.asarray(control_weights, dtype=jnp.float32)
        self.qf = terminal_scale * self.q
        self.q_diag = jnp.diag(self.q)
        self.r_diag = jnp.diag(self.r)
        self.qf_diag = jnp.diag(self.qf)

        self.iterations = int(iterations)
        self.mu_init = float(mu_init)
        self.mu_factor = float(mu_factor)
        self.mu_min = float(mu_min)
        self.mu_max = float(mu_max)
        self.alphas = jnp.asarray(alphas, dtype=jnp.float32)
        self._eye = jnp.eye(self.control_dim, dtype=jnp.float32)

    # ------------------------------------------------------------------
    # Model and cost
    # ------------------------------------------------------------------
    def _step(self, x, u):
        u = jnp.clip(u, self.u_min, self.u_max)
        return x + self.dt * self.dynamics.f(0.0, x, u, self.zero_w)

    def _linearize(self, x, u):
        return (
            jax.jacfwd(self._step, argnums=0)(x, u),
            jax.jacfwd(self._step, argnums=1)(x, u),
        )

    @staticmethod
    def _wrap(angle):
        return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))

    def _state_error(self, x, x_ref):
        err = x - x_ref
        # Only the heading needs wrapping; positions and rates do not.
        return err.at[2].set(self._wrap(err[2]))

    def _cost(self, xs, us, x_ref, u_ref):
        dx = jax.vmap(self._state_error)(xs[:-1], x_ref[:-1])
        du = us - u_ref
        running = 0.5 * jnp.sum(self.q * dx**2, axis=-1) + 0.5 * jnp.sum(
            self.r * du**2, axis=-1
        )
        d_terminal = self._state_error(xs[-1], x_ref[-1])
        terminal = 0.5 * jnp.sum(self.qf * d_terminal**2)
        return jnp.sum(running) + terminal

    def _rollout(self, x0, us):
        def body(x, u):
            xn = self._step(x, u)
            return xn, xn

        _, xs_tail = jax.lax.scan(body, x0, us)
        return jnp.concatenate([x0[None], xs_tail], axis=0)

    # ------------------------------------------------------------------
    # iLQG passes
    # ------------------------------------------------------------------
    def _backward(self, xs, us, x_ref, u_ref, mu):
        def body(carry, inputs):
            vx, vxx = carry
            x, u, xr, ur = inputs

            a, b = self._linearize(x, u)
            dx = self._state_error(x, xr)
            du = u - ur

            q_x = self.q * dx + a.T @ vx
            q_u = self.r * du + b.T @ vx
            q_xx = self.q_diag + a.T @ vxx @ a
            q_uu = self.r_diag + b.T @ vxx @ b + mu * self._eye
            q_xu = a.T @ vxx @ b

            k = -jnp.linalg.solve(q_uu, q_u)
            gain = -jnp.linalg.solve(q_uu, q_xu.T)

            vx_new = q_x + gain.T @ q_u
            vxx_new = q_xx + gain.T @ q_uu @ gain + q_xu @ gain + gain.T @ q_xu.T
            vxx_new = 0.5 * (vxx_new + vxx_new.T)
            return (vx_new, vxx_new), (k, gain)

        d_terminal = self._state_error(xs[-1], x_ref[-1])
        vx = self.qf * d_terminal
        vxx = self.qf_diag
        _, (ks, gains) = jax.lax.scan(
            body,
            (vx, vxx),
            (xs[:-1][::-1], us[::-1], x_ref[:-1][::-1], u_ref[::-1]),
        )
        return ks[::-1], gains[::-1]

    def _forward(self, x0, us, ks, gains, xs, alpha):
        def body(x, inputs):
            u_bar, k, gain, x_bar = inputs
            u = u_bar + alpha * k + gain @ (x - x_bar)
            u = jnp.clip(u, self.u_min, self.u_max)
            xn = self._step(x, u)
            return xn, (xn, u)

        _, (xs_tail, us_new) = jax.lax.scan(
            body, x0, (us, ks, gains, xs[:-1])
        )
        return jnp.concatenate([x0[None], xs_tail], axis=0), us_new

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnums=0)
    def solve(self, x0, x_ref, u_ref):
        """Return the first control of the best tracking trajectory."""
        us0 = jnp.clip(u_ref, self.u_min, self.u_max)
        cost0 = self._cost(self._rollout(x0, us0), us0, x_ref, u_ref)

        def iteration(carry, _):
            us, mu, cost_prev = carry
            xs = self._rollout(x0, us)
            ks, gains = self._backward(xs, us, x_ref, u_ref, mu)

            def evaluate(alpha):
                xs_new, us_new = self._forward(x0, us, ks, gains, xs, alpha)
                return self._cost(xs_new, us_new, x_ref, u_ref), xs_new, us_new

            costs, xs_all, us_all = jax.vmap(evaluate)(self.alphas)
            best = jnp.argmin(costs)
            improved = costs[best] < cost_prev

            us_new = jnp.where(improved, us_all[best], us)
            cost_new = jnp.where(improved, costs[best], cost_prev)
            mu_new = jnp.where(
                improved,
                jnp.maximum(mu / self.mu_factor, self.mu_min),
                jnp.minimum(mu * self.mu_factor, self.mu_max),
            )
            return (us_new, mu_new, cost_new), (cost_new, mu_new)

        (us, mu, cost), (cost_history, mu_history) = jax.lax.scan(
            iteration,
            (us0, jnp.asarray(self.mu_init, dtype=jnp.float32), cost0),
            None,
            length=self.iterations,
        )
        info = {
            "cost": cost,
            "cost_history": cost_history,
            "mu": mu,
            "mu_history": mu_history,
        }
        return us[0], info
