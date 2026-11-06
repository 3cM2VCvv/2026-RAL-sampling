"""RC car 3-state kinematic bicycle: dynamics, BRT tables, and the MPPI update.

Numerics are the paper authors' ``rc_car_jax.ipynb`` unchanged -- forward Euler at
DT = 0.02, both safety thresholds at 0.1, the same cost weights, temperature and MPPI
update.  Two things are deliberately left as they were and flagged where they occur:

* the PRNG key is never split, so every control step draws the *same* block of noise
  (``buzzracer_model.py`` fixes this; fixing it here would change the trajectory);
* ``exp(-TEMPERATURE * cost)`` has no ``min(cost)`` baseline subtraction, unlike
  ``MPPI._command``.  The effective sample size is logged so the effect is visible.

The one substitution is the grid lookup: :class:`grids.NearestGrid` replaces the
notebook's ``argmin`` scan over each axis.  Both return the same cell on a uniform
grid, and both clamp an out-of-range query to the nearest edge cell, so this is a
speed change only.
"""

import jax
import jax.numpy as jnp
import mat73  # brt_rc_wh_track.mat is MATLAB -v7.3; scipy.io cannot read it
import numpy as np
from jax import jit, random

from .grids import NearestGrid

LABELS = ["x", "y", "theta"]

DT = 0.02
L = 0.235  # wheelbase
V_MIN, V_MAX = 0.7, 1.4
DELTA_MIN, DELTA_MAX = -0.436, 0.436
U_MIN = jnp.array([V_MIN, DELTA_MIN])
U_MAX = jnp.array([V_MAX, DELTA_MAX])

HORIZON = 100
NUM_THREADS = 1000  # K
INITIAL_STATE = jnp.array([3.0, 0.5, 0.0])  # BRT grid coordinates
NOISE_SCALE = jnp.array([0.2, 0.4])  # (velocity, steering)

# --- cost weights and temperature -------------------------------------------
W_VEL, W_CENTERING, W_COLLISION = 1.0, 120.0, 50.0
W_IN_BRT = 0.0  # the DualGuard profile costs obstacles, not the BRT
L_CENTER = 0.45  # max l(x) on this track, the centering reference
TEMPERATURE = 0.005 * 4

# --- safety filter ----------------------------------------------------------
GUARD2_THRESHOLD = 0.1  # on the executed control
GUARD1_THRESHOLD = 0.1  # inside the rollouts


@jit
def step_euler(state, control, dt=DT):
    """Ackermann kinematics, forward Euler, heading wrapped to (-pi, pi]."""
    x, y, theta = state
    v, delta = control
    nxt = jnp.array(
        [
            x + v * jnp.cos(theta) * dt,
            y + v * jnp.sin(theta) * dt,
            theta + v * jnp.tan(delta) / L * dt,
        ]
    )
    return nxt.at[2].set((nxt[2] + jnp.pi) % (2 * jnp.pi) - jnp.pi)


def load_brt(path):
    """Read the ``brt_rc_wh`` stage's output into plain numpy arrays.

    ``Deriv`` (3 x 608 x 408 x 181 float64 = 1.0 GiB) is not read: ``uOpt`` already
    carries the optimal control the gradients would be used to recompute.
    """
    raw = mat73.loadmat(path, only_include=["Vx", "lx", "tau2", "uOpt", "g"])
    axes = [np.asarray(a).squeeze() for a in raw["g"]["vs"]]
    brt = {
        "grid_axes": axes,
        "value": np.asarray(raw["Vx"], dtype=np.float32),
        "lx": np.asarray(raw["lx"], dtype=np.float32),
        "uOpt": [
            np.asarray(raw["uOpt"][i]).squeeze().astype(np.float32) for i in (0, 1)
        ],
        # World frame of the reconstructed track vs. this grid: reconstruct_track.py
        # shifts the grid's min corner to the origin, so track coords are grid coords
        # minus `grid_min`.  Lap detection needs the racing line in *grid* coords.
        "grid_min": np.asarray(raw["g"]["min"], dtype=np.float64).ravel(),
    }

    shape = tuple(a.size for a in axes)
    assert brt["value"].shape == shape, f"Vx {brt['value'].shape} != grid {shape}"
    assert brt["lx"].shape == shape[:2], "lx is expected to depend on (x, y)"
    assert all(a.shape == shape for a in brt["uOpt"])

    print(f"{path}\n  grid {shape}, tau 0 .. {raw['tau2'][-1]:.1f} s")
    for lab, a in zip(LABELS, axes):
        print(
            f"    {lab:<6s} [{a[0]:7.3f}, {a[-1]:7.3f}]  N {a.size:4d}  d {a[1] - a[0]:.4f}"
        )
    print(
        f"  V in [{brt['value'].min():.4f}, {brt['value'].max():.4f}] "
        f"(V > 0 safe), lx in [{brt['lx'].min():.4f}, {brt['lx'].max():.4f}], "
        f"uOpt {np.unique(brt['uOpt'][0])} m/s, {np.unique(brt['uOpt'][1])} rad"
    )
    return brt


def build_sim(brt):
    """Bind the device tables into the jitted queries, rollout and MPPI update."""
    grid = NearestGrid(brt["grid_axes"])
    lx_grid = NearestGrid(brt["grid_axes"][:2])
    value_f, lx_f = NearestGrid.flatten(brt["value"]), NearestGrid.flatten(brt["lx"])
    uopt_f = [NearestGrid.flatten(a) for a in brt["uOpt"]]

    # No fill: an out-of-range query clamps to the nearest edge cell, as the notebook's
    # argmin did, and the theta axis clamps rather than wrapping for the same reason (it
    # runs to pi - dx, with no duplicate slice at +pi).  buzzracer_model.py reports
    # out-of-grid states as unsafe instead, because in 6-D the velocity states have
    # bounds a spinning car can exceed; here x and y are bounded by the track and theta
    # by the dynamics, so there is nowhere the state can go that the grid does not cover.
    value_q = jit(lambda s: grid.gather(value_f, s))
    # The margin function: the BRT's own l(x), the signed distance the solve was run
    # against.  Negative means the car is through the track boundary.
    margin_q = jit(lambda s: lx_grid.gather(lx_f, s[:2]))

    @jit
    def ctrl_q(state):
        flat, _ = grid.locate(state)
        return jnp.array([uopt_f[0][flat], uopt_f[1][flat]])

    @jit
    def rollout(state0, u_seq):
        """One hallucinated trajectory with the sample-level filter (guard 1).

        The filter runs on the value at the pre-step state and the applied actions are
        returned, so the MPPI update can be taken with respect to them.
        """

        def step(state, u):
            unsafe = value_q(state) < GUARD1_THRESHOLD
            applied = jnp.where(unsafe, ctrl_q(state), u)
            nxt = step_euler(state, applied)
            return nxt, (nxt, applied, unsafe)

        _, (states, applied, unsafe) = jax.lax.scan(step, state0, u_seq)
        lx, v = jax.vmap(margin_q)(states), jax.vmap(value_q)(states)
        cost = (
            W_VEL * jnp.sum((applied[:, 0] - V_MAX) ** 2)
            + W_CENTERING * jnp.sum(L_CENTER - lx)
            + W_COLLISION * jnp.sum(lx < 0.0)
            + W_IN_BRT * jnp.sum(v < 0.0)
        )
        return states, applied, unsafe, cost

    rollout_batch = jax.vmap(rollout, in_axes=(None, 0))

    @jit
    def mppi_update(U, key, state):
        """The notebook's update: no baseline subtraction, weights ~ exp(-T * cost)."""
        perturbed = jnp.clip(
            U[None] + random.normal(key, (NUM_THREADS, HORIZON, 2)) * NOISE_SCALE,
            U_MIN,
            U_MAX,
        )
        states, applied, unsafe, costs = rollout_batch(state, perturbed)

        w = jnp.exp(-TEMPERATURE * costs)
        # Deltas taken against the applied (post-filter) actions, so safety overrides
        # fold back into the nominal plan.
        U_new = jnp.clip(
            U + jnp.sum(w[:, None, None] * (applied - U[None]), axis=0) / jnp.sum(w),
            U_MIN,
            U_MAX,
        )
        # ESS as 1/sum(p^2) on the *normalised* weights rather than the textbook
        # sum(w)^2/sum(w^2).  Without the baseline subtraction the raw weights run
        # around 1e-30 on this cost scale, and both of those sums then underflow
        # float32 to zero -- the update is fine (the scale cancels) but the diagnostic
        # comes out 0/0.  Normalising first is the same number, computed safely.
        ess = 1.0 / jnp.sum((w / jnp.sum(w)) ** 2)
        return U_new, states, unsafe, costs, ess

    return dict(value=value_q, margin=margin_q, ctrl=ctrl_q, update=mppi_update)


def initial_plan():
    """The notebook's warm start: roll forward at 1 m/s, steering straight."""
    return jnp.stack([jnp.ones(HORIZON), jnp.zeros(HORIZON)], axis=1)
