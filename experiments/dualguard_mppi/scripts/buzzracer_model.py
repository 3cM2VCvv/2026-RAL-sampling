"""Buzzracer 6-state dynamic bicycle: dynamics, BRT tables, and the MPPI update.

Everything here is the model layer -- what the car is, what the BRT says about it, and
how ``controller_mppi.py`` would drive it.  ``buzzracer_jax.py`` owns the experiment.
"""

import jax
import jax.numpy as jnp
import mat73  # the BRT export is MATLAB -v7.3 (HDF5); scipy.io cannot read it
import numpy as np
from jax import jit, random

from .grids import OOB_FILL, NearestGrid

LABELS = ["x", "y", "psi", "v_fwd", "v_side", "omega"]

DT = 0.02
HORIZON = 100  # 2.0 s = the BRT's tMax
NUM_THREADS = 1000  # K
INITIAL_STATE = jnp.array([2.1, 2.1, -jnp.pi / 2, 1.0, 0.0, 0.0], dtype=jnp.float32)

# --- input bounds (brt_buzzracer_track.m) -----------------------------------
DELTA_MAX, THROTTLE_MIN, THROTTLE_MAX = np.deg2rad(26.1), -1.0, 1.0
U_MIN = jnp.array([-DELTA_MAX, THROTTLE_MIN], dtype=jnp.float32)
U_MAX = jnp.array([DELTA_MAX, THROTTLE_MAX], dtype=jnp.float32)
U_INIT = jnp.zeros(2, dtype=jnp.float32)  # appended when the plan shifts forward

# --- MPPI -------------------------------------------------------------------
# Noise scales carried over from the rc_car experiment as ratios of the input range,
# since the inputs changed from (speed, steering) to (steering, throttle).
NOISE_SIGMA = jnp.array([0.40, 0.60], dtype=jnp.float32)
NOISE_MU = jnp.zeros(2, dtype=jnp.float32)
SIGMA_INV_DIAG = jnp.asarray(1.0 / np.asarray(NOISE_SIGMA) ** 2, dtype=jnp.float32)
LAMBDA = 10.0  # controller_mppi.py: lambda_
NOISE_ABS_COST = True  # controller_mppi.py default
U_INIT_IS_MEAN = False  # default_config.yaml: mppi_initial_control_is_mean

# --- cost weights -----------------------------------------------------------
# Set against the smoke-grid BRT to give a sane spread over a 100-step horizon; the
# right values depend on the cost scale the converged value function produces, so they
# want retuning against the full solve.  Left as they were, so switching to the full
# BRT changed only the BRT.  ESS = 1/sum(omega^2) is logged for exactly this purpose:
# ESS -> 1 means LAMBDA is too small (winner-take-all), ESS -> K too large.
V_TARGET = 2.0  # m/s, inside the grid's v_forward range
W_VEL, W_CENTERING, W_COLLISION = 5.0, 20.0, 500.0
W_IN_BRT = 0.0  # the DualGuard profile costs obstacles, not the BRT

# --- safety filter ----------------------------------------------------------
GUARD2_THRESHOLD = 0.1  # on the executed control
GUARD1_THRESHOLD = 0.1  # inside the rollouts
FILTER_SAMPLES = True  # apply_safety_filter_to_samples
FILTER_FINAL_CONTROL = True  # apply_safety_filter_to_final_control


# ---------------------------------------------------------------------------
# Dynamics: term-for-term port of @BuzzracerDynamicBicycle/dynamics.m, itself a port
# of DynamicBicycleCartesian.f() from immrax_dynamic_bicycle_cartesian.py.  Two
# ordering details the MATLAB header flags: in the dynamic branch the tire forces come
# from the *undisturbed* v_sideway and omega, with the disturbance added afterwards
# where it rotates into the global velocity and heading rate; in the kinematic branch
# w_v_sideway is added straight to the global ydot, unrotated.  Both branches are
# evaluated and selected with jnp.where, so v_forward is clamped away from zero in the
# dynamic branch to keep the unselected values finite.
#
# Integration is forward Euler, in plant and rollouts alike.
# ---------------------------------------------------------------------------

# Body parameters: car config audi_11, exactly as in BuzzracerDynamicBicycle.m
LF, LR = 0.04824, 0.09 - 0.04824
L = LF + LR
MASS = 0.1667
IZ = 1.0 / 12.0 * MASS * (0.15**2 + 0.1**2)
TIRE_B, TIRE_C, TIRE_D, REAR_SCALE = 2.3, 1.6, 1.1, 1.15
CM, CV, C0 = 6.17, 15.2, 0.333
V_SWITCH = 0.1
MASS_RATIO = MASS * 9.8 / L


@jit
def wrap_angle(a):
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


@jit
def f_continuous(state, u, d):
    """xdot = f(x, u, d) for the six-state dynamic bicycle."""
    x, y, psi, vf, vs, om = state
    delta, throttle = u
    d1, d2 = d

    a_forward = CM * (throttle - vf / CV - C0)  # shared by both branches
    use_dynamic = vf >= V_SWITCH
    vf_safe = jnp.where(use_dynamic, vf, 1.0)  # keep the unused branch finite

    tire = lambda slip: TIRE_D * jnp.sin(TIRE_C * jnp.arctan(TIRE_B * slip))
    Ffy = tire(delta - jnp.arctan((om * LF + vs) / vf_safe)) * MASS_RATIO * LR
    Fry = REAR_SCALE * tire(jnp.arctan((om * LR - vs) / vf_safe)) * MASS_RATIO * LF
    c, s = jnp.cos(psi), jnp.sin(psi)
    dyn = jnp.array(
        [
            vf * c - (vs + d1) * s,
            vf * s + (vs + d1) * c,
            om + d2,
            a_forward,
            (Fry + Ffy - MASS * vf * om) / MASS,
            (Ffy * LF - Fry * LR) / IZ,
        ]
    )

    beta = jnp.arctan(LR / L * jnp.tan(delta))
    om_ref = vf * jnp.cos(beta) / L * jnp.tan(delta)
    kin = jnp.array(
        [
            vf * jnp.cos(psi + beta),
            vf * jnp.sin(psi + beta) + d1,  # unrotated, as in the source
            om_ref + d2,
            a_forward,
            -vs,
            -(om - om_ref),
        ]
    )

    return jnp.where(use_dynamic, dyn, kin)


@jit
def step_euler(state, u, d, dt=DT):
    nxt = state + dt * f_continuous(state, u, d)
    return nxt.at[2].set(wrap_angle(nxt[2]))


# ---------------------------------------------------------------------------


def load_brt(path):
    """Read an ``export_brt_with_gradient.m`` file into plain numpy arrays.

    The six ``deriv_*`` gradient fields are ~460 MB each and are not loaded: ``uOpt_*``
    and ``dOpt_*`` already carry what they would be used to recompute, and being
    bang-bang they want nearest-neighbour lookup rather than interpolated gradients.
    Axis extents come from ``grid_axes``, never ``grid_max``, because ``grid_max[2]``
    is the *pre-wrap* heading bound (pi - dx) while ``grid_axes[2]`` runs to +pi.
    """
    raw = mat73.loadmat(path)
    axes = lambda cell: [np.asarray(a, dtype=np.float64).squeeze() for a in cell]
    brt = {
        "grid_axes": axes(raw["grid_axes"]),
        "lx_axes": axes(raw["lx_axes"]),
        "value": np.asarray(raw["value"], dtype=np.float32),
        "lx": np.asarray(raw["lx"], dtype=np.float32),
        "uOpt": [np.asarray(raw[f"uOpt_{i}"], dtype=np.float32) for i in (1, 2)],
        "dOpt": [np.asarray(raw[f"dOpt_{i}"], dtype=np.float32) for i in (1, 2)],
        "pdDim": int(raw["pdDim"]) - 1,  # MATLAB -> 0-based
        "uMode": str(raw["uMode"]),
        "dMode": str(raw["dMode"]),
    }

    ax, pd = brt["grid_axes"], brt["pdDim"]
    shape = tuple(a.size for a in ax)
    assert brt["value"].shape == shape, f"value {brt['value'].shape} != grid {shape}"
    assert all(a.shape == shape for a in brt["uOpt"] + brt["dOpt"])
    assert list(np.asarray(raw["lx_dims"], dtype=int).ravel()) == [1, 2], (
        "lx is expected to depend on (x, y)"
    )
    assert brt["lx"].shape == tuple(a.size for a in brt["lx_axes"])
    # the periodic axis carries a duplicate of its first slice at the end
    assert np.array_equal(
        np.take(brt["value"], 0, pd), np.take(brt["value"], -1, pd)
    ), "periodic wrap slice is missing or corrupt"
    assert abs((ax[pd][1] - ax[pd][0]) * (ax[pd].size - 1) - 2 * np.pi) < 1e-9
    # the safety filter assumes the control maximises V and the disturbance opposes it
    assert brt["uMode"] == "max" and brt["dMode"] == "min", (
        f"unexpected (uMode, dMode) = ({brt['uMode']}, {brt['dMode']})"
    )

    print(
        f"{path}\n  grid {shape} = {np.prod(shape):.3g} points, "
        f"{brt['value'].nbytes / 2**20:.0f} MiB per array"
    )
    for i, a in enumerate(ax):
        print(
            f"    {LABELS[i]:<7s} [{a[0]:7.3f}, {a[-1]:7.3f}]  N {a.size:4d}  "
            f"d {a[1] - a[0]:8.4f}{'  (periodic, wrapped)' if i == pd else ''}"
        )
    print(
        f"  value in [{brt['value'].min():.4f}, {brt['value'].max():.4f}], "
        f"dOpt {np.unique(brt['dOpt'][0])}, {np.unique(brt['dOpt'][1])}"
    )
    return brt


def build_sim(brt, track):
    """Bind the device tables into the jitted queries, rollout and MPPI update."""
    grid = NearestGrid(brt["grid_axes"], periodic_dim=brt["pdDim"])  # 6-D state grid
    lx_grid = NearestGrid(brt["lx_axes"])  # coarse (x, y)
    ref_grid = NearestGrid([track["lx_ref_x"], track["lx_ref_y"]])  # 1 cm truth
    value_f, lx_f = NearestGrid.flatten(brt["value"]), NearestGrid.flatten(brt["lx"])
    uopt_f = [NearestGrid.flatten(a) for a in brt["uOpt"]]
    dopt_f = [NearestGrid.flatten(a) for a in brt["dOpt"]]
    lx_ref_f = NearestGrid.flatten(track["lx_ref"].astype(np.float32))
    lx_max = float(track["lx_ref"].max())  # corridor half-width, 0.2478 m
    print(
        f"  tables on device: "
        f"{sum(a.nbytes for a in [value_f, lx_f, lx_ref_f, *uopt_f, *dopt_f]) / 2**30:.2f} GiB"
    )

    # Queries mirroring the callables experiment_runner.py hands to MPPI.  The unsafe
    # test is V <= threshold, matching dubins_environment.check_brt_collision.
    value_q = jit(lambda s: grid.gather(value_f, s, fill=OOB_FILL))
    oob_q = jit(lambda s: grid.locate(s)[1])
    lx_q = jit(lambda s: lx_grid.gather(lx_f, s[:2], fill=OOB_FILL))
    # The margin function: the exact 1 cm signed distance, deliberately not the coarse
    # lx the planner and the filter see.  Negative means the car is through the wall.
    margin_q = jit(lambda s: ref_grid.gather(lx_ref_f, s[:2], fill=OOB_FILL))

    def table_pair(tables):
        # uOpt/dOpt clamp instead of filling: there is no meaningful fill value for a
        # control.  A safety control read out of grid is the nearest in-grid cell's and
        # is not certified; the logged `oob` flag is what marks those steps.
        @jit
        def query(state):
            flat, _ = grid.locate(state)
            return jnp.array([tables[0][flat], tables[1][flat]])

        return query

    ctrl_q, dstb_q = table_pair(uopt_f), table_pair(dopt_f)

    @jit
    def rollout(state0, u_seq):
        """One hallucinated trajectory with the sample-level filter (guard 1).

        ``MPPI._compute_rollout_costs`` filters *before* stepping, on the value at the
        pre-step state, and returns the applied actions so the update can be taken with
        respect to them.  The running cost follows the ``obs`` profile and is evaluated
        on post-step states with the applied actions, as the repo does.  Rollouts use
        the nominal (zero-disturbance) model, as ``controller_mppi.py`` does -- catching
        the resulting prediction error is guard 2's job.
        """

        def step(state, u):
            unsafe = value_q(state) <= GUARD1_THRESHOLD
            applied = jnp.clip(
                jnp.where(unsafe & FILTER_SAMPLES, ctrl_q(state), u), U_MIN, U_MAX
            )
            nxt = step_euler(state, applied, jnp.zeros(2))
            return nxt, (nxt, applied, unsafe)

        _, (states, applied, unsafe) = jax.lax.scan(step, state0, u_seq)
        lx, v = jax.vmap(lx_q)(states), jax.vmap(value_q)(states)
        cost = (
            W_VEL * jnp.sum((states[:, 3] - V_TARGET) ** 2)
            + W_CENTERING * jnp.sum(lx_max - lx)
            + W_COLLISION * jnp.sum(lx <= 0.0)
            + W_IN_BRT * jnp.sum(v <= 0.0)
        )
        return states, applied, unsafe, cost

    rollout_batch = jax.vmap(rollout, in_axes=(None, 0))

    @jit
    def mppi_update(U, key, state):
        """``MPPI._compute_total_cost_batch`` + ``MPPI._command``, line for line."""
        noise = random.normal(key, (NUM_THREADS, HORIZON, 2)) * NOISE_SIGMA + NOISE_MU
        states, applied, unsafe, rollout_cost = rollout_batch(
            state, jnp.clip(U[None] + noise, U_MIN, U_MAX)
        )

        # Noise measured against what was actually applied, so the safety overrides
        # fold into the update (controller_mppi.py: noise = sampled_actions - U).
        eff_noise = applied - U[None]
        action_cost = (
            LAMBDA
            * SIGMA_INV_DIAG
            * (jnp.abs(eff_noise) if NOISE_ABS_COST else eff_noise)
        )
        cost = rollout_cost + jnp.sum(U[None] * action_cost, axis=(1, 2))

        w = jnp.exp(-(1.0 / LAMBDA) * (cost - jnp.min(cost)))
        omega = w / jnp.sum(w)
        U_new = jnp.clip(
            U + jnp.sum(omega[:, None, None] * eff_noise, axis=0), U_MIN, U_MAX
        )
        return U_new, states, unsafe, cost, 1.0 / jnp.sum(omega**2)

    return dict(
        value=value_q,
        oob=oob_q,
        lx=lx_q,
        margin=margin_q,
        ctrl=ctrl_q,
        dstb=dstb_q,
        update=mppi_update,
        lx_max=lx_max,
    )


def initial_plan(key):
    """Initial nominal plan: a noise sample, not the mean."""
    if U_INIT_IS_MEAN:
        return jnp.broadcast_to(NOISE_MU, (HORIZON, 2)).astype(jnp.float32)
    return jnp.clip(
        random.normal(key, (HORIZON, 2)) * NOISE_SIGMA + NOISE_MU, U_MIN, U_MAX
    )
