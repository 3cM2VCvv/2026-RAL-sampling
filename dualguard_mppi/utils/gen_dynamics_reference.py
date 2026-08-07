"""Sample the jax dynamic bicycle model to produce a MATLAB parity fixture.

Writes matlab/buzzracer_dynamics_reference.mat, which
matlab/verify_buzzracer_port.m checks @BuzzracerDynamicBicycle/dynamics.m
against.

x64 is enabled so the fixture is not limited to jax's default float32; the
MATLAB port then agrees to ~1e-13 rather than ~1e-6.

NOT a DVC stage, and the checked-in fixture is NOT its current output. The
fixture, @BuzzracerDynamicBicycle and the BRT built from it all correspond to
immrax_dynamic_bicycle_cartesian.py as of commit 6094bd6, whose motor term ran
ungated. 9ffdb56 added a negative-velocity gate, which moves d/dt v_forward by
up to 8.17 on the kinematic branch (v_forward < 0.1, where the sampling range
goes down to -0.40). Running this script against HEAD therefore produces a
fixture the MATLAB port and the notebooks disagree with by design. Reconciling
them means re-solving the 115-hour BRT against the gated model.

Usage (only after that re-solve):
    uv run python dualguard_mppi/utils/gen_dynamics_reference.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))    # repo root, for immrax_*

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from scipy.io import savemat

from immrax_dynamic_bicycle_cartesian import DynamicBicycleCartesian

OUTPUT = HERE.parents[1] / "matlab" / "buzzracer_dynamics_reference.mat"

# CarParams defaults, which car config audi_11 inherits for every body term
# (it only overrides max_steering_left/right). Buzzracer/buzzracer/cars/car.py
LF = 0.04824
LR = 0.09 - 0.04824
MASS = 0.1667
IZ = 1 / 12 * 0.1667 * (0.15**2 + 0.1**2)

# Input bounds, from car.py and immrax_controller.py:118-119
DELTA_MAX = np.deg2rad(26.1)
THROTTLE = (-1.0, 1.0)
D_MAX = 0.1

# Sampling ranges, matching the state grid in matlab/brt_buzzracer_track.m
STATE_RANGES = [
    (-0.05, 2.45),    # x
    (-0.05, 3.65),    # y
    (-np.pi, np.pi),  # heading
    (-0.40, 3.20),    # v_forward
    (-1.50, 1.50),    # v_sideway
    (-14.0, 14.0),    # omega
]

N = 4000
V_SWITCH = 0.1


class _Params:
    lf, lr, m, Iz = LF, LR, MASS, IZ


class _FakeCar:
    params = _Params()

    def init_param(self):
        pass


def main():
    model = DynamicBicycleCartesian(_FakeCar())
    print(f"lf {model.lf}  lr {model.lr}  L {model.L}  m {model.m}  Iz {model.Iz}")

    f = jax.jit(lambda x, u, w: model.f(0.0, x, u, w))

    rng = np.random.default_rng(11)
    X = np.column_stack([rng.uniform(lo, hi, N) for lo, hi in STATE_RANGES])

    # Oversample the low-speed branch, and land exactly on the switch
    X[:400, 3] = rng.uniform(STATE_RANGES[3][0], V_SWITCH, 400)
    X[400:450, 3] = V_SWITCH

    U = np.column_stack([
        rng.uniform(-DELTA_MAX, DELTA_MAX, N),
        rng.uniform(THROTTLE[0], THROTTLE[1], N),
    ])
    W = rng.uniform(-D_MAX, D_MAX, (N, 2))

    XD = np.array([np.asarray(f(jnp.array(x), jnp.array(u), jnp.array(w)))
                   for x, u, w in zip(X, U, W)])

    print(f"{N} samples, {int((X[:, 3] < V_SWITCH).sum())} on the kinematic branch, "
          f"{int((X[:, 3] == V_SWITCH).sum())} exactly at the switch")
    assert np.isfinite(XD).all()

    savemat(OUTPUT, {
        "ref_x": X, "ref_u": U, "ref_w": W, "ref_xdot": XD,
        "ref_lf": model.lf, "ref_lr": model.lr, "ref_Iz": model.Iz, "ref_m": model.m,
        "ref_vSwitch": V_SWITCH,
    }, do_compression=True)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
