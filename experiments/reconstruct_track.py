"""Reconstruct the DualGuard "RC warehouse" racetrack from an HJB artifact.

`brt_rc_wh_fine_v3.mat` (HelperOC / Level-Set-Toolbox, MATLAB v7.3) stores a
Dubins-car value function on an [x, y, theta] grid.  Crucially it also stores
`lx`, the *position-only* target function l(x): the raw collision geometry,
independent of the reachability solve.  Its `lx = 0` level set is the exact
track boundary.

Sign convention (standard HJB avoid problem, matching DualGuard's
`dubins_environment.py` where collision is ``obstacle_value <= 0``):
``lx > 0`` is drivable/safe, ``lx < 0`` is inside a wall / the infield, and the
``lx = 0`` level set is the exact track boundary.

This script reads that geometry and produces everything `WarehouseTrack` needs:

  * outer + inner wall polygons (the two `lx = 0` contours),
  * a minimum-curvature racing line inside the corridor,
  * per-point left/right boundary distances (ray-cast to `lx = 0`),
  * a traction-limited speed profile,
  * track dimensions and a start pose,

and saves them to ``data/warehouse_track.npz``.

Two independent margins, which must not be conflated:

  ``--buffer``           shrinks the *drivable corridor* itself.  Because `lx` is a
                         signed distance field, subtracting a constant from it moves
                         the `lx = 0` wall inward by exactly that many meters.  It is
                         applied once, at the source, so the boundary ray-casts, the
                         collision check and the eq.-16 `l(x)` cost all see the same
                         shrunken track.  Negative values widen it.
  ``--raceline-offset``  keeps the *racing line* off the wall inside the QP.  It does
                         not change the corridor anyone else sees.

Run via DVC (``dvc repro reconstruct_track``) or directly:
    uv run python experiments/reconstruct_track.py --buffer 0.05
"""

import argparse
import os

import _env  # noqa: F401  sets up sys.path for buzzracer + root helpers

import h5py
import matplotlib

matplotlib.use("Agg")  # headless: qp_smooth imports pyplot
import numpy as np
from scipy.interpolate import RegularGridInterpolator, splev, splprep
from scipy.ndimage import convolve, label
from skimage.measure import find_contours
from skimage.morphology import medial_axis

from buzzracer.scripts.qp_smooth import QpSmooth

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_MAT = os.path.join(_env.PROJECT_ROOT, "brt_rc_wh_fine_v3.mat")
DEFAULT_OUT = os.path.join(_env.PROJECT_ROOT, "data", "warehouse_track.npz")

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------
N_CTRL = 150  # centerline control points seeding the QP smoother
N_DENSE = 2000  # samples used to store the final smoothed racing line


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mat", default=DEFAULT_MAT, help="HJB artifact to read")
    p.add_argument("--out", default=DEFAULT_OUT, help="npz to write")
    p.add_argument("--buffer", type=float, default=0.0,
                   help="[m] shrink the drivable corridor (lx -= buffer); "
                        "negative widens it")
    p.add_argument("--raceline-offset", dest="raceline_offset", type=float, default=0.08,
                   help="[m] keep the racing line this far from either wall (QP only)")
    return p.parse_args(argv)


def load_geometry(mat_path):
    """Return (lx, dx, x_shift, y_shift, x_limit, y_limit).

    `lx` is indexed [y_idx, x_idx].  World coords are shifted into the first
    quadrant: x = x_idx * dx, y = y_idx * dx (so the min corner is the origin).
    """
    with h5py.File(mat_path, "r") as f:
        lx = np.array(f["lx"])  # (ny, nx) = (408, 608)
        gmin = np.array(f["g"]["min"]).ravel()  # [-0.04, -0.04, -pi]
        dx = float(np.array(f["g"]["dx"]).ravel()[0])
    ny, nx = lx.shape
    x_shift, y_shift = -gmin[0], -gmin[1]  # shift so min -> 0
    x_limit = (nx - 1) * dx
    y_limit = (ny - 1) * dx
    return lx, dx, x_shift, y_shift, x_limit, y_limit


def idx_to_world(rc, dx):
    """(row, col) float index array (M,2) -> (x, y) meters (M,2), origin-shifted."""
    rc = np.asarray(rc)
    x = rc[:, 1] * dx
    y = rc[:, 0] * dx
    return np.column_stack([x, y])


def extract_walls(lx, dx):
    """Return (outer_xy, inner_xy) wall polygons in meters, largest first."""
    contours = sorted(find_contours(lx, 0.0), key=len, reverse=True)[:2]
    polys = [idx_to_world(c, dx) for c in contours]
    # order by bounding-box area: outer wall is the larger silhouette
    polys.sort(key=lambda p: np.ptp(p[:, 0]) * np.ptp(p[:, 1]), reverse=True)
    return polys[0], polys[1]


def _skeleton_neighbor_count(sk):
    return (
        convolve(sk.astype(int), np.ones((3, 3)), mode="constant") - sk.astype(int)
    ) * sk


def _walk_skeleton_loop(sk):
    """Order a pruned (endpoint-free) skeleton into a single closed pixel loop.

    Greedy directional walk: at each step continue to the unvisited neighbor
    whose direction best matches the current heading.  This follows the main
    loop and treats residual branch stubs as negligible detours.
    """
    pts = list(map(tuple, np.argwhere(sk)))
    ptset = set(pts)
    start = pts[0]
    visited = {start}
    order = [start]
    cur, pdir = start, np.zeros(2)
    while True:
        r, c = cur
        cands = [
            (r + dr, c + dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)
            and (r + dr, c + dc) in ptset
            and (r + dr, c + dc) not in visited
        ]
        if not cands:
            break

        def score(q):
            d = np.array([q[0] - r, q[1] - c], float)
            d /= np.linalg.norm(d)
            return float(d @ pdir)

        nxt = max(cands, key=score)
        d = np.array([nxt[0] - r, nxt[1] - c], float)
        pdir = d / np.linalg.norm(d)
        visited.add(nxt)
        order.append(nxt)
        cur = nxt
    return np.array(order, dtype=float)


def centerline_medial_axis(lx, dx, n):
    """Centerline of the drivable corridor via its medial axis (skeleton).

    The medial axis is equidistant from both walls, so it is contained in the
    drivable region (lx>0) by construction and rounds the infield smoothly --
    unlike an inner/outer midpoint, which can leave the corridor at concave
    features.  Returns `n` ordered, evenly-spaced points (meters).
    """
    drive = lx > 0.0
    lbl, k = label(drive)
    sizes = [(lbl == i).sum() for i in range(1, k + 1)]
    corridor = lbl == (1 + int(np.argmax(sizes)))
    sk = medial_axis(corridor)
    # strip spur endpoints until only closed loops remain
    while True:
        endpoints = (_skeleton_neighbor_count(sk) == 1) & sk
        if not endpoints.any():
            break
        sk[endpoints] = False
    order = _walk_skeleton_loop(sk)  # (M, 2) as (row, col)
    xy = idx_to_world(order, dx)
    return resample_closed(xy, n)


def resample_closed(xy, n):
    """Fit a periodic spline through xy and return n evenly-spaced points."""
    xy = np.asarray(xy)
    if not np.allclose(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[0]])
    tck, _ = splprep([xy[:, 0], xy[:, 1]], s=0, per=1)
    u = np.linspace(0.0, 1.0, n, endpoint=False)
    x, y = splev(u, tck)
    return np.column_stack([x, y])


def normals(pts):
    """Unit left-normals (heading + 90 deg) for a closed polyline of points."""
    tang = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    return np.column_stack([-tang[:, 1], tang[:, 0]])  # rotate +90 deg (CCW)


def raycast_halfwidths(pts, nrm, lx_interp, dx, max_dist=1.5):
    """For each point, distance to the lx=0 wall along +normal (left) and -normal (right)."""
    ts = np.arange(0.0, max_dist, dx)
    left = np.full(len(pts), max_dist)
    right = np.full(len(pts), max_dist)
    for i, (p, nv) in enumerate(zip(pts, nrm)):
        for sign, store in ((+1.0, "L"), (-1.0, "R")):
            samples = p[None, :] + sign * ts[:, None] * nv[None, :]
            vals = lx_interp(samples[:, ::-1])  # interp expects (y, x)
            # first index where we cross from drivable (lx>0) into wall (lx<=0)
            crossing = np.where(vals <= 0.0)[0]
            if len(crossing):
                j = crossing[0]
                if j == 0:
                    d = 0.0
                else:
                    v0, v1 = vals[j - 1], vals[j]
                    frac = v0 / (v0 - v1)  # linear interp to zero (v0>0, v1<=0)
                    d = ts[j - 1] + frac * dx
                if sign > 0:
                    left[i] = d
                else:
                    right[i] = d
    return left, right


class LxQpSmooth(QpSmooth):
    """Buzzracer's `QpSmooth` racing-line optimizer, driven by the `lx` corridor.

    `QpSmooth` normally minimizes curvature over an RCP grid track, querying the
    grid for boundary distances.  Here we seed it with the lx-derived centerline
    and override the boundary query (and the RCP-grid drawing hooks) so the same
    QP smoother + traction-limited speed profile run on arbitrary geometry.
    """

    def __init__(self, ctrl_pts, lx_interp, dx, x_limit, y_limit):
        QpSmooth.__init__(self)
        self._lx = lx_interp
        self._dx = dx
        self.ctrl_pts = np.asarray(ctrl_pts, dtype=float)
        self.scale = 1.0  # coords are already in meters
        self.resolution = 200
        self.x_limit = float(x_limit)
        self.y_limit = float(y_limit)
        # parameter scale for the output spline / speed profile (arbitrary > 0)
        self.track_length_grid = 3 * len(self.ctrl_pts)
        self.save_dir = "."

    def _raycast(self, coord, direction, max_dist=1.5):
        """Distance from coord to the lx=0 wall along a unit `direction`."""
        ts = np.arange(0.0, max_dist, self._dx)
        samples = np.asarray(coord)[None, :] + ts[:, None] * direction[None, :]
        vals = self._lx(samples[:, ::-1])  # interp expects (y, x)
        crossing = np.where(vals <= 0.0)[0]
        if not len(crossing):
            return max_dist
        j = crossing[0]
        if j == 0:
            return 0.0
        v0, v1 = vals[j - 1], vals[j]
        return ts[j - 1] + (v0 / (v0 - v1)) * self._dx

    # override: boundary distance from the lx level set instead of the RCP grid
    def check_track_boundary_along_dir(self, coord, n, delta_max=np.inf, offset=0.0):
        n = np.asarray(n, dtype=float)
        n = n / (np.linalg.norm(n) + 1e-12)
        F = max(self._raycast(coord, n) - offset, 0.0)  # +n is left
        R = max(self._raycast(coord, -n) - offset, 0.0)  # -n is right
        return min(F, delta_max), min(R, delta_max)

    # RCP-grid drawing hooks are unused during optimization: make them cheap
    def draw_track(self, *args, **kwargs):
        h = max(int(self.resolution * self.y_limit), 1)
        w = max(int(self.resolution * self.x_limit), 1)
        return 255 * np.ones((h, w, 3), dtype=np.uint8)

    def draw_raceline(self, *args, img=None, **kwargs):
        return img if img is not None else self.draw_track()

    def verify_speed_profile(self, *args, **kwargs):
        return


def optimize_raceline(centerline, lx_interp, dx, x_limit, y_limit, raceline_offset):
    """Run qp_smooth on the lx corridor; return (raceline_xy, ss, vv, length)."""
    smoother = LxQpSmooth(centerline, lx_interp, dx, x_limit, y_limit)
    smoother.optimize_path(
        offset=raceline_offset,
        max_iter=40,
        visualize=False,
        visualize_final_result=False,
        save_gif=False,
        save_steps=False,
    )

    # sample the optimized racing line + qp_smooth's traction-limited speed
    u = np.linspace(0.0, smoother.track_length_grid, N_DENSE, endpoint=False)
    raceline = np.array(splev(u, smoother.raceline)).T  # (N_DENSE, 2)
    vv = np.asarray(smoother.targetVfromU(u), dtype=float)
    seg = np.linalg.norm(np.diff(raceline, axis=0, append=raceline[:1]), axis=1)
    ss = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    return raceline, ss, vv, ss[-1] + seg[-1]


def main(argv=None):
    args = parse_args(argv)

    print(f"Loading {args.mat}")
    lx_raw, dx, x_shift, y_shift, x_limit, y_limit = load_geometry(args.mat)
    print(f"  grid dx={dx} m, track {x_limit:.2f} x {y_limit:.2f} m")

    # Apply the corridor buffer at the source.  lx is a signed distance field, so
    # subtracting a constant slides the lx=0 wall inward by that many meters -- every
    # consumer below (wall contours, medial axis, boundary ray-casts, and downstream
    # the collision check and the eq.-16 l(x) cost) then sees one consistent corridor.
    lx = lx_raw - args.buffer
    l_center = float(lx.max())
    print(f"  buffer {args.buffer:+.3f} m -> l_center {float(lx_raw.max()):.4f} "
          f"-> {l_center:.4f} m")

    ny, nx = lx.shape
    xs = np.arange(nx) * dx
    ys = np.arange(ny) * dx
    # beyond the grid is wall (lx <= 0), so ray-casts terminate at the domain
    lx_interp = RegularGridInterpolator(
        (ys, xs), lx, method="linear", bounds_error=False, fill_value=-1.0
    )

    outer_wall, inner_wall = extract_walls(lx, dx)
    print(f"  walls: outer {len(outer_wall)} pts, inner {len(inner_wall)} pts")

    centerline = centerline_medial_axis(lx, dx, N_CTRL)
    print(f"  centerline: {len(centerline)} control points (medial axis)")

    raceline, ss, vv, length = optimize_raceline(
        centerline, lx_interp, dx, x_limit, y_limit, args.raceline_offset
    )
    print(f"  racing line optimized via qp_smooth (offset {args.raceline_offset} m)")

    # per-point boundaries on the dense racing line
    nrm = normals(raceline)
    left, right = raycast_halfwidths(raceline, nrm, lx_interp, dx)
    print(
        f"  boundary half-width: median L={np.median(left):.3f} "
        f"R={np.median(right):.3f} m"
    )
    print(f"  lap length {length:.2f} m, v in [{vv.min():.2f}, {vv.max():.2f}] m/s")

    # start pose: first racing-line point, tangent heading
    start_pos = raceline[0]
    t0 = raceline[1] - raceline[0]
    start_dir = float(np.arctan2(t0[1], t0[0]))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        raceline=raceline.astype(np.float64),  # (N_DENSE, 2) closed loop, meters
        raceline_left=left.astype(np.float64),  # (N_DENSE,) dist to left wall
        raceline_right=right.astype(np.float64),  # (N_DENSE,) dist to right wall
        speed_ss=ss.astype(np.float64),  # (N_DENSE,) arc-length
        speed_vv=vv.astype(np.float64),  # (N_DENSE,) target speed
        outer_wall=outer_wall.astype(np.float64),  # (.,2) meters
        inner_wall=inner_wall.astype(np.float64),  # (.,2) meters
        lx=lx.astype(np.float32),  # (ny,nx) buffered l(x): boundaries, collision, cost
        lx_raw=lx_raw.astype(np.float32),  # (ny,nx) unbuffered l(x): rendering only
        l_center=np.float64(l_center),  # max l(x); the eq.-16 centering reference
        buffer=np.float64(args.buffer),
        dx=np.float64(dx),
        x_limit=np.float64(x_limit),
        y_limit=np.float64(y_limit),
        start_pos=np.asarray(start_pos, dtype=np.float64),
        start_dir=np.float64(start_dir),
    )
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
