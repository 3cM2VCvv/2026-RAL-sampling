"""Convert the Buzzracer RCP track pickle into a MATLAB .mat for the BRT pipeline.

Reads Buzzracer/buzzracer/data/raceline.p -- the geometry dependency of the
reachset_trial matrix stage -- and writes matlab/buzzracer_track_data.mat.

Usage:
    uv run python dualguard_mppi/utils/raceline_to_mat.py
"""

import collections
import hashlib
import pickle
from pathlib import Path

import numpy as np
from scipy.interpolate import splev
from scipy.io import savemat

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]                      # dualguard_mppi/
PICKLE = HERE.parents[2] / "Buzzracer" / "buzzracer" / "data" / "raceline.p"
OUTPUT = REPO / "matlab" / "buzzracer_track_data.mat"

# Wall half-width as a fraction of the tile side, from RCPTrack.check_track_boundary
DEADZONE = 0.087

# Which tile corner each turn type curves around, in local (x, y) tile coordinates
APEX = {"SE": (1, 0), "SW": (0, 0), "NE": (1, 1), "NW": (0, 1)}
STRAIGHTS = ("WE", "NS")

# Padding added around the tile grid for the reference lx grid, and its spacing
LX_PAD = 0.05
LX_STEP = 0.01

# Number of equal-arc-length raceline samples, matching
# Track.prepare_discretized_raceline()
N_RACELINE = 1024


# --------------------------------------------------------------------------
# Reading the pickle
# --------------------------------------------------------------------------

GridSize = collections.namedtuple("GridSize", ["rows", "cols"])


class _StateCapture:
    """Stand-in for a scipy interpolator: keeps the pickled state, nothing else.

    The stored interp1d predates this project's scipy and raises inside
    _bsplines.__setstate__ on a naive load.
    """

    def __setstate__(self, state):
        self.state = state


class _ShimUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("buzzracer"):
            if name == "GridSize":
                return GridSize
            raise RuntimeError(f"unexpected buzzracer class in pickle: {module}.{name}")
        if module.startswith("scipy.interpolate"):
            return _StateCapture
        return super().find_class(module, name)


def load_raceline_pickle(path):
    with open(path, "rb") as f:
        return _ShimUnpickler(f).load()


def interp1d_table(stub):
    """Pull the (x, y) sample table out of a shimmed scipy interp1d."""
    state = stub.state
    # interp1d pickles as (instance __dict__, base-class __dict__)
    d = state[0] if isinstance(state, tuple) else state
    return np.asarray(d["x"], dtype=float), np.asarray(d["y"], dtype=float)


# --------------------------------------------------------------------------
# Boundary primitives
# --------------------------------------------------------------------------


def build_walls(tile_grid, scale, deadzone=DEADZONE):
    """Return (segments, arcs) describing the track boundary, in metres.

    tile_grid is indexed [col][row] as stored in the pickle, so tile (i, j)
    covers x in [i*scale, (i+1)*scale] and y in [j*scale, (j+1)*scale].

    segments : (N, 4) array of x1, y1, x2, y2
    arcs     : (M, 5) array of cx, cy, radius, theta_start, theta_end
               with theta_start < theta_end, spanning a quarter turn
    """
    segments, arcs = [], []
    for i, column in enumerate(tile_grid):
        for j, tile in enumerate(column):
            ox, oy = i * scale, j * scale
            if tile == "WE":
                for frac in (deadzone, 1 - deadzone):
                    segments.append((ox, oy + frac * scale, ox + scale, oy + frac * scale))
            elif tile == "NS":
                for frac in (deadzone, 1 - deadzone):
                    segments.append((ox + frac * scale, oy, ox + frac * scale, oy + scale))
            elif tile in APEX:
                ax, ay = APEX[tile]
                cx, cy = ox + ax * scale, oy + ay * scale
                # Quarter turn from the apex towards the tile interior
                theta0 = np.arctan2(0.5 - ay, 0.5 - ax) - np.pi / 4
                for frac in (deadzone, 1 - deadzone):
                    arcs.append((cx, cy, frac * scale, theta0, theta0 + np.pi / 2))
            else:
                raise ValueError(f"unknown tile type {tile!r} at ({i}, {j})")
    return np.array(segments, dtype=float), np.array(arcs, dtype=float)


def distance_to_walls(pts, segments, arcs):
    """Unsigned Euclidean distance from each point to the nearest wall."""
    a = segments[:, :2]
    ab = segments[:, 2:] - a
    ab_len_sq = (ab**2).sum(axis=1)
    t = np.clip(((pts[:, None, :] - a) * ab).sum(axis=-1) / ab_len_sq, 0.0, 1.0)
    d_seg = np.linalg.norm(pts[:, None, :] - (a + t[..., None] * ab), axis=-1).min(axis=1)

    centre = arcs[:, :2]
    radius = arcs[:, 2]
    th0, th1 = arcs[:, 3], arcs[:, 4]
    rel = pts[:, None, :] - centre
    rho = np.linalg.norm(rel, axis=-1)
    theta = np.arctan2(rel[..., 1], rel[..., 0])
    within = ((theta - th0) % (2 * np.pi)) <= (th1 - th0)

    def endpoint(th):
        offset = radius[:, None] * np.stack([np.cos(th), np.sin(th)], axis=1)
        return np.linalg.norm(pts[:, None, :] - (centre + offset), axis=-1)

    d_arc = np.where(within, np.abs(rho - radius), np.minimum(endpoint(th0), endpoint(th1)))
    return np.minimum(d_seg, d_arc.min(axis=1))


def tile_clearance(pts, tile_grid, scale, deadzone=DEADZONE):
    """Signed clearance in metres from the tile-local test, as in
    RCPTrack.check_track_boundary (which returns tile units, hence the *scale).

    Points outside the tile grid are reported as -inf: the grid covers the whole
    track, so anything beyond it is off-track by definition.
    """
    cols, rows = len(tile_grid), len(tile_grid[0])
    out = np.full(len(pts), -np.inf)

    ii = np.floor(pts[:, 0] / scale).astype(int)
    jj = np.floor(pts[:, 1] / scale).astype(int)
    inside_grid = (ii >= 0) & (ii < cols) & (jj >= 0) & (jj < rows)

    for i in range(cols):
        for j in range(rows):
            sel = inside_grid & (ii == i) & (jj == j)
            if not sel.any():
                continue
            xl = pts[sel, 0] / scale - i
            yl = pts[sel, 1] / scale - j
            tile = tile_grid[i][j]
            if tile == "WE":
                left, right = yl - deadzone, 1 - deadzone - yl
            elif tile == "NS":
                left, right = xl - deadzone, 1 - deadzone - xl
            else:
                ax, ay = APEX[tile]
                r = np.hypot(xl - ax, yl - ay)
                left, right = 1 - deadzone - r, r - deadzone
            out[sel] = np.minimum(left, right) * scale
    return out


def track_sdf(pts, segments, arcs, tile_grid, scale, deadzone=DEADZONE):
    """Signed distance to the track boundary; positive on the drivable surface."""
    d = distance_to_walls(pts, segments, arcs)
    return np.where(tile_clearance(pts, tile_grid, scale, deadzone) > 0, d, -d)


# --------------------------------------------------------------------------
# Raceline resampling
# --------------------------------------------------------------------------


def resample_raceline(tck, track_length_grid, vel_u, vel_v, n=N_RACELINE):
    """Resample the raceline spline at n equally spaced arc lengths.

    Mirrors RCPTrack.reconstruct_raceline() + Track.prepare_discretized_raceline():
    build the u -> arc-length map on a fine sampling, invert it, and evaluate
    position, heading and reference velocity at the resampled points.
    """
    u_fine = np.linspace(0.0, track_length_grid, 10001)
    xy_fine = np.array(splev(u_fine, tck)).T
    steps = np.linalg.norm(np.diff(xy_fine, axis=0), axis=1)
    s_fine = np.concatenate([[0.0], np.cumsum(steps)])
    length = float(s_fine[-1])

    s = np.linspace(0.0, length, n, endpoint=False)
    u = np.interp(s, s_fine, u_fine)

    xy = np.array(splev(u, tck)).T
    dxy = np.array(splev(u, tck, der=1)).T
    heading = np.arctan2(dxy[:, 1], dxy[:, 0])
    v = np.interp(u, vel_u, vel_v)
    return xy, s, heading, v, length


# --------------------------------------------------------------------------


def main():
    raw = PICKLE.read_bytes()
    print(f"reading  {PICKLE}  ({len(raw)} bytes, md5 {hashlib.md5(raw).hexdigest()})")

    save = load_raceline_pickle(PICKLE)
    scale = float(save["scale"])
    gridsize = save["gridsize"]
    tile_grid = save["track"]
    rows, cols = int(gridsize.rows), int(gridsize.cols)
    assert len(tile_grid) == cols and len(tile_grid[0]) == rows

    segments, arcs = build_walls(tile_grid, scale)
    print(f"tiles    {cols} x {rows} at {scale} m  ->  "
          f"{len(segments)} wall segments, {len(arcs)} wall arcs")

    # --- self-check: the primitive SDF must reproduce the tile-local test ---
    rng = np.random.default_rng(0)
    probe = np.column_stack([
        rng.uniform(-0.2, cols * scale + 0.2, 60000),
        rng.uniform(-0.2, rows * scale + 0.2, 60000),
    ])
    clear = tile_clearance(probe, tile_grid, scale)
    sdf = track_sdf(probe, segments, arcs, tile_grid, scale)
    on_track = clear > 0
    err = np.abs(sdf[on_track] - clear[on_track]).max()
    print(f"check    max |sdf - check_track_boundary| on track: {err:.3e} "
          f"({on_track.sum()} pts)")
    assert err < 1e-9, "boundary primitives disagree with the tile-local test"

    # --- reference lx on a fixed 2-D grid, for validating the MATLAB port ---
    lx_x = np.arange(-LX_PAD, cols * scale + LX_PAD + LX_STEP / 2, LX_STEP)
    lx_y = np.arange(-LX_PAD, rows * scale + LX_PAD + LX_STEP / 2, LX_STEP)
    gx, gy = np.meshgrid(lx_x, lx_y, indexing="ij")
    lx_ref = track_sdf(
        np.column_stack([gx.ravel(), gy.ravel()]), segments, arcs, tile_grid, scale
    ).reshape(gx.shape)
    print(f"lx_ref   {lx_ref.shape} at {LX_STEP} m, "
          f"range [{lx_ref.min():.4f}, {lx_ref.max():.4f}]")

    # --- raceline ---
    vel_u, vel_v = interp1d_table(save["targetVfromU"])
    xy, s, heading, v, length = resample_raceline(
        save["raceline"], int(save["track_length"]), vel_u, vel_v
    )
    clearance = track_sdf(xy, segments, arcs, tile_grid, scale)
    print(f"raceline {len(xy)} samples, {length:.4f} m, "
          f"v in [{v.min():.3f}, {v.max():.3f}] m/s, "
          f"min clearance {clearance.min():.4f} m")
    assert clearance.min() > 0, "raceline leaves the drivable surface"

    tile_cells = np.empty((cols, rows), dtype=object)
    for i in range(cols):
        for j in range(rows):
            tile_cells[i, j] = tile_grid[i][j]

    mat = {
        # --- boundary geometry: everything shapeRCPTrack2D needs ---
        "wall_segments": segments,          # (N, 4)  x1 y1 x2 y2
        "wall_arcs": arcs,                  # (M, 5)  cx cy r th_start th_end
        "tile_grid": tile_cells,            # (cols, rows) cell of char
        "tile_scale": scale,
        "tile_deadzone": DEADZONE,
        "grid_rows": rows,
        "grid_cols": cols,
        "grid_sequence": np.array(save["grid_sequence"], dtype=float),
        "track_xlim": np.array([0.0, cols * scale]),
        "track_ylim": np.array([0.0, rows * scale]),
        # --- reference failure set, for validating the MATLAB implementation ---
        "lx_ref": lx_ref,
        "lx_ref_x": lx_x,
        "lx_ref_y": lx_y,
        # --- raceline, for cost terms and plotting ---
        "raceline_xy": xy,                  # (K, 2)
        "raceline_s": s,
        "raceline_heading": heading,
        "raceline_v": v,
        "raceline_len_m": length,
        "start_pos": np.asarray(save["start_pos"], dtype=float),
        "start_dir": float(save["start_dir"]),
        "min_v": float(save["min_v"]),
        "max_v": float(save["max_v"]),
        # --- provenance ---
        "source_file": "Buzzracer/buzzracer/data/raceline.p",
        "source_md5": hashlib.md5(raw).hexdigest(),
    }
    savemat(OUTPUT, mat, do_compression=True)
    print(f"wrote    {OUTPUT}  ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
