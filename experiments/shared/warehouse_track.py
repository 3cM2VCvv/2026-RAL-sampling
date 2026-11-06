"""DualGuard "RC warehouse" track reconstructed from an HJB value function.

The geometry is produced by the ``reconstruct_track`` DVC stage
(``experiments/dualguard_rc/reconstruct_track.py``, which reads ``brt_rc_wh_fine_v3.mat``) and
cached in ``outputs/dualguard_rc/warehouse_track.npz``.  This class loads that artifact and
exposes the standard base-``Track`` API -- the 6-column ``discretized_raceline``
``[x, y, heading, v_target, left, right]`` plus ``local_trajectory`` and
``precise_track_boundary`` -- so it is a drop-in for the ``ImmraxController`` /
Stanley / BoundaryChecker / Laptimer / VisualizationGL pipeline used by the
``stanley_immrax`` experiment (identical schema to ``RCPTrack``).
"""

import os
from math import atan2, cos, pi, sin

import cv2
import numpy as np
from buzzracer.tracks.track import Track
from scipy.interpolate import RegularGridInterpolator, interp1d, splev, splprep
from scipy.spatial import cKDTree

DATA_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        os.pardir,
        "outputs",
        "dualguard_rc",
        "warehouse_track.npz",
    )
)


class WarehouseTrack(Track):
    def __init__(self, main=None, config=None, data_path=DATA_PATH):
        Track.__init__(self, main, config)
        self.resolution = 200  # pixels per meter
        self.discretized_raceline_len = 1024
        self._load(np.load(data_path))

    # ------------------------------------------------------------------ load
    def _load(self, d):
        self.dx = float(d["dx"])
        self.x_limit = float(d["x_limit"])
        self.y_limit = float(d["y_limit"])
        self.start_pos = tuple(np.asarray(d["start_pos"], dtype=float))
        self.start_dir = float(d["start_dir"])
        self.outer_wall = np.asarray(d["outer_wall"], dtype=float)
        self.inner_wall = np.asarray(d["inner_wall"], dtype=float)

        # Collision geometry l(x): >0 drivable, <0 wall.  `lx` is the *buffered*
        # field (l(x) - buffer) and is what every boundary / collision / cost query
        # sees; `lx_raw` is the unbuffered geometry, kept only so the rendered track
        # still shows the true walls.  Older npz files have neither extra key.
        self._lx = np.asarray(d["lx"], dtype=float)
        self._lx_raw = (
            np.asarray(d["lx_raw"], dtype=float) if "lx_raw" in d else self._lx
        )
        self.buffer = float(d["buffer"]) if "buffer" in d else 0.0
        # l_center: the largest attainable l(x), i.e. l(x) on the track centerline.
        # It is the reference the eq.-16 centering term K_c*(l_center - l(x)) measures from.
        self.l_center = (
            float(d["l_center"]) if "l_center" in d else float(self._lx.max())
        )

        ny, nx = self._lx.shape
        self._lx_interp = RegularGridInterpolator(
            (np.arange(ny) * self.dx, np.arange(nx) * self.dx),
            self._lx,
            method="linear",
            bounds_error=False,
            fill_value=-1.0,
        )

        # arc-length-parameterized periodic racing-line spline
        rl = np.asarray(d["raceline"], dtype=float)  # (N, 2)
        ss = np.asarray(d["speed_ss"], dtype=float)  # (N,) arc length
        close_seg = float(np.linalg.norm(rl[0] - rl[-1]))
        self.raceline_len_m = float(ss[-1] + close_seg)
        s_full = np.append(ss, self.raceline_len_m)
        rl_full = np.vstack([rl, rl[0]])
        tck, _ = splprep([rl_full[:, 0], rl_full[:, 1]], u=s_full, s=0, per=1)
        self.raceline_s = tck
        self.raceline = tck  # (u==s here)

        # progress -> speed / boundary offsets (interpolated by arc length)
        sp = np.append(ss, self.raceline_len_m)
        wrap = lambda a: np.append(a, a[0])
        self.sToV = interp1d(
            sp, wrap(d["speed_vv"]), kind="linear", fill_value="extrapolate"
        )
        self._left_interp = interp1d(
            sp, wrap(d["raceline_left"]), fill_value="extrapolate"
        )
        self._right_interp = interp1d(
            sp, wrap(d["raceline_right"]), fill_value="extrapolate"
        )
        self.curvature_s = self._signed_curvature

        # nearest-point lookup for local_trajectory
        self._rl_pts = rl
        self._rl_ss = ss
        self._rl_tree = cKDTree(rl)

        # fill discretized_raceline (x, y, heading, v, left, right)
        self.prepare_discretized_raceline()

    # l(x) grid + its spacing, for the JAX eq.-16 cost (LxDubinsRacingCost).
    @property
    def lx_grid(self):
        """Buffered l(x) on the (ny, nx) grid; index as ``lx_grid[y_idx, x_idx]``."""
        return self._lx

    @property
    def lx_dx(self):
        """Grid spacing (m).  World coords map to indices as ``round(x / lx_dx)``."""
        return self.dx

    # --------------------------------------------------------- geometry API
    def _signed_curvature(self, s):
        s = np.asarray(s) % self.raceline_len_m
        dr = np.array(splev(s, self.raceline_s, der=1))
        ddr = np.array(splev(s, self.raceline_s, der=2))
        num = dr[0] * ddr[1] - dr[1] * ddr[0]
        return num / (np.linalg.norm(dr, axis=0) ** 3 + 1e-12)

    def create_boundary(self, show=False):
        """Override: boundaries come straight from the stored lx ray-casts."""
        s = self.ss % self.raceline_len_m
        self.raceline_left_boundary = np.asarray(self._left_interp(s))
        self.raceline_right_boundary = np.asarray(self._right_interp(s))

    def _raycast(self, coord, angle, max_dist=1.5):
        """Distance from coord to the lx=0 wall along `angle` (radians)."""
        d = np.array([cos(angle), sin(angle)])
        ts = np.arange(0.0, max_dist, self.dx)
        samples = np.asarray(coord, dtype=float)[None, :] + ts[:, None] * d[None, :]
        vals = self._lx_interp(samples[:, ::-1])  # interp expects (y, x)
        crossing = np.where(vals <= 0.0)[0]
        if not len(crossing):
            return max_dist
        j = crossing[0]
        if j == 0:
            return 0.0
        v0, v1 = vals[j - 1], vals[j]
        return ts[j - 1] + (v0 / (v0 - v1)) * self.dx

    def precise_track_boundary(self, coord, heading):
        """(left, right) distances from coord to the walls, normal to heading."""
        left = self._raycast(coord, heading + pi / 2)
        right = self._raycast(coord, heading - pi / 2)
        return (left, right)

    def local_trajectory(self, state, wheelbase=0.0):
        x = state[0] + wheelbase * cos(state[2])
        y = state[1] + wheelbase * sin(state[2])
        _, i = self._rl_tree.query([x, y])
        s = float(self._rl_ss[i]) % self.raceline_len_m

        r = np.array(splev(s, self.raceline_s, der=0))
        dr = np.array(splev(s, self.raceline_s, der=1))
        tang = dr / (np.linalg.norm(dr) + 1e-12)
        orientation = atan2(dr[1], dr[0])
        # positive offset => car is left of the raceline (matches RCPTrack)
        offset = float(np.cross(tang, np.array([x, y]) - r))
        curvature = float(self._signed_curvature(s))
        v_target = float(self.sToV(s))
        return (r, offset, orientation, curvature, v_target)

    def lx_at(self, x, y):
        """Buffered l(x) at a world point.  <= 0 means inside a wall."""
        return float(self._lx_interp([[y, x]])[0])

    def is_outside(self, coord):
        """True if the point is inside a wall (used by the watchdog)."""
        return self.lx_at(coord[0], coord[1]) <= 0.0

    # ------------------------------------------------------------- drawing
    def draw_track(self, img=None, show=False):
        """Buzzracer-style canvas: white drivable surface, blue (BGR) walls.

        Drawn from the *unbuffered* geometry -- the buffer shrinks the corridor the
        planner is allowed to use, but the physical walls are where they are.
        """
        res = self.resolution
        w = int(self.x_limit * res)
        h = int(self.y_limit * res)
        up = cv2.resize(self._lx_raw, (w, h), interpolation=cv2.INTER_LINEAR)
        drivable = up > 0.0
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[drivable] = (255, 255, 255)  # white drivable (BGR)
        img[~drivable] = (255, 0, 0)  # blue walls
        return np.flipud(img).copy()  # world-y up -> image-row down

    def draw_raceline(self, img=None, points=None):
        """Overlay the racing line, colored by speed (slow red -> fast green)."""
        if img is None:
            img = self.draw_track()
        ss = np.linspace(0.0, self.raceline_len_m, 1000)
        pts = np.array(splev(ss % self.raceline_len_m, self.raceline_s)).T
        vv = np.asarray(self.sToV(ss % self.raceline_len_m))
        vmin, vmax = float(vv.min()), float(vv.max())
        canvas = [self.m2canvas(p) for p in pts]
        for i in range(len(ss) - 1):
            t = (vv[i] - vmin) / (vmax - vmin + 1e-9)
            color = (0, int(t * 255), int(255 - 255 * t))  # BGR
            img = cv2.line(img, canvas[i], canvas[i + 1], color, 3)
        if points is not None:
            for p in points:
                img = cv2.circle(img, self.m2canvas(p), 5, (0, 0, 255), -1)
        return img


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    track = WarehouseTrack()
    print(
        "raceline length %.2f m, discretized %s"
        % (track.raceline_len_m, track.discretized_raceline.shape)
    )
    img = track.draw_raceline(track.draw_track())
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("WarehouseTrack")
    plt.axis("off")
    plt.show()
