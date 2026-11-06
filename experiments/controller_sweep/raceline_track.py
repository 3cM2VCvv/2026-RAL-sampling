"""Generic Buzzracer ``Track`` backed by a normalized raceline fixture.

Fixtures are written by ``experiments/controller_sweep/build_track_fixtures.py`` to
``outputs/controller_sweep/track_fixtures/<name>.npz`` with the keys produced by
``experiments/_track_setup.extract_arrays``.  This class turns those arrays
back into the standard ``Track`` API used by ``ImmraxController``, the Stanley
warm start, ``Laptimer``, ``BoundaryChecker`` and the simulator's adversarial
noise.
"""

import os
from math import atan2, cos, sin

import cv2
import numpy as np
from buzzracer.tracks.track import Track
from scipy.interpolate import interp1d, splev, splprep
from scipy.spatial import cKDTree

FIXTURE_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        os.pardir,
        "outputs",
        "controller_sweep",
        "track_fixtures",
    )
)


def fit_closed_spline(ss, points, length):
    """Periodic ``splprep`` tck over ``(ss, points)`` for a closed raceline."""
    ss = np.asarray(ss, dtype=float)
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] != 2:
        pts = pts.T
    if not np.isclose(ss[0], 0.0):
        raise ValueError(f"raceline ss must start at 0, got {ss[0]}")

    # splprep needs strictly increasing u.  Drop a stored endpoint equal to
    # `length` and close the loop explicitly with the first point.
    keep = ss < length - 1e-9
    ss_fit = np.append(ss[keep], length)
    pts_fit = np.vstack([pts[:, keep].T, pts[:, :1].T])
    tck, _ = splprep([pts_fit[:, 0], pts_fit[:, 1]], u=ss_fit, s=0, per=1)
    return tck


def periodic_interp(ss, values, length):
    """Periodic linear interpolation over arc length; always returns an array."""
    ss = np.asarray(ss, dtype=float)
    values = np.asarray(values, dtype=float)
    if abs(ss[-1] - length) > 1e-9:
        ss_ext = np.append(ss, length)
        vv_ext = np.append(values, values[0])
    else:
        ss_ext = ss
        vv_ext = values
    f = interp1d(ss_ext, vv_ext, kind="linear", fill_value="extrapolate")

    def s_to_v(s):
        return f(np.mod(np.asarray(s, dtype=float), length))

    return s_to_v


def signed_curvature(tck, length):
    """Signed curvature (positive = left turn) from an arc-length spline."""

    def curvature_s(s):
        scalar = np.ndim(s) == 0
        s_arr = np.atleast_1d(np.mod(np.asarray(s, dtype=float), length))
        dr = np.asarray(splev(s_arr, tck, der=1))
        ddr = np.asarray(splev(s_arr, tck, der=2))
        num = dr[0] * ddr[1] - dr[1] * ddr[0]
        den = np.linalg.norm(dr, axis=0) ** 3
        out = num / (den + 1e-12)
        return float(out[0]) if scalar else out

    return curvature_s


class RacelineTrack(Track):
    """A ``Track`` reconstructed from one normalized raceline fixture."""

    def __init__(self, main=None, config=None, fixture_path=None):
        Track.__init__(self, main, config)
        self.resolution = 200
        if fixture_path is None:
            raise ValueError("fixture_path is required")
        self._load(np.load(fixture_path))
        self.discretized_raceline_len = len(self.ss)

    def _load(self, d):
        self.ss = np.asarray(d["ss"], dtype=float)
        self.raceline_points = np.asarray(d["raceline"], dtype=float).T
        self.raceline_headings = np.asarray(d["headings"], dtype=float)
        self.raceline_velocity = np.asarray(d["velocity"], dtype=float)
        self.raceline_left_boundary = np.asarray(d["left"], dtype=float)
        self.raceline_right_boundary = np.asarray(d["right"], dtype=float)
        self.raceline_len_m = float(d["raceline_len_m"])
        self.start_pos = tuple(np.asarray(d["start_pos"], dtype=float))
        self.start_dir = float(d["start_dir"])
        self._x_min = float(d["x_min"]) if "x_min" in d else 0.0
        self._y_min = float(d["y_min"]) if "y_min" in d else 0.0
        self.x_limit = (
            float(d["x_max"] - d["x_min"])
            if "x_max" in d
            else float(np.ptp(self.raceline_points[0]) + 0.2)
        )
        self.y_limit = (
            float(d["y_max"] - d["y_min"])
            if "y_max" in d
            else float(np.ptp(self.raceline_points[1]) + 0.2)
        )

        self.raceline_s = fit_closed_spline(
            self.ss, self.raceline_points, self.raceline_len_m
        )
        self.sToV = periodic_interp(
            self.ss, self.raceline_velocity, self.raceline_len_m
        )
        self.curvature_s = signed_curvature(self.raceline_s, self.raceline_len_m)

        self.discretized_raceline = np.column_stack(
            [
                self.raceline_points.T,
                self.raceline_headings,
                self.raceline_velocity,
                self.raceline_left_boundary,
                self.raceline_right_boundary,
            ]
        )

        self._tree = cKDTree(self.raceline_points.T)
        self._rl_ss = self.ss % self.raceline_len_m

    # ----------------------------------------------------------- geometry API
    def local_trajectory(self, state, wheelbase=0.0):
        x = float(state[0]) + wheelbase * cos(float(state[2]))
        y = float(state[1]) + wheelbase * sin(float(state[2]))
        _, i = self._tree.query([x, y])
        s = float(self._rl_ss[i]) % self.raceline_len_m

        r = np.asarray(splev(s, self.raceline_s, der=0))
        dr = np.asarray(splev(s, self.raceline_s, der=1))
        tang = dr / (np.linalg.norm(dr) + 1e-12)
        orientation = atan2(dr[1], dr[0])
        offset = float(np.cross(tang, np.array([x, y]) - r))
        curvature = float(self.curvature_s(s))
        v_target = float(np.asarray(self.sToV(np.array([s])))[0])
        return (r, offset, orientation, curvature, v_target)

    def precise_track_boundary(self, coord, heading):
        _, i = self._tree.query(coord)
        raceline_heading = self.raceline_headings[i]
        tang = np.array([cos(raceline_heading), sin(raceline_heading)])
        offset = float(
            np.cross(tang, np.asarray(coord, dtype=float) - self.raceline_points[:, i])
        )
        left = float(self.raceline_left_boundary[i]) - offset
        right = float(self.raceline_right_boundary[i]) + offset
        return (left, right)

    def is_outside(self, coord):
        left, right = self.precise_track_boundary(coord, 0.0)
        return left < 0 or right < 0

    # --------------------------------------------------------------- drawing
    def _boundary_points(self):
        normal = np.vstack(
            [-np.sin(self.raceline_headings), np.cos(self.raceline_headings)]
        )
        left = self.raceline_points + normal * self.raceline_left_boundary
        right = self.raceline_points - normal * self.raceline_right_boundary
        return left.T, right.T

    def draw_track(self, img=None, show=False):
        if img is None:
            w = max(int(self.x_limit * self.resolution), 1)
            h = max(int(self.y_limit * self.resolution), 1)
            img = np.full((h, w, 3), 255, dtype=np.uint8)
        left, right = self._boundary_points()
        img = self.draw_polyline(left, img, lineColor=(255, 0, 0), thickness=2)
        img = self.draw_polyline(right, img, lineColor=(255, 0, 0), thickness=2)
        return img

    def draw_raceline(self, img=None, points=None):
        if img is None:
            img = self.draw_track()
        img = self.draw_polyline(
            self.raceline_points.T, img, lineColor=(0, 180, 0), thickness=2
        )
        if points is not None:
            for p in points:
                img = cv2.circle(img, self.m2canvas(p), 5, (0, 0, 255), -1)
        return img

    def m2canvas(self, coord):
        x_new = int(
            (np.clip(coord[0] - self._x_min, 0, self.x_limit)) * self.resolution
        )
        y_new = int(
            (self.y_limit - np.clip(coord[1] - self._y_min, 0, self.y_limit))
            * self.resolution
        )
        return (x_new, y_new)
