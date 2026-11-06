"""Track construction and normalization for the controller sweep fixtures.

Used only by ``experiments/controller_sweep/build_track_fixtures.py``.  Each factory track is
built with whatever native preparation it has, then the fields required by the
``ImmraxController`` / Stanley / Laptimer / BoundaryChecker stack are filled in
and returned as the canonical fixture dict.

Importing this module assumes ``experiments/shared/_env.py`` has already put the
Buzzracer package dir and the project root on ``sys.path``.
"""

import os
import sys

import buzzracer.common as _common
import numpy as np
from scipy.interpolate import splev

# NascarTrack does ``from common import *``; alias it to the package module so
# its ConfigObject is the same class as Track's.
sys.modules.setdefault("common", _common)

from buzzracer.scripts.qp_smooth import QpSmooth
from buzzracer.tracks.orca_track import OrcaTrack
from buzzracer.tracks.track_factory import TrackFactory
from raceline_track import periodic_interp, signed_curvature

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)

SUPPORTED_TRACKS = ("saved", "small", "easy", "big", "skidpad", "nascar", "orca")
_RCP_QP_TRACKS = ("small", "easy", "big")

# Quasi-steady-state lateral acceleration limit used to derive a speed profile
# for tracks that do not ship one.
A_LAT_MAX = 1.2 * 9.81
V_MIN = 0.5
V_MAX = 4.0


def _curvature_limited_speed(curvature):
    kabs = np.maximum(np.abs(np.asarray(curvature, dtype=float)), 1e-6)
    return np.clip(np.sqrt(A_LAT_MAX / kabs), V_MIN, V_MAX)


def _discretize_from_spline(track, left, right):
    """Fill the standard arrays from ``track.raceline_s`` and given boundaries."""
    n = track.discretized_raceline_len
    ss = np.linspace(0.0, track.raceline_len_m, n)
    rr = np.asarray(splev(ss % track.raceline_len_m, track.raceline_s, der=0))
    dr = np.asarray(splev(ss % track.raceline_len_m, track.raceline_s, der=1))
    headings = np.arctan2(dr[1], dr[0])

    track.curvature_s = signed_curvature(track.raceline_s, track.raceline_len_m)
    velocity = _curvature_limited_speed(track.curvature_s(ss))
    track.sToV = periodic_interp(ss, velocity, track.raceline_len_m)

    track.ss = ss
    track.raceline_points = rr
    track.raceline_headings = headings
    track.raceline_velocity = velocity
    track.raceline_left_boundary = np.asarray(left, dtype=float)
    track.raceline_right_boundary = np.asarray(right, dtype=float)
    track.discretized_raceline = np.column_stack(
        [
            rr.T,
            headings,
            velocity,
            track.raceline_left_boundary,
            track.raceline_right_boundary,
        ]
    )


def _finalize_constant_width(track):
    n = track.discretized_raceline_len
    half_width = np.full(n, track.width / 2.0)
    _discretize_from_spline(track, half_width, half_width)


def _finalize_orca(track):
    n = track.discretized_raceline_len
    ss = np.linspace(0.0, track.raceline_len_m, n)
    raw_s = np.asarray(track.s, dtype=float)

    def lookup(values):
        values = np.asarray(values, dtype=float)
        if abs(raw_s[-1] - track.raceline_len_m) > 1e-9:
            s_ext = np.append(raw_s, track.raceline_len_m)
            v_ext = np.append(values, values[0])
        else:
            s_ext = raw_s
            v_ext = values
        return np.interp(np.mod(ss, track.raceline_len_m), s_ext, v_ext)

    _discretize_from_spline(track, lookup(track.d_upper), lookup(-track.d_lower))
    track.start_pos = tuple(
        float(v) for v in np.asarray(splev(0.0, track.raceline_s, der=0))
    )
    track.start_dir = float(track.raceline_headings[0])


class _OrcaMain:
    basedir = os.path.join(PROJECT_ROOT, "vendor", "Buzzracer", "buzzracer", "RL")


class OrcaTrackAdapter(OrcaTrack):
    """``OrcaTrack`` with the data path corrected for this checkout."""

    def load_track(self):
        path = os.path.join(
            self.main.basedir, "copg", "car_racing_simulator", "ORCA_OPT"
        )
        self.path = path + os.sep
        xy = np.loadtxt(os.path.join(path, "x_center.txt"))
        self.X = xy[:, 0]
        self.Y = xy[:, 1]
        self.N = len(self.X)
        self.s = np.loadtxt(os.path.join(path, "s_center.txt"))
        self.phi = np.loadtxt(os.path.join(path, "phi_center.txt"))
        self.kappa = np.loadtxt(os.path.join(path, "kappa_center.txt"))
        self.diff_s = float(np.mean(np.diff(self.s)))
        self.d_upper = np.loadtxt(os.path.join(path, "con_inner.txt"))
        self.d_lower = np.loadtxt(os.path.join(path, "con_outer.txt"))
        self.d_upper[520:565] = np.clip(self.d_upper[520:565], 0.2, 1)
        self.border_angle_upper = np.loadtxt(os.path.join(path, "con_angle_inner.txt"))
        self.border_angle_lower = np.loadtxt(os.path.join(path, "con_angle_outer.txt"))


def build_track(name):
    """Build and finalize one supported track."""
    if name == "saved":
        return TrackFactory.build("saved")
    if name in _RCP_QP_TRACKS:
        track = QpSmooth()
        track = TrackFactory.build(name, track=track)
        track.optimize_path(offset=0.1, visualize_final_result=False)
        track.reconstruct_raceline()
        track.prepare_discretized_raceline()
        return track
    if name == "skidpad":
        track = TrackFactory.build("skidpad")
        _finalize_constant_width(track)
        return track
    if name == "nascar":
        track = TrackFactory.build("nascar")
        _finalize_constant_width(track)
        return track
    if name == "orca":
        track = OrcaTrackAdapter(_OrcaMain(), None)
        _finalize_orca(track)
        return track
    raise ValueError(
        f"unsupported track {name!r}; choose from {list(SUPPORTED_TRACKS)} "
        "(empty has no raceline)"
    )


def extract_arrays(track):
    """Return the canonical fixture dict for a finalized track."""
    ss = np.asarray(track.ss, dtype=float)
    points = np.asarray(track.raceline_points, dtype=float)
    if points.shape[0] != 2:
        points = points.T
    headings = np.asarray(track.raceline_headings, dtype=float)
    velocity = np.asarray(track.raceline_velocity, dtype=float)
    left = np.asarray(track.raceline_left_boundary, dtype=float)
    right = np.asarray(track.raceline_right_boundary, dtype=float)
    length = float(track.raceline_len_m)

    normal = np.vstack([-np.sin(headings), np.cos(headings)])
    left_pts = points + normal * left
    right_pts = points - normal * right
    all_pts = np.hstack([points, left_pts, right_pts])
    x_min = float(all_pts[0].min() - 0.1)
    x_max = float(all_pts[0].max() + 0.1)
    y_min = float(all_pts[1].min() - 0.1)
    y_max = float(all_pts[1].max() + 0.1)

    start_pos = np.asarray(getattr(track, "start_pos", points[:, 0]), dtype=float)
    start_dir = float(getattr(track, "start_dir", headings[0]))

    return {
        "ss": ss,
        "raceline": points.T,
        "headings": headings,
        "velocity": velocity,
        "left": left,
        "right": right,
        "raceline_len_m": np.asarray(length),
        "start_pos": start_pos,
        "start_dir": np.asarray(start_dir),
        "x_min": np.asarray(x_min),
        "x_max": np.asarray(x_max),
        "y_min": np.asarray(y_min),
        "y_max": np.asarray(y_max),
    }


def validate_fixture(data, name):
    """Assert the fixture supplies every field the controller stack needs."""
    n = int(np.asarray(data["ss"]).size)
    labels = {
        "ss": (n,),
        "raceline": (n, 2),
        "headings": (n,),
        "velocity": (n,),
        "left": (n,),
        "right": (n,),
    }
    for key, shape in labels.items():
        arr = np.asarray(data[key])
        if arr.shape != shape:
            raise ValueError(f"{name}: {key} has shape {arr.shape}, expected {shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name}: {key} contains non-finite values")

    length = float(np.asarray(data["raceline_len_m"]))
    if not np.isfinite(length) or length <= 0:
        raise ValueError(f"{name}: invalid raceline_len_m {length}")
    ss = np.asarray(data["ss"])
    if not np.isclose(ss[0], 0.0) or not np.all(np.diff(ss) > 0):
        raise ValueError(f"{name}: ss must start at 0 and be strictly increasing")
    if np.any(np.asarray(data["velocity"]) <= 0):
        raise ValueError(f"{name}: velocity must be positive everywhere")
    if np.any(np.asarray(data["left"]) <= 0) or np.any(np.asarray(data["right"]) <= 0):
        raise ValueError(f"{name}: boundary distances must be positive")
    return {
        "n": n,
        "length": length,
        "v_min": float(np.min(data["velocity"])),
        "v_max": float(np.max(data["velocity"])),
        "left_min": float(np.min(data["left"])),
        "right_min": float(np.min(data["right"])),
    }


def build_fixture(name):
    """Build, finalize and validate one fixture; returns ``(dict, summary)``."""
    data = extract_arrays(build_track(name))
    summary = validate_fixture(data, name)
    return data, summary
