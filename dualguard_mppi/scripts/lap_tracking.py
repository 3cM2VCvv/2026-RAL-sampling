"""Lap detection against a closed racing line.

The method is the one ``experiments/dualguard_rc_trial.py`` uses for the reachset
planner on these same tracks, reimplemented here so the baseline counts laps the way
the thing it is a baseline for does.  That module is not imported: it pulls in
``ImmraxPlanner``, ``dvclive`` and the reachset costs, none of which these scripts
need.  Keep the two in step.

Progress is cumulative arc length, not a finish-line crossing test: project onto the
nearest racing-line sample, difference the arc length ``s``, unwrap the jump at the
``s = 0`` seam, accumulate, and book a lap every ``track_len``.  So a lap is a lap
wherever the run started, idling across the seam books nothing, and going backwards
gives the progress back.
"""

import numpy as np


class FrenetProjector:
    """Nearest-racing-line projection for an executed trajectory.

    Says nothing about the track boundary -- whether a state is off-track is the
    margin function's verdict, and each script brings its own.
    """

    def __init__(self, xy, headings, ss, track_len):
        xy = np.asarray(xy, dtype=float)
        self.rx, self.ry = xy[:, 0], xy[:, 1]
        self.rh = np.asarray(headings, dtype=float)
        self.ss = np.asarray(ss, dtype=float)
        self.track_len = float(track_len)

    @classmethod
    def from_track(cls, track):
        """Build from a buzzracer base-``Track`` (e.g. ``WarehouseTrack``)."""
        return cls(np.asarray(track.raceline_points).T, track.raceline_headings,
                   track.ss, track.raceline_len_m)

    def project(self, x, y):
        """Return (progress_s, signed_lateral_err, closest_idx)."""
        idx = int(np.argmin((self.rx - x) ** 2 + (self.ry - y) ** 2))
        heading = self.rh[idx]
        lateral = -(x - self.rx[idx]) * np.sin(heading) + (y - self.ry[idx]) * np.cos(heading)
        return self.ss[idx], lateral, idx


class LapCounter:
    """Cumulative-arc-length lap counter.  :meth:`update` returns True when done."""

    def __init__(self, projector, start_xy, target_laps=3, start_time=0.0):
        self.proj = projector
        self.target_laps = int(target_laps)
        self._prev_s = projector.project(start_xy[0], start_xy[1])[0]
        self.cumulative_progress = 0.0
        self.laps_completed = 0
        self.lap_boundary_times = [float(start_time)]

    def update(self, x, y, t):
        s = self.proj.project(x, y)[0]
        ds = s - self._prev_s
        half = self.proj.track_len / 2
        # A control step never covers half a lap, so a jump that big is the seam.
        ds += self.proj.track_len if ds < -half else (-self.proj.track_len if ds > half else 0.0)
        self.cumulative_progress += ds
        self._prev_s = s
        while self.cumulative_progress >= (self.laps_completed + 1) * self.proj.track_len:
            self.laps_completed += 1
            self.lap_boundary_times.append(float(t))
        return self.finished

    @property
    def finished(self):
        return self.laps_completed >= self.target_laps

    @property
    def lap_times(self):
        return np.diff(np.asarray(self.lap_boundary_times))

    @property
    def lap_speeds(self):
        times = self.lap_times
        return self.proj.track_len / times if len(times) else times
