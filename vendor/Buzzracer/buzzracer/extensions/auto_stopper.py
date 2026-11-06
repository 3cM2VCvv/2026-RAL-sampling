"""Terminate experiment after a number of laps, wall-clock timeout,
crash, or stall (no progress)."""

import time


from buzzracer.extensions.extension import Extension


class AutoStopper(Extension):
    """Terminate experiment after a number of laps or wall-clock timeout.

    Config attributes (set via XML or programmatically):
        lap_count: Number of full laps before stopping (default 1)
        wall_clock: Wall-clock timeout in seconds (default 1200)
        crash_stop: If True, stop on first boundary collision (default False)
        stall_timeout: Seconds of no/negative progress before stopping (default None = disabled)

    Lap progress is read from the ``Laptimer`` extension, which counts laps from
    each car's actual starting point; this extension must be ordered after it.
    """

    def __init__(self):
        Extension.__init__(self, "auto_stopper")
        self.lap_count = 1
        self.wall_clock = 1200  # seconds (20 min)
        self.crash_stop = False
        self.stall_timeout = None

        self.stop_reason = None

    def post_init(self):
        self.start_time = time.time()

        # Progress tracking for stall detection.  Laptimer progress starts at 0.
        self._last_progress_time = time.time()
        self._last_progress = {car: 0.0 for car in self.main.cars}

    def update(self):
        # Wall-clock timeout
        if time.time() - self.start_time >= self.wall_clock:
            self.print_ok(
                "Wall-clock timeout (%ds) reached, stopping.", self.wall_clock
            )
            self.stop_reason = "wall_clock"
            self.main.exit_request.set()
            return

        for car in self.main.cars:
            # Crash stop
            if self.crash_stop and getattr(car, 'in_collision', False):
                self.print_ok("Car %d crashed, stopping.", car.id)
                self.stop_reason = "crash"
                self.main.exit_request.set()
                return

            # Lap detection: the Laptimer counts full laps from the car's start.
            if self.main.laptimer.total_laps_by_car[car] >= self.lap_count:
                self.print_ok(
                    "Car %d finished race (%d laps), stopping.",
                    car.id, self.lap_count,
                )
                self.stop_reason = "race_finished"
                self.main.exit_request.set()
                return

            # Stall detection (reuses the Laptimer's monotonic progress).
            if self.stall_timeout is not None:
                progress = self.main.laptimer.progress_by_car[car]
                if progress - self._last_progress[car] > 0.01:  # forward progress
                    self._last_progress_time = time.time()
                    self._last_progress[car] = progress
                if time.time() - self._last_progress_time >= self.stall_timeout:
                    self.print_ok(
                        "Car %d stalled for %ds, stopping.",
                        car.id, self.stall_timeout,
                    )
                    self.stop_reason = "stall"
                    self.main.exit_request.set()
                    return
