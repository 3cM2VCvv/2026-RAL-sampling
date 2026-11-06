"""Parent-side extensions for the controller sweep."""

from buzzracer.extensions.auto_stopper import AutoStopper


class SimTimeAutoStopper(AutoStopper):
    """``AutoStopper`` with wall-clock and stall limits in simulated seconds.

    Same stop reasons and thresholds as ``AutoStopper``; only the clock differs,
    so slower controllers are not penalised by real-time execution speed.
    """

    def post_init(self):
        self.start_time = self.main.time
        self._last_progress_time = self.main.time
        self._last_progress = {car: 0.0 for car in self.main.cars}

    def update(self):
        now = self.main.time

        if now - self.start_time >= self.wall_clock:
            self.print_ok(
                "Wall-clock timeout (%ds) reached, stopping.", self.wall_clock
            )
            self.stop_reason = "wall_clock"
            self.main.exit_request.set()
            return

        for car in self.main.cars:
            if self.crash_stop and getattr(car, "in_collision", False):
                self.print_ok("Car %d crashed, stopping.", car.id)
                self.stop_reason = "crash"
                self.main.exit_request.set()
                return

            if self.main.laptimer.total_laps_by_car[car] >= self.lap_count:
                self.print_ok(
                    "Car %d finished race (%d laps), stopping.",
                    car.id,
                    self.lap_count,
                )
                self.stop_reason = "race_finished"
                self.main.exit_request.set()
                return

            if self.stall_timeout is not None:
                progress = self.main.laptimer.progress_by_car[car]
                if progress - self._last_progress[car] > 0.01:
                    self._last_progress_time = now
                    self._last_progress[car] = progress
                if now - self._last_progress_time >= self.stall_timeout:
                    self.print_ok(
                        "Car %d stalled for %ds, stopping.",
                        car.id,
                        self.stall_timeout,
                    )
                    self.stop_reason = "stall"
                    self.main.exit_request.set()
                    return


def register_simtime_stopper():
    from buzzracer import extensions

    extensions.SimTimeAutoStopper = SimTimeAutoStopper
