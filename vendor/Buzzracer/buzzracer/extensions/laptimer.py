''' Track laptimes of cars'''
from __future__ import annotations
from typing import TYPE_CHECKING

import os
from time import time
from threading import Thread, Event
from math import sin, cos
import logging

import pickle
import numpy as np

from buzzracer.common import BASEDIR
from buzzracer.extensions.extension import Extension
if TYPE_CHECKING:
    from buzzracer.cars.car import Car


class Laptimer(Extension):
    ''' Laptimer for simulation and experiments.

    Laps are measured by arc-length progress along the raceline, counted from
    each car's *actual* starting point (its position on the first update), so a
    lap is a full loop back to where that car started -- not a crossing of a
    fixed track finish line.  This is robust to multiple laps, fast laps, and
    arbitrary start positions, and a lap message is printed only when a full lap
    is completed from the start.
    '''

    def __init__(self):
        Extension.__init__(self, 'laptimer')
        # self.text_logger.setLevel(logging.WARNING)

        cars = self.main.cars
        self.laptime_mean_by_car: dict[Car, float] = {car: 0 for car in cars}
        ''' Mean laptime for each car '''
        self.laptime_stddev_by_car: dict[Car, float] = {car: 0 for car in cars}
        ''' Stddev for each car '''
        self.total_laps_by_car: dict[Car, int] = {car: 0 for car in cars}
        ''' Full laps completed from the start, for each car '''
        self.laptime_vec_by_car: dict[Car, float] = {car: [] for car in cars}
        ''' Laptime history for each car '''
        self.progress_by_car: dict[Car, float] = {car: 0.0 for car in cars}
        ''' Net arc-length distance travelled since start, for each car '''

        # Per-car arc-length tracking state, baselined lazily on first update
        # (the car's start state may be overridden after post_init runs).
        self._start_s: dict[Car, float] = {car: None for car in cars}
        self._prev_s: dict[Car, float] = {car: None for car in cars}
        self._last_lap_time: dict[Car, float] = {car: 0.0 for car in cars}

    def _car_s(self, car):
        ''' Nearest raceline arc-length s for a car's current (x, y). '''
        xy = np.array([car.state[0], car.state[1]])
        track = self.main.track
        dists = np.sum((track.raceline_points - xy.reshape(2, 1)) ** 2, axis=0)
        return float(track.ss[int(np.argmin(dists))])

    def update(self):
        track_len = self.main.track.raceline_len_m
        for car in self.main.cars:
            s = self._car_s(car)

            # Lazy baseline: first update establishes this car's start point.
            if self._start_s[car] is None:
                self._start_s[car] = s
                self._prev_s[car] = s
                self._last_lap_time[car] = self.main.time
                continue

            # Accumulate wrap-corrected forward progress.
            delta = s - self._prev_s[car]
            if delta > track_len / 2:
                delta -= track_len
            elif delta < -track_len / 2:
                delta += track_len
            self.progress_by_car[car] += delta
            self._prev_s[car] = s

            laps_done = int(np.floor(self.progress_by_car[car] / track_len))
            while self.total_laps_by_car[car] < laps_done:
                laptime = self.main.time - self._last_lap_time[car]
                self._last_lap_time[car] = self.main.time
                self.total_laps_by_car[car] += 1
                self.laptime_vec_by_car[car].append(laptime)
                self.print_info(
                    'car%d completed lap %d (from start), laptime: %.4f s' %
                    (car.id, self.total_laps_by_car[car], laptime))
                self.main.log_metric(f'laptime_car{car.id}', laptime)

    def final(self):
        for car in self.main.cars:
            car.debug_dict.update(
                {'laptime_vec': self.laptime_vec_by_car[car]})
        self.show_stats()
        self.log_laptime()

    def log_laptime(self):
        try:
            logname = os.path.join(self.main.logger.log_folder,
                                   f'laptime{self.main.logger.log_no}.p')
        except AttributeError:
            logname = os.path.join(BASEDIR, 'log', 'laptime_latest.p')
            self.print_warning(
                f"Logger extension wasn't enabled, saving to {logname}")
        with open(logname, 'wb') as f:
            laptime_vec_by_car_id = {
                car.id: self.laptime_vec_by_car[car]
                for car in self.main.cars
            }
            pickle.dump(laptime_vec_by_car_id, f)
            self.main.register_artifact(logname)
            self.print_ok('Saved laptime vec to ' + logname)

    def show_stats(self):
        for car in self.main.cars:
            if len(self.laptime_vec_by_car[car]) > 0:
                mean = np.mean(self.laptime_vec_by_car[car])
                stddev = np.std(self.laptime_vec_by_car[car])
                laps = len(self.laptime_vec_by_car[car])
                self.print_info(
                    'car%d, %d laps, mean %.4f, stddev %.4f (sec)' %
                    (car.id, laps, mean, stddev))
            else:
                mean = -1
                stddev = -1
                self.print_warning(f'car{car.id} has no laps')
            self.laptime_mean_by_car[car] = mean
            self.laptime_stddev_by_car[car] = stddev
            self.main.log_metric(f'laptime_mean_car{car.id}', mean)
            self.main.log_metric(f'laptime_stddev_car{car.id}', stddev)


class _Laptimer:

    def __init__(self, finish: np.array, orientation: float, voice=False):
        ''' Internal Laptimer
        Args:
            finish: np.array dim (2,)
            orientation: float Orientation in rad for track orientation at finish
        '''
        self.finish = np.array(finish)
        ''' Coordinate dim(2,) for finishing line '''
        self.orientation = orientation
        ''' Orientation in rad for track orientation at finish '''
        self.finish_vec = np.array([cos(orientation), sin(orientation)])

        self.voice = voice
        if voice:
            global pyttsx3
            import pyttsx3
            self.engine = pyttsx3.init()

        self.last_coord = np.array([0, 0])
        self.last_lap_ts = 0
        self.last_laptime = 0

        # indicator that this lap is new
        self.new_lap = Event()
        self.lap_count = 0
        self.p1dist = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])
        self.p1norm = lambda a: abs(a[0]) + abs(a[1])

        # freeze laptimer for a certain time after a new lap to prevent immediate recounting
        self.timeout = 1.0
        self.hotzone_radius = 0.5
        self.thread = None

    def update(self, coord, current_time=None):
        # let the finish location be O
        # last position/coord be A
        # current position be B
        if current_time is None:
            current_time = time()
        if current_time < self.last_lap_ts + self.timeout:
            self.new_lap.clear()
            return False
        coord = np.array(coord)
        OB = coord - self.finish
        if self.p1norm(OB) > self.hotzone_radius:
            self.last_coord = coord
            self.new_lap.clear()
            return False
        OA = self.last_coord - self.finish
        if np.dot(OA, self.finish_vec) * np.dot(OB, self.finish_vec) < 0:
            self.last_laptime = current_time - self.last_lap_ts
            self.last_lap_ts = current_time
            self.lap_count += 1
            self.new_lap.set()
            return True

    def announce(self):
        self.thread = Thread(target=self.__announceThread)
        self.thread.start()

    def __announceThread(self):
        self.engine.say('%.2f' % self.last_laptime)
        self.engine.runAndWait()
        return
