"""
MDP environment wrapping the simulation for a single controlled ego-aircraft.

State  (discretised):
    dist_to_tfr_bucket     : {0: inside, 1: <5nm, 2: 5-15, 3: 15-40, 4: >40}
    bearing_delta_bucket   : angular error (heading vs bearing-to-dest), 8 bins
    min_sep_bucket         : {0: <3nm, 1: 3-5, 2: 5-10, 3: >10}

Actions:
    0 = maintain heading
    1 = turn left 15 deg
    2 = turn right 15 deg
    3 = turn left 30 deg
    4 = turn right 30 deg

Reward:
    -100  if inside a TFR
     -20  if separation < 3nm
      -5  if separation < 5nm
      +1  per step that made progress (got closer to destination)
      +50 on reaching destination
     -0.1 baseline cost per step (encourage efficiency)

The env runs many OTHER aircraft in REPLAY mode (as traffic), and ONE ego
aircraft that responds to the action.
"""

import numpy as np

from .aircraft import AircraftAgent, STEP_DURATION_SECONDS
from .geometry import (
    bearing_between,
    haversine,
    move_by_heading,
)


N_ACTIONS = 5
ACTION_HEADING_DELTA = {0: 0.0, 1: -15.0, 2: 15.0, 3: -30.0, 4: 30.0}


def _bucket(value, edges):
    """Return the index of the bucket that contains `value`."""
    for i, e in enumerate(edges):
        if value < e:
            return i
    return len(edges)


class EgoAircraft:
    """A minimal aircraft the Q-learning policy controls."""

    def __init__(self, trajectory):
        r0 = trajectory.iloc[0]
        last = trajectory.iloc[-1]
        self.start_lat = float(r0["lat"])
        self.start_lon = float(r0["lon"])
        self.destination_lat = float(last["lat"])
        self.destination_lon = float(last["lon"])
        self.nominal_speed = float(trajectory["Speed"].mean())
        self.reset()

    def reset(self):
        self.lat = self.start_lat
        self.lon = self.start_lon
        self.speed = self.nominal_speed
        self.heading = bearing_between(
            self.lat, self.lon, self.destination_lat, self.destination_lon
        )
        self.history_lat = [self.lat]
        self.history_lon = [self.lon]

    def apply_action(self, action):
        self.heading = (self.heading + ACTION_HEADING_DELTA[action]) % 360
        self.lat, self.lon = move_by_heading(
            self.lat, self.lon, self.heading, self.speed, STEP_DURATION_SECONDS
        )
        self.history_lat.append(self.lat)
        self.history_lon.append(self.lon)


class MDPEnv:
    def __init__(self, ego_trajectory, traffic_data, tfrs,
                 max_steps=500, sep_warn_nm=5.0, sep_min_nm=3.0):
        self.ego_trajectory = ego_trajectory
        self.traffic_data = traffic_data
        self.tfrs = list(tfrs)
        self.max_steps = max_steps
        self.sep_warn_nm = sep_warn_nm
        self.sep_min_nm = sep_min_nm

        self.ego = EgoAircraft(ego_trajectory)
        self._build_traffic()

    # ----------------------------------------------------------

    def _build_traffic(self):
        """Create replay-only aircraft for every background flight."""
        self.traffic = []
        for cs, df in self.traffic_data.items():
            # simple shim: wrap in an object with .step(), lat, lon, finished
            self.traffic.append(_ReplayAircraft(cs, df))

    # ----------------------------------------------------------
    # Public RL API
    # ----------------------------------------------------------

    def reset(self):
        self.t = 0
        self.done = False
        self.ego.reset()
        for ac in self.traffic:
            ac.reset()
        self.prev_dist_to_dest = haversine(
            self.ego.lat, self.ego.lon,
            self.ego.destination_lat, self.ego.destination_lon,
        )
        return self._observe()

    def step(self, action):
        if self.done:
            return self._observe(), 0.0, True, {}

        self.ego.apply_action(action)
        for ac in self.traffic:
            ac.step()

        self.t += 1
        reward, info = self._reward()
        self.done = info["terminal"] or self.t >= self.max_steps
        return self._observe(), reward, self.done, info

    # ----------------------------------------------------------

    def _min_tfr_distance(self):
        """Distance to nearest TFR edge (negative = inside)."""
        best = float("inf")
        for tfr in self.tfrs:
            d = tfr.distance_to_edge(self.ego.lat, self.ego.lon)
            if d < best:
                best = d
        return best

    def _inside_any_tfr(self):
        return any(t.contains(self.ego.lat, self.ego.lon) for t in self.tfrs)

    def _min_separation(self):
        best = float("inf")
        for ac in self.traffic:
            if ac.finished:
                continue
            d = haversine(self.ego.lat, self.ego.lon, ac.lat, ac.lon)
            if d < best:
                best = d
        return best

    def _observe(self):
        d_tfr = self._min_tfr_distance()
        if self._inside_any_tfr():
            tfr_bucket = 0
        else:
            tfr_bucket = 1 + _bucket(d_tfr, [5.0, 15.0, 40.0])  # 1..4

        bearing_to_dest = bearing_between(
            self.ego.lat, self.ego.lon,
            self.ego.destination_lat, self.ego.destination_lon,
        )
        delta = ((bearing_to_dest - self.ego.heading + 540) % 360) - 180  # -180..180
        # 8 bins of 45 deg each
        bearing_bucket = int((delta + 180) // 45) % 8

        d_sep = self._min_separation()
        sep_bucket = _bucket(d_sep, [3.0, 5.0, 10.0])  # 0..3

        return (tfr_bucket, bearing_bucket, sep_bucket)

    def _reward(self):
        info = {"terminal": False}

        # Reached destination?
        d_dest = haversine(
            self.ego.lat, self.ego.lon,
            self.ego.destination_lat, self.ego.destination_lon,
        )
        if d_dest < 10.0:
            info["terminal"] = True
            info["outcome"] = "reached"
            return 50.0, info

        r = -0.1  # per-step cost

        # Progress shaping
        if d_dest < self.prev_dist_to_dest:
            r += 1.0
        self.prev_dist_to_dest = d_dest

        # TFR violation
        if self._inside_any_tfr():
            r -= 100.0
            info["tfr_violation"] = True

        # Separation
        d_sep = self._min_separation()
        if d_sep < self.sep_min_nm:
            r -= 20.0
            info["separation_violation"] = True
        elif d_sep < self.sep_warn_nm:
            r -= 5.0

        return r, info

    @property
    def observation_space_size(self):
        return 5 * 8 * 4  # tfr x bearing x sep

    @property
    def n_actions(self):
        return N_ACTIONS


# ---------------------------------------------------------------------------
# Minimal replay-only aircraft for background traffic
# ---------------------------------------------------------------------------

class _ReplayAircraft:
    def __init__(self, unique_id, trajectory):
        self.unique_id = unique_id
        self.trajectory = trajectory
        self.reset()

    def reset(self):
        self.step_index = 0
        r0 = self.trajectory.iloc[0]
        self.lat = float(r0["lat"])
        self.lon = float(r0["lon"])
        self.finished = False

    def step(self):
        if self.step_index < len(self.trajectory) - 1:
            self.step_index += 1
            row = self.trajectory.iloc[self.step_index]
            self.lat = float(row["lat"])
            self.lon = float(row["lon"])
        else:
            self.finished = True
