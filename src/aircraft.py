"""
Aircraft agent.

Two navigation modes:
    'replay'  -> follow the recorded CSV trajectory (historical baseline)
    'policy'  -> rule-based avoidance: turn away if a TFR or separation
                 conflict is predicted within the lookahead horizon
"""

from .geometry import (
    bearing_between,
    haversine,
    move_by_heading,
)

STEP_DURATION_SECONDS = 30
CONFLICT_LOOKAHEAD_STEPS = 20
SEPARATION_WARNING_NM = 5.0
DEST_REACHED_NM = 10.0


class AircraftAgent:
    def __init__(self, unique_id, model, trajectory, mode="replay"):
        self.unique_id = unique_id
        self.model = model
        self.trajectory = trajectory
        self.step_index = 0
        self.mode = mode

        r0 = trajectory.iloc[0]
        self.lat = float(r0["lat"])
        self.lon = float(r0["lon"])
        self.altitude = float(r0["Altitude"])
        self.speed = float(r0["Speed"])

        if len(trajectory) > 1:
            r1 = trajectory.iloc[1]
            self.heading = bearing_between(r0["lat"], r0["lon"], r1["lat"], r1["lon"])
        else:
            self.heading = float(r0.get("Direction", 0))

        last = trajectory.iloc[-1]
        self.destination_lat = float(last["lat"])
        self.destination_lon = float(last["lon"])

        self.nominal_speed = float(trajectory["Speed"].mean())

        # Rule-based avoidance state
        self.is_avoiding = False
        self.avoidance_steps_remaining = 0

        self.finished = False

        # History for plotting
        self.history_lat = [self.lat]
        self.history_lon = [self.lon]
        self.history_altitude = [self.altitude]
        self.history_time = [0]

    # ------------------------------------------------------------
    # Conflict prediction
    # ------------------------------------------------------------

    def predict_future_position(self, steps_ahead):
        lat, lon = self.lat, self.lon
        for _ in range(steps_ahead):
            lat, lon = move_by_heading(lat, lon, self.heading, self.speed, STEP_DURATION_SECONDS)
        return lat, lon

    def tfr_conflict_ahead(self):
        """Will we enter any TFR within the lookahead window?"""
        for step in range(1, CONFLICT_LOOKAHEAD_STEPS + 1):
            lat, lon = self.predict_future_position(step)
            for tfr in self.model.tfrs:
                if tfr.contains(lat, lon):
                    return tfr
        return None

    def separation_conflicts_ahead(self):
        threats = []
        for other in self.model.aircraft_agents:
            if other.unique_id == self.unique_id or other.finished:
                continue
            for step in range(1, CONFLICT_LOOKAHEAD_STEPS + 1):
                my_lat, my_lon = self.predict_future_position(step)
                their_lat, their_lon = other.predict_future_position(step)
                if haversine(my_lat, my_lon, their_lat, their_lon) < SEPARATION_WARNING_NM:
                    threats.append(other)
                    break
        return threats

    def heading_away_from(self, threat_lat, threat_lon):
        """Turn 90° off the line to the threat."""
        direct = bearing_between(self.lat, self.lon, threat_lat, threat_lon)
        return (direct + 90) % 360

    # ------------------------------------------------------------
    # Step
    # ------------------------------------------------------------

    def _step_replay(self):
        if self.step_index < len(self.trajectory) - 1:
            self.step_index += 1
            row = self.trajectory.iloc[self.step_index]
            self.heading = bearing_between(self.lat, self.lon, float(row["lat"]), float(row["lon"]))
            self.lat = float(row["lat"])
            self.lon = float(row["lon"])
            self.altitude = float(row["Altitude"])
            self.speed = float(row["Speed"])
        else:
            self.finished = True

    def _step_policy(self):
        if self.finished:
            return
        if haversine(self.lat, self.lon, self.destination_lat, self.destination_lon) < DEST_REACHED_NM:
            self.finished = True
            return

        if self.is_avoiding and self.avoidance_steps_remaining > 0:
            self.avoidance_steps_remaining -= 1
        else:
            self.is_avoiding = False
            tfr = self.tfr_conflict_ahead()
            if tfr is not None:
                c_lat, c_lon = tfr.centroid()
                self.heading = self.heading_away_from(c_lat, c_lon)
                self.is_avoiding = True
                self.avoidance_steps_remaining = 15
            else:
                threats = self.separation_conflicts_ahead()
                if threats:
                    t = threats[0]
                    self.heading = self.heading_away_from(t.lat, t.lon)
                    self.is_avoiding = True
                    self.avoidance_steps_remaining = 10
                else:
                    self.heading = bearing_between(
                        self.lat, self.lon, self.destination_lat, self.destination_lon
                    )

        self.lat, self.lon = move_by_heading(
            self.lat, self.lon, self.heading, self.speed, STEP_DURATION_SECONDS
        )

    def step(self):
        if self.mode == "replay":
            self._step_replay()
        else:
            self._step_policy()

        self.history_lat.append(self.lat)
        self.history_lon.append(self.lon)
        self.history_altitude.append(self.altitude)
        self.history_time.append(self.model.time)
