"""
AirTrafficModel — the Mesa environment that ticks agents and logs violations.

Supports multiple polygon TFRs.
"""

from .aircraft import AircraftAgent
from .geometry import haversine

SEPARATION_MIN_NM = 3.0  # ICAO radar separation minimum


class AirTrafficModel:
    def __init__(self, aircraft_data, tfrs, mode="replay", verbose=True):
        self.time = 0
        self.mode = mode
        self.verbose = verbose

        # tfrs : list of PolygonTFR
        self.tfrs = list(tfrs)

        # Event logs (entry events only, not every tick)
        self.violations = []            # (callsign, tfr_name, step)
        self.separation_violations = [] # (cs1, cs2, step, nm)

        self._active_tfr = set()  # set of (callsign, tfr_name) currently violating
        self._active_sep = set()  # set of frozenset({cs1, cs2}) currently violating

        self.aircraft_agents = []
        for cs, df in aircraft_data.items():
            a = AircraftAgent(cs, self, df, mode=mode)
            self.aircraft_agents.append(a)

        if verbose:
            print(f"Model initialised: {len(self.aircraft_agents)} aircraft, "
                  f"{len(self.tfrs)} TFR zones, mode='{mode}'")

    # ------------------------------------------------------------

    def _check_tfr(self):
        for ac in self.aircraft_agents:
            if ac.finished:
                continue
            for tfr in self.tfrs:
                key = (ac.unique_id, tfr.name)
                if tfr.contains(ac.lat, ac.lon):
                    if key not in self._active_tfr:
                        self.violations.append((ac.unique_id, tfr.name, self.time))
                        self._active_tfr.add(key)
                        if self.verbose:
                            print(f"  [t={self.time}] TFR {tfr.name} violated by {ac.unique_id}")
                else:
                    self._active_tfr.discard(key)

    def _check_separation(self):
        agents = self.aircraft_agents
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a, b = agents[i], agents[j]
                if a.finished or b.finished:
                    continue
                d = haversine(a.lat, a.lon, b.lat, b.lon)
                pair = frozenset([a.unique_id, b.unique_id])
                if d < SEPARATION_MIN_NM:
                    if pair not in self._active_sep:
                        self.separation_violations.append(
                            (a.unique_id, b.unique_id, self.time, round(float(d), 2))
                        )
                        self._active_sep.add(pair)
                        if self.verbose:
                            print(f"  [t={self.time}] separation: {a.unique_id} - {b.unique_id} ({d:.2f} nm)")
                else:
                    self._active_sep.discard(pair)

    # ------------------------------------------------------------

    def step(self):
        for a in self.aircraft_agents:
            a.step()
        self.time += 1
        self._check_tfr()
        self._check_separation()

    def run(self, n_steps):
        for _ in range(n_steps):
            self.step()

    def metrics(self):
        return {
            "mode": self.mode,
            "aircraft_count": len(self.aircraft_agents),
            "steps": self.time,
            "tfr_violations": len(self.violations),
            "separation_violations": len(self.separation_violations),
        }
