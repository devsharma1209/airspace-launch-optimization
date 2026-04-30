"""
POMDP extension: surveillance noise + particle filter + QMDP-style action
selection on top of the Q-learning table.

We model uncertainty in the EGO aircraft's OWN position (think ADS-B/GPS
jitter + reporting latency). The true position is still simulated inside
MDPEnv, but the agent only sees noisy observations and must maintain a
belief over its current position.

Action selection uses QMDP (Littman et al.): for each action, take the
expected Q-value over the belief distribution, and pick the best action.
QMDP is cheap and works well when most of the uncertainty is transient.
"""

import math
import random
import numpy as np

from .geometry import move_by_heading
from .mdp_env import MDPEnv, ACTION_HEADING_DELTA, STEP_DURATION_SECONDS


class NoisyObservationEnv(MDPEnv):
    """MDPEnv that returns a noisy (lat, lon) observation instead of the truth."""

    def __init__(self, *args, obs_noise_nm=1.0, dropout_prob=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_noise_nm = obs_noise_nm
        self.dropout_prob = dropout_prob

    def observe_noisy(self):
        """Return (obs_lat, obs_lon) or None if the observation was dropped."""
        if random.random() < self.dropout_prob:
            return None
        # 1 nm ~ 1/60 deg latitude
        noise_deg = self.obs_noise_nm / 60.0
        return (
            self.ego.lat + random.gauss(0, noise_deg),
            self.ego.lon + random.gauss(0, noise_deg),
        )


# ---------------------------------------------------------------------------
# Particle filter over (lat, lon)
# ---------------------------------------------------------------------------

class ParticleFilter:
    def __init__(self, n_particles=200, init_lat=0.0, init_lon=0.0,
                 process_noise_nm=0.3, obs_noise_nm=1.0):
        self.n = n_particles
        self.particles = np.array(
            [[init_lat, init_lon] for _ in range(n_particles)], dtype=float
        )
        self.weights = np.ones(n_particles) / n_particles
        self.process_noise_deg = process_noise_nm / 60.0
        self.obs_noise_deg = obs_noise_nm / 60.0

    def predict(self, heading, speed_kts, dt_seconds):
        """Propagate every particle forward by one time step."""
        for i in range(self.n):
            lat, lon = move_by_heading(
                self.particles[i, 0], self.particles[i, 1],
                heading, speed_kts, dt_seconds,
            )
            lat += random.gauss(0, self.process_noise_deg)
            lon += random.gauss(0, self.process_noise_deg)
            self.particles[i, 0] = lat
            self.particles[i, 1] = lon

    def update(self, obs):
        """Reweight particles by likelihood of the observation."""
        if obs is None:
            return  # dropout — no update
        o_lat, o_lon = obs
        sigma = self.obs_noise_deg
        dlat = self.particles[:, 0] - o_lat
        dlon = self.particles[:, 1] - o_lon
        log_w = -(dlat ** 2 + dlon ** 2) / (2 * sigma ** 2)
        w = np.exp(log_w - log_w.max())
        self.weights = w / w.sum()
        # Resample if effective sample size is low
        ess = 1.0 / np.sum(self.weights ** 2)
        if ess < self.n / 2:
            self._resample()

    def _resample(self):
        idx = np.random.choice(self.n, size=self.n, p=self.weights)
        self.particles = self.particles[idx]
        self.weights = np.ones(self.n) / self.n

    def mean(self):
        return (
            float(np.average(self.particles[:, 0], weights=self.weights)),
            float(np.average(self.particles[:, 1], weights=self.weights)),
        )

    def sample(self):
        """Return a single (lat, lon) sampled from the belief."""
        idx = np.random.choice(self.n, p=self.weights)
        return float(self.particles[idx, 0]), float(self.particles[idx, 1])


# ---------------------------------------------------------------------------
# QMDP: expected Q over belief particles
# ---------------------------------------------------------------------------

class QMDPAgent:
    """
    Uses a trained Q-table (from q_learning.py) plus a belief distribution
    to pick actions when the true state is unknown.
    """

    def __init__(self, q_agent, env):
        self.q = q_agent
        self.env = env  # used for state discretisation helpers

    def _state_from_position(self, lat, lon, heading):
        # Temporarily place ego at (lat, lon, heading) to reuse _observe()
        saved = (self.env.ego.lat, self.env.ego.lon, self.env.ego.heading)
        self.env.ego.lat = lat
        self.env.ego.lon = lon
        self.env.ego.heading = heading
        s = self.env._observe()
        (self.env.ego.lat, self.env.ego.lon, self.env.ego.heading) = saved
        return s

    def act(self, particle_filter, heading):
        """Pick the action with the highest expected Q across belief particles."""
        n_actions = self.q.n_actions
        expected_q = np.zeros(n_actions)
        for i in range(particle_filter.n):
            lat = float(particle_filter.particles[i, 0])
            lon = float(particle_filter.particles[i, 1])
            w = particle_filter.weights[i]
            s = self._state_from_position(lat, lon, heading)
            q_values = self.q.Q[s]
            for a in range(n_actions):
                expected_q[a] += w * q_values[a]
        return int(np.argmax(expected_q))


# ---------------------------------------------------------------------------
# Rollout helper
# ---------------------------------------------------------------------------

def rollout_pomdp(env, q_agent, n_particles=200, max_steps=500, verbose=False):
    """Run one POMDP episode; returns metrics dict."""
    assert isinstance(env, NoisyObservationEnv), "env must be NoisyObservationEnv"
    env.reset()

    pf = ParticleFilter(
        n_particles=n_particles,
        init_lat=env.ego.lat,
        init_lon=env.ego.lon,
        obs_noise_nm=env.obs_noise_nm,
    )
    qmdp = QMDPAgent(q_agent, env)

    total_reward = 0.0
    tfr_v = 0
    sep_v = 0
    reached = False

    for t in range(max_steps):
        action = qmdp.act(pf, env.ego.heading)

        # Take step in the real environment
        _, r, done, info = env.step(action)
        total_reward += r
        if info.get("tfr_violation"): tfr_v += 1
        if info.get("separation_violation"): sep_v += 1

        # Belief update: predict with our commanded heading + speed,
        # then correct with the noisy observation
        applied_heading = (env.ego.heading)  # ego already updated inside step()
        pf.predict(applied_heading, env.ego.speed, STEP_DURATION_SECONDS)
        obs = env.observe_noisy()
        pf.update(obs)

        if done:
            reached = info.get("outcome") == "reached"
            break

    return {
        "reward": total_reward,
        "tfr_violations": tfr_v,
        "sep_violations": sep_v,
        "reached": reached,
        "steps": env.t,
    }
