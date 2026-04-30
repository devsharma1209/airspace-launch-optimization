"""
Tabular Q-learning over the discretised MDP state.

Because the state space is small (5 x 8 x 4 = 160 states, 5 actions) we can
store Q as a plain dict and still converge in a few hundred training episodes.
"""

import random
import pickle
from collections import defaultdict


class QLearningAgent:
    def __init__(self, n_actions, alpha=0.2, gamma=0.95,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.995):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.Q = defaultdict(lambda: [0.0] * n_actions)

    # ---------------------------------------------------------

    def act(self, state, greedy=False):
        if not greedy and random.random() < self.eps:
            return random.randrange(self.n_actions)
        q = self.Q[state]
        return max(range(self.n_actions), key=lambda a: q[a])

    def update(self, s, a, r, s_next, done):
        q_next = 0.0 if done else max(self.Q[s_next])
        target = r + self.gamma * q_next
        self.Q[s][a] += self.alpha * (target - self.Q[s][a])

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    # ---------------------------------------------------------

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(dict(self.Q), f)

    def load(self, path):
        with open(path, "rb") as f:
            self.Q = defaultdict(lambda: [0.0] * self.n_actions, pickle.load(f))


# -------------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------------

def train(env, episodes=300, max_steps=500, verbose=True):
    agent = QLearningAgent(env.n_actions)
    history = []
    for ep in range(episodes):
        s = env.reset()
        total = 0.0
        for _ in range(max_steps):
            a = agent.act(s)
            s2, r, done, info = env.step(a)
            agent.update(s, a, r, s2, done)
            s = s2
            total += r
            if done:
                break
        agent.decay_epsilon()
        history.append(total)
        if verbose and ep % max(1, episodes // 10) == 0:
            print(f"  ep {ep:4d} | reward {total:8.1f} | eps {agent.eps:.3f}")
    return agent, history


def evaluate(env, agent, episodes=20, max_steps=500):
    """Run greedy rollouts and return averaged metrics."""
    results = {"reward": [], "tfr_violations": [], "sep_violations": [], "reached": 0}
    for _ in range(episodes):
        s = env.reset()
        total = 0.0
        tfr_v = 0
        sep_v = 0
        reached = False
        for _ in range(max_steps):
            a = agent.act(s, greedy=True)
            s, r, done, info = env.step(a)
            total += r
            if info.get("tfr_violation"):
                tfr_v += 1
            if info.get("separation_violation"):
                sep_v += 1
            if done:
                reached = info.get("outcome") == "reached"
                break
        results["reward"].append(total)
        results["tfr_violations"].append(tfr_v)
        results["sep_violations"].append(sep_v)
        if reached:
            results["reached"] += 1
    return results
