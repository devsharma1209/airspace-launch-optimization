"""
Phase 4 deliverable: load the Q-table trained in Phase 3, add surveillance
noise + dropout, and run QMDP with a particle filter for belief tracking.

Usage:
    python run_pomdp.py
"""

import statistics as st

from src.data_loader import load_aircraft_data
from src.tfr import load_tfrs
from src.pomdp import NoisyObservationEnv, rollout_pomdp
from src.q_learning import QLearningAgent
from src.mdp_env import MDPEnv
from run_mdp import pick_ego

DATA_FOLDER = r"Datasets/Validation data (FlightRader24)"
TFR_FILE = r"tfr/TFR_Lat_Lon.xlsx"


def main():
    print("Loading data...")
    aircraft_data = load_aircraft_data(DATA_FOLDER, verbose=False)
    tfrs = load_tfrs(TFR_FILE)

    ego_cs, _ = pick_ego(aircraft_data, tfrs)
    print(f"Ego aircraft: {ego_cs}")

    ego_df = aircraft_data[ego_cs]
    traffic = {cs: df for cs, df in aircraft_data.items() if cs != ego_cs}

    # Load trained Q-table
    q_agent = QLearningAgent(n_actions=5)
    try:
        q_agent.load("q_table.pkl")
    except FileNotFoundError:
        print("q_table.pkl not found. Run `python run_mdp.py` first.")
        return

    # ---- MDP baseline (no noise) ----
    mdp_env = MDPEnv(ego_df, traffic, list(tfrs.values()), max_steps=400)
    from src.q_learning import evaluate
    mdp_results = evaluate(mdp_env, q_agent, episodes=10, max_steps=400)
    print("\nMDP (no noise):")
    print(f"  avg reward : {st.mean(mdp_results['reward']):.1f}")
    print(f"  tfr v      : {st.mean(mdp_results['tfr_violations']):.2f}")
    print(f"  sep v      : {st.mean(mdp_results['sep_violations']):.2f}")
    print(f"  reached    : {mdp_results['reached']}/10")

    # ---- POMDP under noise ----
    for noise_nm in [0.5, 1.5, 3.0]:
        print(f"\nPOMDP obs_noise={noise_nm} nm, dropout=15%")
        env = NoisyObservationEnv(
            ego_df, traffic, list(tfrs.values()),
            max_steps=400, obs_noise_nm=noise_nm, dropout_prob=0.15,
        )
        rewards, tfr_v, sep_v, reached = [], [], [], 0
        for _ in range(10):
            r = rollout_pomdp(env, q_agent, n_particles=150, max_steps=400)
            rewards.append(r["reward"])
            tfr_v.append(r["tfr_violations"])
            sep_v.append(r["sep_violations"])
            if r["reached"]:
                reached += 1
        print(f"  avg reward : {st.mean(rewards):.1f}")
        print(f"  tfr v      : {st.mean(tfr_v):.2f}")
        print(f"  sep v      : {st.mean(sep_v):.2f}")
        print(f"  reached    : {reached}/10")


if __name__ == "__main__":
    main()
