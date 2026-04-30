"""
Phase 3 deliverable: train Q-learning on a single ego aircraft navigating
around the polygon TFRs and evaluate against the rule-based baseline.

Pick the flight whose route most clearly crosses a TFR as the ego; the rest
act as background traffic.

Usage:
    python run_mdp.py
"""

from src.data_loader import load_aircraft_data
from src.tfr import load_tfrs
from src.mdp_env import MDPEnv
from src.q_learning import train, evaluate

DATA_FOLDER = r"Datasets/Validation data (FlightRader24)"
TFR_FILE = r"tfr/TFR_Lat_Lon.xlsx"
EGO_CALLSIGN = None  # None -> pick the flight whose trajectory enters most TFRs


def pick_ego(aircraft_data, tfrs):
    """Pick the aircraft whose CSV path enters the most TFR polygons."""
    best_cs, best_count = None, -1
    for cs, df in aircraft_data.items():
        count = 0
        for _, row in df.iterrows():
            for tfr in tfrs.values():
                if tfr.contains(row["lat"], row["lon"]):
                    count += 1
                    break
        if count > best_count:
            best_cs, best_count = cs, count
    return best_cs, best_count


def main():
    print("Loading data...")
    aircraft_data = load_aircraft_data(DATA_FOLDER, verbose=False)
    tfrs = load_tfrs(TFR_FILE)

    ego_cs = EGO_CALLSIGN or pick_ego(aircraft_data, tfrs)[0]
    print(f"Ego aircraft: {ego_cs}")

    ego_df = aircraft_data[ego_cs]
    traffic = {cs: df for cs, df in aircraft_data.items() if cs != ego_cs}

    env = MDPEnv(ego_df, traffic, list(tfrs.values()), max_steps=400)

    print("\nTraining Q-learning...")
    agent, history = train(env, episodes=300, max_steps=400)
    agent.save("q_table.pkl")
    print("Saved q_table.pkl")

    print("\nEvaluating greedy policy...")
    results = evaluate(env, agent, episodes=20, max_steps=400)
    import statistics as st
    print(f"  avg reward         : {st.mean(results['reward']):.1f}")
    print(f"  avg TFR violations : {st.mean(results['tfr_violations']):.2f}")
    print(f"  avg sep violations : {st.mean(results['sep_violations']):.2f}")
    print(f"  reached dest       : {results['reached']}/20")


if __name__ == "__main__":
    main()
