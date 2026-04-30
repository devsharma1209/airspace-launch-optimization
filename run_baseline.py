"""
Phase 1-2 deliverable: run the simulation in REPLAY vs rule-based POLICY
mode against real polygon TFRs and save interactive maps.

Usage:
    python run_baseline.py
"""

from src.data_loader import load_aircraft_data
from src.tfr import load_tfrs
from src.model import AirTrafficModel
from src.visualize import plot_matplotlib, plot_plotly, print_summary

DATA_FOLDER = r"Datasets/Validation data (FlightRader24)"
TFR_FILE = r"tfr/TFR_Lat_Lon.xlsx"


def main():
    print("Loading aircraft data...")
    aircraft_data = load_aircraft_data(DATA_FOLDER, verbose=True)
    print(f"Total aircraft: {len(aircraft_data)}")

    print("\nLoading TFR polygons...")
    tfrs = load_tfrs(TFR_FILE)
    for name, t in tfrs.items():
        print(f"  {name}: {len(t.polygon)} vertices, centroid {t.centroid()}")

    max_steps = max(len(df) for df in aircraft_data.values())

    print("\n--- REPLAY baseline ---")
    replay = AirTrafficModel(aircraft_data, list(tfrs.values()), mode="replay")
    replay.run(max_steps - 1)
    print_summary(replay)

    print("\n--- POLICY (rule-based avoidance) ---")
    policy = AirTrafficModel(aircraft_data, list(tfrs.values()), mode="policy")
    policy.run(max_steps - 1)
    print_summary(policy)

    print("\n--- COMPARISON ---")
    rm, pm = replay.metrics(), policy.metrics()
    print(f"  TFR violations: replay={rm['tfr_violations']}  policy={pm['tfr_violations']}")
    print(f"  Sep violations: replay={rm['separation_violations']}  policy={pm['separation_violations']}")

    print("\nGenerating Plotly maps...")
    plot_plotly(replay, "aircraft_map_replay.html", open_browser=False)
    plot_plotly(policy, "aircraft_map_policy.html", open_browser=False)
    print("  -> aircraft_map_replay.html")
    print("  -> aircraft_map_policy.html")


if __name__ == "__main__":
    main()
