"""
Load aircraft trajectories from FlightRadar24 CSVs.

Each CSV has one flight; columns include UTC, Position ("lat,lon"), Altitude,
Speed. Some flights have ADS-B dead zones (gaps > 60 s) — we linearly
interpolate through them at 30-second steps so every agent advances on the
same clock.
"""

import os
import pandas as pd

STEP_DURATION_SECONDS = 30
GAP_THRESHOLD_SECONDS = 60


def _fill_gaps(df, gap_threshold=GAP_THRESHOLD_SECONDS, step=STEP_DURATION_SECONDS):
    """Linearly interpolate ADS-B dead zones."""
    rows = []
    for i in range(len(df) - 1):
        rows.append(df.iloc[i].to_dict())

        t0 = df["UTC"].iloc[i]
        t1 = df["UTC"].iloc[i + 1]
        gap = (t1 - t0).total_seconds()

        if gap > gap_threshold:
            n_fill = int(gap / step) - 1
            for k in range(1, n_fill + 1):
                frac = (k * step) / gap
                fill = df.iloc[i].to_dict()
                fill["UTC"]      = t0 + pd.Timedelta(seconds=k * step)
                fill["lat"]      = df["lat"].iloc[i]      + frac * (df["lat"].iloc[i + 1]      - df["lat"].iloc[i])
                fill["lon"]      = df["lon"].iloc[i]      + frac * (df["lon"].iloc[i + 1]      - df["lon"].iloc[i])
                fill["Altitude"] = df["Altitude"].iloc[i] + frac * (df["Altitude"].iloc[i + 1] - df["Altitude"].iloc[i])
                fill["Speed"]    = df["Speed"].iloc[i]    + frac * (df["Speed"].iloc[i + 1]    - df["Speed"].iloc[i])
                fill["interpolated"] = True
                rows.append(fill)

    rows.append(df.iloc[-1].to_dict())
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["interpolated"] = out.get("interpolated", False).fillna(False)
    return out


def load_aircraft_data(folder_path, verbose=True):
    """
    Load every CSV in `folder_path` into a dict { callsign: DataFrame }.

    Handles both single-flight files and combined (with a Callsign column).
    """
    data = {}
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(".csv")]
    if verbose:
        print(f"Found {len(files)} CSV file(s) in {folder_path}")

    for fname in files:
        df = pd.read_csv(os.path.join(folder_path, fname))
        df["UTC"] = pd.to_datetime(df["UTC"])
        df[["lat", "lon"]] = df["Position"].str.split(",", expand=True).astype(float)
        df["interpolated"] = False

        if "Callsign" in df.columns:
            for cs, grp in df.groupby("Callsign"):
                cleaned = _fill_gaps(grp.sort_values("UTC").reset_index(drop=True))
                data[cs] = cleaned
        else:
            cs = fname.replace(".csv", "")
            cleaned = _fill_gaps(df.sort_values("UTC").reset_index(drop=True))
            data[cs] = cleaned

        if verbose:
            print(f"  loaded {fname}")

    return data
