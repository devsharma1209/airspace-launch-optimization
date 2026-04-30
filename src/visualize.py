"""
2D matplotlib preview + interactive Plotly map.

Both draw the POLYGON TFRs (not a circle) and the actual simulated paths
from each agent's history arrays.
"""

import os
import webbrowser
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


COLOURS = [
    "#00c8e0", "#ff9d00", "#00ff9d", "#ff4488",
    "#bf55ff", "#ffcc00", "#00ffcc", "#ff6600",
    "#66ccff", "#ff66cc", "#ccff66",
]


# ---------------------------------------------------------------------------
# Matplotlib 2D
# ---------------------------------------------------------------------------

def plot_matplotlib(model, title_suffix=""):
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_facecolor("#0d1b2a")

    # Flight paths
    for ac in model.aircraft_agents:
        ax.plot(ac.history_lon, ac.history_lat, linewidth=1.5, label=ac.unique_id)
        ax.scatter(ac.history_lon[0], ac.history_lat[0], color="lime", s=30, zorder=5)
        ax.scatter(ac.history_lon[-1], ac.history_lat[-1], color="red", s=30, zorder=5)

    # TFR polygons
    for tfr in model.tfrs:
        poly = np.array(tfr.polygon + [tfr.polygon[0]])
        ax.fill(poly[:, 1], poly[:, 0], color="red", alpha=0.15)
        ax.plot(poly[:, 1], poly[:, 0], color="red", linestyle="--", linewidth=1.0)
        c_lat, c_lon = tfr.centroid()
        ax.text(c_lon, c_lat, tfr.name, color="red", fontsize=8)

    ax.set_xlabel("Longitude", color="#c8eaf5")
    ax.set_ylabel("Latitude", color="#c8eaf5")
    ax.set_title(f"Simulation {model.mode} {title_suffix}".strip(), color="#00c8e0")
    ax.tick_params(colors="#c8eaf5")
    ax.grid(True, alpha=0.2)
    ax.legend(facecolor="#0d1b2a", labelcolor="#c8eaf5", fontsize=6, loc="best")
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a3a5c")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Plotly interactive
# ---------------------------------------------------------------------------

def plot_plotly(model, output_html=None, open_browser=True):
    fig = go.Figure()

    all_lats, all_lons = [], []
    for ac in model.aircraft_agents:
        all_lats.extend(ac.history_lat)
        all_lons.extend(ac.history_lon)
    for tfr in model.tfrs:
        for lat, lon in tfr.polygon:
            all_lats.append(lat)
            all_lons.append(lon)

    mid_lat = (max(all_lats) + min(all_lats)) / 2
    mid_lon = (max(all_lons) + min(all_lons)) / 2

    # Aircraft tracks
    for i, ac in enumerate(model.aircraft_agents):
        col = COLOURS[i % len(COLOURS)]
        fig.add_trace(go.Scattergeo(
            lat=ac.history_lat, lon=ac.history_lon,
            mode="lines+markers",
            line=dict(width=1.5, color=col),
            marker=dict(size=3, color=col),
            name=ac.unique_id,
            text=[f"{ac.unique_id} t={t}" for t in ac.history_time],
            hoverinfo="text",
        ))

    # TFR polygons
    for tfr in model.tfrs:
        poly = tfr.polygon + [tfr.polygon[0]]
        lats = [p[0] for p in poly]
        lons = [p[1] for p in poly]
        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons,
            mode="lines",
            line=dict(color="rgba(255,60,60,0.9)", width=2, dash="dash"),
            fill="toself",
            fillcolor="rgba(255,60,60,0.12)",
            name=f"TFR {tfr.name}",
        ))

    m = model.metrics()
    fig.update_geos(
        projection_type="mercator",
        center=dict(lat=mid_lat, lon=mid_lon),
        showland=True, landcolor="#0d1b2a",
        showocean=True, oceancolor="#060d1a",
        showcountries=True, countrycolor="rgba(60,100,160,0.5)",
        showcoastlines=True, coastlinecolor="rgba(0,140,200,0.7)",
        lonaxis=dict(range=[min(all_lons) - 3, max(all_lons) + 3]),
        lataxis=dict(range=[min(all_lats) - 2, max(all_lats) + 2]),
        bgcolor="#030810",
    )
    fig.update_layout(
        title=dict(
            text=f"Simulation - {m['mode'].upper()} | "
                 f"TFR violations: {m['tfr_violations']} | "
                 f"Sep violations: {m['separation_violations']}",
            font=dict(color="#00c8e0"),
        ),
        paper_bgcolor="#030810",
        legend=dict(font=dict(color="#c8eaf5"),
                    bgcolor="rgba(4,12,26,0.8)"),
        margin=dict(l=0, r=0, t=50, b=0),
        height=780,
    )

    if output_html is None:
        output_html = f"aircraft_map_{model.mode}.html"
    fig.write_html(output_html, config={"scrollZoom": True})
    if open_browser:
        webbrowser.open("file://" + os.path.abspath(output_html))
    return fig


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(model):
    m = model.metrics()
    print()
    print("=" * 55)
    print(f"  SIMULATION COMPLETE - {m['mode'].upper()} MODE")
    print("=" * 55)
    print(f"  Aircraft              : {m['aircraft_count']}")
    print(f"  Steps run             : {m['steps']}")
    print(f"  TFR violations        : {m['tfr_violations']}")
    print(f"  Separation violations : {m['separation_violations']}")
    print("=" * 55)

    if model.violations:
        print("\n  TFR violations:")
        for cs, name, t in model.violations:
            print(f"    {cs} entered {name} at step {t}")

    if model.separation_violations:
        print("\n  Separation losses:")
        for cs1, cs2, t, d in model.separation_violations:
            print(f"    {cs1} <-> {cs2} at step {t} ({d} nm)")
