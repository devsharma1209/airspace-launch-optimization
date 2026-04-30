"""
Temporary Flight Restriction zones (polygon version).

Replaces the circular TFR from the original prototype with real polygon
boundaries loaded from `tfr/TFR_Lat_Lon.xlsx`. The file has one sheet per
TFR (TFR1 through TFR5); each sheet has both DMS and decimal columns.
"""

import pandas as pd

from .geometry import (
    dms_to_decimal,
    order_polygon_ccw,
    point_in_polygon,
    signed_distance_to_polygon,
)


class PolygonTFR:
    """A single TFR polygon."""

    def __init__(self, name, polygon):
        self.name = name
        self.polygon = list(polygon)  # list of (lat, lon)

    def contains(self, lat, lon):
        """True if (lat, lon) is inside the TFR (a violation)."""
        return point_in_polygon(lat, lon, self.polygon)

    def distance_to_edge(self, lat, lon):
        """
        Signed distance (nm) to the TFR boundary.
        Negative = inside (violation), positive = outside (safe).
        """
        return signed_distance_to_polygon(lat, lon, self.polygon)

    def centroid(self):
        lats = [p[0] for p in self.polygon]
        lons = [p[1] for p in self.polygon]
        return sum(lats) / len(lats), sum(lons) / len(lons)


def _sheet_to_polygon(df, name):
    """Convert one Excel sheet into an ordered list of (lat, lon)."""
    df = df.rename(columns=lambda c: c.strip())  # clean stray whitespace

    if "Lat_dd" in df.columns and "Lon_dd" in df.columns:
        points = list(zip(df["Lat_dd"].astype(float), df["Lon_dd"].astype(float)))
    else:
        # Fall back to converting DMS ourselves
        points = [
            (dms_to_decimal(r["Lat_dms"]), dms_to_decimal(r["Lon_dms"]))
            for _, r in df.iterrows()
        ]

    return PolygonTFR(name, order_polygon_ccw(points))


def load_tfrs(xlsx_path):
    """
    Load every sheet of the TFR workbook into a dict { name: PolygonTFR }.
    """
    xl = pd.ExcelFile(xlsx_path)
    tfrs = {}
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        tfrs[sheet] = _sheet_to_polygon(df, sheet)
    return tfrs
