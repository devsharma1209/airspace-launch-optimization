"""
Geometry helpers — all the math used across the project.

Everything here is plain numpy + math so it is easy to read and fast enough
for the ~20 aircraft we simulate.

Units convention:
    - latitude / longitude : decimal degrees
    - distances            : nautical miles (nm)
    - bearings / headings  : compass degrees (0 = North, 90 = East)
"""

import math
import re
import numpy as np

EARTH_RADIUS_NM = 3440.0  # mean Earth radius in nautical miles


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def dms_to_decimal(dms_str):
    """
    Convert a DDMMSS coordinate string (with hemisphere letter) to decimal
    degrees.

    Examples
    --------
    "255500N"  ->   25.9166...
    "972200W"  ->  -97.3666...
    "180000N"  ->   18.0
    """
    s = str(dms_str).strip().upper()

    # Pull out the hemisphere letter
    hemi = s[-1]
    digits = re.sub(r"[^0-9.]", "", s[:-1])

    # Expect 6 digits: DDMMSS  (sometimes 7 for longitudes like 1030000W)
    if len(digits) < 6:
        raise ValueError(f"Bad DMS string: {dms_str!r}")

    seconds = float(digits[-2:])
    minutes = float(digits[-4:-2])
    degrees = float(digits[:-4])

    dd = degrees + minutes / 60.0 + seconds / 3600.0
    if hemi in ("S", "W"):
        dd = -dd
    return dd


# ---------------------------------------------------------------------------
# Great-circle math
# ---------------------------------------------------------------------------

def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points in nautical miles."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_NM * np.arcsin(np.sqrt(a))


def bearing_between(lat1, lon1, lat2, lon2):
    """Compass bearing (0-360) from point 1 to point 2."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def move_by_heading(lat, lon, heading_deg, speed_kts, time_seconds):
    """Dead-reckon a new lat/lon after flying at (heading, speed) for time_seconds."""
    distance_nm = speed_kts * (time_seconds / 3600.0)
    ang = distance_nm / EARTH_RADIUS_NM

    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    hdg_r = math.radians(heading_deg)

    new_lat = math.asin(
        math.sin(lat_r) * math.cos(ang)
        + math.cos(lat_r) * math.sin(ang) * math.cos(hdg_r)
    )
    new_lon = lon_r + math.atan2(
        math.sin(hdg_r) * math.sin(ang) * math.cos(lat_r),
        math.cos(ang) - math.sin(lat_r) * math.sin(new_lat),
    )
    return math.degrees(new_lat), math.degrees(new_lon)


# ---------------------------------------------------------------------------
# Polygon geometry (used by TFR zones)
# ---------------------------------------------------------------------------

def order_polygon_ccw(points):
    """
    Sort polygon vertices counter-clockwise around their centroid.

    The Excel TFR sheets list points in a scrambled order, so we reorder them
    here to get a clean closed polygon.

    Parameters
    ----------
    points : list of (lat, lon)

    Returns
    -------
    list of (lat, lon) sorted CCW.
    """
    pts = np.array(points, dtype=float)
    cx = pts[:, 1].mean()
    cy = pts[:, 0].mean()
    angles = np.arctan2(pts[:, 0] - cy, pts[:, 1] - cx)
    order = np.argsort(angles)
    return [tuple(pts[i]) for i in order]


def point_in_polygon(lat, lon, polygon):
    """
    Ray-casting point-in-polygon test.

    polygon : list of (lat, lon), assumed closed-polygon order (CCW or CW).

    Returns True if (lat, lon) is inside.
    """
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        yi, xi = polygon[i]     # (lat, lon)
        yj, xj = polygon[j]
        # check if the horizontal ray from (lon, lat) crosses edge (i, j)
        if ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def _distance_point_to_segment_nm(lat, lon, lat1, lon1, lat2, lon2):
    """
    Approximate distance (nm) from a point to a line segment defined by two
    lat/lon endpoints. We use a local equirectangular projection — fine for
    the short edge lengths of a TFR.
    """
    # project to local nm coordinates (y = lat, x = lon scaled by cos lat)
    mean_lat_rad = math.radians((lat1 + lat2) / 2.0)
    scale_x = 60.0 * math.cos(mean_lat_rad)   # nm per degree longitude
    scale_y = 60.0                             # nm per degree latitude

    px = lon * scale_x
    py = lat * scale_y
    ax = lon1 * scale_x
    ay = lat1 * scale_y
    bx = lon2 * scale_x
    by = lat2 * scale_y

    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-9:
        return math.hypot(px - ax, py - ay)

    # parameter of nearest point on segment (clamped 0..1)
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    nx = ax + t * dx
    ny = ay + t * dy
    return math.hypot(px - nx, py - ny)


def signed_distance_to_polygon(lat, lon, polygon):
    """
    Signed distance (nm) from (lat, lon) to the polygon boundary.

    Negative  = inside  (a violation, if polygon is a TFR)
    Positive  = outside

    Value = distance to nearest edge.
    """
    n = len(polygon)
    nearest = float("inf")
    for i in range(n):
        lat1, lon1 = polygon[i]
        lat2, lon2 = polygon[(i + 1) % n]
        d = _distance_point_to_segment_nm(lat, lon, lat1, lon1, lat2, lon2)
        if d < nearest:
            nearest = d
    inside = point_in_polygon(lat, lon, polygon)
    return -nearest if inside else nearest
