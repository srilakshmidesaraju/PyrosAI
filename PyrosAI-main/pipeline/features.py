"""Behavioural feature extraction.

The premise of the whole system: *what* a thermal source is shows up in *how it
behaves over time*, not in how hot it is at one instant. A flare and a wildfire
can both read 40 MW; only one of them is still burning at 02:00 next Tuesday.

Four groups:
  temporal  persistence, duty cycle, hour-of-day and day-of-week rhythm
  intensity FRP central tendency and variability
  spatial   footprint, spread rate, distance to industry, neighbour density
  context   nearest OSM facility and its risk weight

One deliberate choice: FRP variability is computed twice. `frp_variance` is the
plain variance and `frp_robust_var` is the IQR-over-mean. A single 5x excursion
inflates the plain figure enough to make a steady factory look like a wildfire,
so classification reads the robust one and anomaly detection reads the raw one.
The spike must not be allowed to rewrite the identity of the site it happened at.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.db_setup import get_conn  # noqa: E402

EARTH_RADIUS_KM = 6371.0088
NIGHT_HOURS = set(range(0, 7)) | set(range(20, 24))   # 20:00-06:59 IST
NEIGHBOUR_RADIUS_KM = 5.0
NEAR_INDUSTRIAL_M = 1000.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# --------------------------------------------------------------------------
# spatial context - one batched query, not one per event
# --------------------------------------------------------------------------

NEAREST_SQL = """
WITH pts(idx, lon, lat) AS (VALUES %s)
SELECT p.idx,
       f.name,
       f.facility_type,
       f.risk_weight,
       ST_Distance(
           f.geom::geography,
           ST_SetSRID(ST_MakePoint(p.lon, p.lat), 4326)::geography
       ) AS dist_m
FROM pts p
CROSS JOIN LATERAL (
    SELECT name, facility_type, risk_weight, geom
    FROM facilities
    ORDER BY geom <-> ST_SetSRID(ST_MakePoint(p.lon, p.lat), 4326)
    LIMIT 1
) f
"""


def nearest_facilities(points: list[tuple[float, float]]) -> dict[int, dict]:
    """Nearest OSM facility for every event centroid, in a single round trip.

    CROSS JOIN LATERAL with the `<->` KNN operator lets PostGIS walk the GiST
    index once per point inside one query, instead of the pipeline issuing
    hundreds of separate round trips.
    """
    if not points:
        return {}
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM facilities")
        if cur.fetchone()[0] == 0:
            return {}                       # no OSM data loaded yet

        values = ",".join(
            cur.mogrify("(%s,%s,%s)", (i, lon, lat)).decode()
            for i, (lat, lon) in enumerate(points)
        )
        cur.execute(NEAREST_SQL % values)
        return {
            row[0]: {"facility_name": row[1], "facility_type": row[2],
                     "facility_risk_weight": float(row[3] or 1.0),
                     "dist_to_facility_m": float(row[4])}
            for row in cur.fetchall()
        }


# --------------------------------------------------------------------------
# per-event features
# --------------------------------------------------------------------------

def _spread(event: dict, clat: float, clon: float) -> tuple[float, float]:
    """(spread_rate km/day, footprint km).

    Spread is measured as the growth of the *daily* footprint, not the
    cumulative one: a static plant scatters its pixels the same ~1 km every day
    (slope 0), while a fire front's daily extent widens.
    """
    lats, lons = event["lats"], event["lons"]
    footprint = float(haversine_km(clat, clon, lats, lons).max()) if len(lats) else 0.0

    dates = pd.to_datetime(pd.Series(event["dates"]))
    if dates.nunique() < 2:
        return 0.0, footprint

    day_index = (dates - dates.min()).dt.days.to_numpy()
    extents, days = [], []
    for day in np.unique(day_index):
        mask = day_index == day
        if mask.sum() < 2:
            extents.append(0.0)
        else:
            dlat, dlon = lats[mask].mean(), lons[mask].mean()
            extents.append(float(haversine_km(dlat, dlon, lats[mask], lons[mask]).max()))
        days.append(float(day))

    days_arr, ext_arr = np.array(days), np.array(extents)
    if np.ptp(days_arr) == 0:
        return 0.0, footprint
    slope = float(np.polyfit(days_arr, ext_arr, 1)[0])
    return slope, footprint


def extract_one(event: dict, window_days: int) -> dict:
    frp = np.asarray(event["frp_values"], dtype=float)
    hours = np.asarray(event["hours"], dtype=int)
    dows = np.asarray(event["dows"], dtype=int)
    dates = pd.to_datetime(pd.Series(event["dates"]))

    active_days = int(dates.dt.normalize().nunique())
    date_range = int((dates.max() - dates.min()).days) + 1
    persistence = active_days / date_range if date_range else 0.0

    q25, q75 = (np.percentile(frp, [25, 75]) if len(frp) else (0.0, 0.0))
    mean_frp = float(frp.mean()) if len(frp) else 0.0
    robust_var = float((q75 - q25) / mean_frp) if mean_frp > 0 else 0.0

    hour_hist = np.bincount(hours, minlength=24).astype(float)
    dow_hist = np.bincount(dows, minlength=7).astype(float)
    n = max(len(hours), 1)

    clat, clon = event["latitude"], event["longitude"]
    spread_rate, footprint = _spread(event, clat, clon)

    return {
        **event,
        "detection_count": int(len(frp)),
        "active_days": active_days,
        "date_range": date_range,
        "persistence": round(persistence, 4),
        "duty_cycle": round(active_days / window_days, 4) if window_days else 0.0,

        "frp_mean": round(mean_frp, 3),
        "frp_peak": round(float(frp.max()), 3) if len(frp) else 0.0,
        "frp_min": round(float(frp.min()), 3) if len(frp) else 0.0,
        "frp_variance": round(float(frp.var(ddof=0)), 4) if len(frp) else 0.0,
        "frp_robust_var": round(robust_var, 4),

        "hour_histogram": (hour_hist / n).round(4).tolist(),
        "dow_histogram": (dow_hist / n).round(4).tolist(),
        "hour_counts": hour_hist.astype(int).tolist(),
        "dow_counts": dow_hist.astype(int).tolist(),
        "night_share": round(float(np.isin(hours, list(NIGHT_HOURS)).mean()), 4),
        "weekend_share": round(float((dows >= 5).mean()), 4),

        "spread_rate": round(spread_rate, 4),
        "footprint_km": round(footprint, 3),

        "first_seen": dates.min().strftime("%Y-%m-%d %H:%M"),
        "last_seen": dates.max().strftime("%Y-%m-%d %H:%M"),
        "frp_history": [round(float(v), 2) for v in frp],
        "frp_dates": [str(d) for d in event["dates"]],
    }


def run(events: list[dict], window_days: int = 60,
        verbose: bool = True) -> list[dict]:
    if not events:
        return []

    out = [extract_one(e, window_days) for e in events]

    # --- spatial context: nearest facility, batched ------------------------
    points = [(e["latitude"], e["longitude"]) for e in out]
    nearest = nearest_facilities(points)
    for i, event in enumerate(out):
        info = nearest.get(i, {"facility_name": None, "facility_type": None,
                               "facility_risk_weight": 1.0,
                               "dist_to_facility_m": 999999.0})
        event.update(info)
        event["near_industrial"] = bool(info["dist_to_facility_m"] < NEAR_INDUSTRIAL_M)

    # --- neighbour density: vectorised in memory, no DB round trip ---------
    lats = np.array([e["latitude"] for e in out])
    lons = np.array([e["longitude"] for e in out])
    for i, event in enumerate(out):
        d = haversine_km(lats[i], lons[i], lats, lons)
        d[i] = np.inf
        event["neighbour_count"] = int((d <= NEIGHBOUR_RADIUS_KM).sum())
        event["nearest_event_km"] = round(float(d.min()), 3) if len(d) > 1 else 999.0

    if verbose:
        near = sum(1 for e in out if e["near_industrial"])
        med = float(np.median([e["dist_to_facility_m"] for e in out]))
        print(f"  {len(out):,} events featurised "
              f"({len(extract_one(events[0], window_days))} fields each)")
        print(f"  {near:,} within {NEAR_INDUSTRIAL_M:.0f} m of a mapped facility; "
              f"median distance {med / 1000:.2f} km")
    return out


if __name__ == "__main__":
    from pipeline.cluster import run as cluster_run
    run(cluster_run())
