"""DBSCAN spatial clustering: hot pixels -> discrete thermal events.

FIRMS reports independent pixels. A refinery flare produces hundreds of them
over a season and a spreading wildfire produces a moving swarm, so alerting on
raw pixels is the noise problem this project exists to solve.

DBSCAN rather than k-means: the number of events is unknown, the shapes are
arbitrary (a plant is a blob, a fire front is a line), and it has a native
notion of noise. eps = 0.045 degrees is roughly 5 km at this latitude.

Noise points are kept rather than discarded. An isolated one-off detection is
exactly what a new industrial incident looks like on its first day, so it
becomes a singleton event and gets judged on its own evidence.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.db_setup import IST_OFFSET_HOURS, get_engine  # noqa: E402

EPS_DEGREES = 0.045        # ~5 km at Andhra Pradesh latitudes
MIN_SAMPLES = 2

DETECTION_QUERY = """
SELECT id, latitude, longitude, bright_ti4, frp, confidence,
       acq_date, acq_time, satellite, instrument, daynight
FROM detections
WHERE acq_date >= %(since)s
ORDER BY acq_date, acq_time
"""


def load_detections(days: int | None = None) -> pd.DataFrame:
    """One query for the whole working set - never a query per event."""
    since = "1900-01-01"
    if days:
        since = (pd.Timestamp.utcnow().normalize()
                 - pd.Timedelta(days=days)).strftime("%Y-%m-%d")

    df = pd.read_sql(DETECTION_QUERY, get_engine(), params={"since": since})
    if df.empty:
        return df

    # FIRMS publishes UTC. Every temporal feature in this project is local:
    # "does this site run at night?" is a question about IST, not Greenwich.
    hhmm = df["acq_time"].astype(str).str.zfill(4)
    utc = pd.to_datetime(df["acq_date"].astype(str) + " " + hhmm,
                         format="%Y-%m-%d %H%M", errors="coerce")
    df = df[utc.notna()].copy()
    utc = utc[utc.notna()]
    df["ts_utc"] = utc.to_numpy()
    df["ts_ist"] = df["ts_utc"] + pd.Timedelta(hours=IST_OFFSET_HOURS)
    df["hour_ist"] = df["ts_ist"].dt.hour
    df["dow_ist"] = df["ts_ist"].dt.dayofweek          # 0 = Monday
    df["date_ist"] = df["ts_ist"].dt.strftime("%Y-%m-%d")
    df["frp"] = pd.to_numeric(df["frp"], errors="coerce").fillna(0.0)
    return df.reset_index(drop=True)


def cluster_detections(df: pd.DataFrame,
                       eps: float = EPS_DEGREES,
                       min_samples: int = MIN_SAMPLES) -> pd.DataFrame:
    """Attach a `cluster` label to every detection (-1 = noise)."""
    if df.empty:
        df["cluster"] = []
        return df
    coords = df[["latitude", "longitude"]].to_numpy(dtype=float)
    df = df.copy()
    df["cluster"] = DBSCAN(eps=eps, min_samples=min_samples,
                           metric="euclidean").fit_predict(coords)
    return df


def build_events(df: pd.DataFrame) -> list[dict]:
    """Clustered detections -> one dict per event, carrying its raw series."""
    if df.empty:
        return []

    events: list[dict] = []

    def pack(event_id: str, grp: pd.DataFrame) -> dict:
        return {
            "event_id": event_id,
            "latitude": float(grp["latitude"].mean()),
            "longitude": float(grp["longitude"].mean()),
            "detection_ids": grp["id"].tolist(),
            "lats": grp["latitude"].to_numpy(dtype=float),
            "lons": grp["longitude"].to_numpy(dtype=float),
            "frp_values": grp["frp"].to_numpy(dtype=float),
            "bright_values": pd.to_numeric(grp["bright_ti4"],
                                           errors="coerce").to_numpy(dtype=float),
            "timestamps": grp["ts_ist"].tolist(),
            "dates": grp["date_ist"].tolist(),
            "hours": grp["hour_ist"].to_numpy(dtype=int),
            "dows": grp["dow_ist"].to_numpy(dtype=int),
            "satellites": sorted({str(s) for s in grp["satellite"].dropna()}),
            "instruments": sorted({str(s) for s in grp["instrument"].dropna()}),
            "daynight": grp["daynight"].tolist(),
            "confidences": grp["confidence"].tolist(),
            "is_noise": event_id.startswith("EVT-NOISE"),
        }

    clustered = df[df["cluster"] >= 0]
    for label, grp in clustered.groupby("cluster", sort=True):
        events.append(pack(f"EVT-{int(label):04d}", grp))

    # Singletons: real detections that simply had no neighbour within 5 km.
    for _, row in df[df["cluster"] < 0].iterrows():
        grp = df.loc[[row.name]]
        events.append(pack(f"EVT-NOISE-{int(row['id'])}", grp))

    return events


def run(days: int | None = None, verbose: bool = True) -> list[dict]:
    df = load_detections(days)
    if df.empty:
        if verbose:
            print("  no detections in the database — run data/fetch_firms.py first")
        return []

    df = cluster_detections(df)
    events = build_events(df)

    if verbose:
        n_noise = int((df["cluster"] < 0).sum())
        n_clustered = len(events) - n_noise
        print(f"  {len(df):,} detections → {len(events):,} events "
              f"({n_clustered:,} clustered, {n_noise:,} singleton)")
        print(f"  eps={EPS_DEGREES}° (~5 km), min_samples={MIN_SAMPLES}")
        sizes = sorted((len(e["detection_ids"]) for e in events), reverse=True)[:5]
        print(f"  largest events: {sizes}")
    return events


if __name__ == "__main__":
    run()
