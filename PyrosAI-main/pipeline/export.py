"""Risk scoring, PostGIS persistence, and the frontend payload.

Risk blends four things a duty officer actually cares about: what the thing is,
whether it is behaving abnormally, how sure we are, and how dangerous its
surroundings are.

    score = severity*40 + anomaly*30 + confidence*15 + facility_weight*15
    if anomaly: score = min(99, score + min(deviation/10, 1)*10)

Two files are written, with identical content:

  frontend/results.json     canonical, for the API and anything else
  frontend/results.data.js  `window.PYROS_DATA = {...}` for file:// use

The second exists because browsers block fetch() against file:// URLs under the
same-origin policy, so a double-clicked index.html cannot read a local JSON
file - but it can load a script. That is what makes the offline demo work.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
from psycopg2.extras import execute_values

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.classify import (AGRICULTURAL, GAS_FLARE, INDUSTRIAL,  # noqa: E402
                               INSUFFICIENT, WILDFIRE)
from pipeline.db_setup import REGION_NAME, get_conn  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND = os.path.join(ROOT, "frontend")
RESULTS_JSON = os.path.join(FRONTEND, "results.json")
RESULTS_JS = os.path.join(FRONTEND, "results.data.js")

SEVERITY = {
    INDUSTRIAL: 1.0,
    WILDFIRE: 0.9,
    GAS_FLARE: 0.4,
    AGRICULTURAL: 0.3,
    INSUFFICIENT: 0.1,
}

# Charts only ever draw the tail, and the payload has to stay small enough to
# parse in well under a second from file://.
FRP_HISTORY_LIMIT = 90


def compute_risk(event: dict) -> float:
    severity = SEVERITY.get(event["classification"], 0.1)
    anomaly = 1.0 if event["anomaly_flag"] else 0.0
    weight = min(float(event.get("facility_risk_weight", 1.0)) / 5.0, 1.0)

    score = (severity * 40.0
             + anomaly * 30.0
             + float(event["confidence"]) * 15.0
             + weight * 15.0)

    if event["anomaly_flag"]:
        bump = min(float(event.get("deviation_mult", 1.0)) / 10.0, 1.0) * 10.0
        score = min(99.0, score + bump)

    return round(float(score), 1)


def event_name(event: dict) -> str:
    """Readable label: the facility if we are on one, coordinates otherwise."""
    facility = event.get("facility_name")
    dist = event.get("dist_to_facility_m", 1e9)
    if facility and dist < 2000:
        return f"{facility} ({dist:.0f} m)"
    return f"{event['classification']} · {event['latitude']:.3f}, {event['longitude']:.3f}"


def _clean(value):
    """numpy/pandas scalars -> JSON-safe primitives, NaN -> None."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else round(value, 4)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_clean(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    return value


def build_payload(events: list[dict], detection_count: int,
                  window_days: int) -> dict:
    for event in events:
        event["risk_score"] = compute_risk(event)
        event["name"] = event_name(event)

    events.sort(key=lambda e: e["risk_score"], reverse=True)

    out = []
    for rank, e in enumerate(events, start=1):
        history = e["frp_history"][-FRP_HISTORY_LIMIT:]
        dates = e["frp_dates"][-FRP_HISTORY_LIMIT:]
        out.append(_clean({
            "id": e["event_id"],
            "rank": rank,
            "lat": e["latitude"],
            "lon": e["longitude"],
            "name": e["name"],

            "classification": e["classification"],
            "confidence": e["confidence"],
            "risk_score": e["risk_score"],
            "type_color": e["type_color"],
            "class_scores": e["class_scores"],
            "margin": e["margin"],
            "reason": e["reason"],
            "evidence": e["evidence"],

            "anomaly_flag": e["anomaly_flag"],
            "deviation_mult": e["deviation_mult"],
            "z_score": e["z_score"],
            "iso_score": e["iso_score"],
            "baseline_mean": e["baseline_mean"],
            "baseline_std": e["baseline_std"],
            "baseline_established": e["baseline_established"],
            "anomaly_desc": e["anomaly_desc"],
            "offhours": e["offhours"],
            "active_hours": e.get("active_hours", []),

            "frp_history": history,
            "frp_dates": dates,
            "hour_histogram": e["hour_histogram"],
            "dow_histogram": e["dow_histogram"],

            "active_days": e["active_days"],
            "date_range": e["date_range"],
            "persistence": e["persistence"],
            "duty_cycle": e["duty_cycle"],
            "frp_mean": e["frp_mean"],
            "frp_peak": e["frp_peak"],
            "frp_variance": e["frp_variance"],
            "frp_robust_var": e["frp_robust_var"],
            "detection_count": e["detection_count"],
            "spread_rate": e["spread_rate"],
            "footprint_km": e["footprint_km"],
            "night_share": e["night_share"],
            "weekend_share": e["weekend_share"],
            "neighbour_count": e.get("neighbour_count", 0),

            "dist_to_facility_m": e["dist_to_facility_m"],
            "facility_name": e.get("facility_name"),
            "facility_type": e.get("facility_type"),
            "facility_risk_weight": e.get("facility_risk_weight", 1.0),

            # World Bank flare registry cross-check (pipeline/validate_flares.py).
            "wb_match": e.get("wb_match"),
            "flare_validated": e.get("flare_validated"),

            "sensors": sorted(set(e.get("instruments", []))) or ["VIIRS"],
            "satellites": e.get("satellites", []),
            "first_seen": e["first_seen"],
            "last_seen": e["last_seen"],

            # Populated by vision/cnn_verify.py when --vision is used.
            "cnn_status": e.get("cnn_status"),
            "cnn_result": e.get("cnn_result"),
            "cnn_confidence": e.get("cnn_confidence"),
            "cnn_context": e.get("cnn_context"),
            "cnn_image": e.get("cnn_image"),
        }))

    dates = [e["first_seen"][:10] for e in events if e.get("first_seen")]
    counts: dict[str, int] = {}
    for e in events:
        counts[e["classification"]] = counts.get(e["classification"], 0) + 1

    return {
        "meta": {
            "project": "Pyros-AI — PS 26162 (NTRO)",
            "region": REGION_NAME,
            "event_count": len(out),
            "detection_count": detection_count,
            "anomaly_count": sum(1 for e in out if e["anomaly_flag"]),
            "window_start": min(dates) if dates else None,
            "window_end": max(dates) if dates else None,
            "window_days": window_days,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "phase": "Phase 2 — real FIRMS data, rule-based classification",
            "timezone": "IST (UTC+05:30)",
            "class_counts": counts,
            "flares_validated": sum(1 for e in out if e.get("flare_validated") is True),
            "flares_unvalidated": sum(1 for e in out if e.get("flare_validated") is False),
            "data_source": "NASA FIRMS (VIIRS S-NPP/NOAA-20/NOAA-21 + MODIS) · OSM facilities",
        },
        "events": out,
    }


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------

EVENT_INSERT = """
INSERT INTO events (
    event_id, geom, latitude, longitude, name, first_seen, last_seen,
    detection_count, active_days, frp_mean, frp_peak, frp_variance,
    spread_rate, footprint_km, persistence, night_share, weekend_share,
    classification, confidence, anomaly_flag, deviation_mult, risk_score,
    reason, frp_history, hour_histogram, dow_histogram, class_scores
) VALUES %s
"""
EVENT_TEMPLATE = ("(%s, ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s, %s, %s, %s, %s, "
                  "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                  "%s, %s, %s, %s)")


def save_to_db(payload: dict) -> int:
    rows = []
    for e in payload["events"]:
        rows.append((
            e["id"], e["lon"], e["lat"], e["lat"], e["lon"], e["name"],
            e["first_seen"], e["last_seen"],
            e["detection_count"], e["active_days"], e["frp_mean"], e["frp_peak"],
            e["frp_variance"], e["spread_rate"], e["footprint_km"],
            e["persistence"], e["night_share"], e["weekend_share"],
            e["classification"], e["confidence"], e["anomaly_flag"],
            e["deviation_mult"], e["risk_score"], e["reason"],
            json.dumps(e["frp_history"]), json.dumps(e["hour_histogram"]),
            json.dumps(e["dow_histogram"]), json.dumps(e["class_scores"]),
        ))
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE events RESTART IDENTITY")
        execute_values(cur, EVENT_INSERT, rows, template=EVENT_TEMPLATE, page_size=500)
        conn.commit()
    return len(rows)


def write_files(payload: dict) -> tuple[int, int]:
    os.makedirs(FRONTEND, exist_ok=True)
    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

    with open(RESULTS_JSON, "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(RESULTS_JS, "w", encoding="utf-8") as fh:
        fh.write("// Generated by pipeline/export.py — do not edit.\n"
                 "// Mirrors results.json so the dashboard runs from file:// with no server.\n"
                 f"window.PYROS_DATA = {text};\n")

    return os.path.getsize(RESULTS_JSON), os.path.getsize(RESULTS_JS)


def run(events: list[dict], detection_count: int = 0, window_days: int = 60,
        verbose: bool = True) -> dict:
    payload = build_payload(events, detection_count, window_days)
    n = save_to_db(payload)
    size_json, size_js = write_files(payload)

    if verbose:
        print(f"  {n:,} events written to PostGIS")
        print(f"  results.json      {size_json / 1024:.0f} KB")
        print(f"  results.data.js   {size_js / 1024:.0f} KB")
        print("\n  highest risk:")
        for e in payload["events"][:5]:
            # Same reasoning as the dashboard badge: a 1.0x "deviation" from an
            # isolation-forest-only flag would misrepresent a population
            # comparison as a departure from the site's own baseline.
            flag = ""
            if e["anomaly_flag"]:
                flag = (f"  ANOMALY {e['deviation_mult']:.1f}x"
                        if e["deviation_mult"] >= 1.5
                        else f"  OUTLIER iso={e['iso_score']:.2f}")
            print(f"    {e['rank']:>3}. {e['risk_score']:>5.1f}  {e['classification']:<22}"
                  f" {e['name'][:44]}{flag}")
    return payload


if __name__ == "__main__":
    from pipeline.anomaly import run as anomaly_run
    from pipeline.classify import run as classify_run
    from pipeline.cluster import load_detections
    from pipeline.cluster import run as cluster_run
    from pipeline.features import run as features_run

    evs = cluster_run()
    n_det = len(load_detections())
    run(anomaly_run(classify_run(features_run(evs))), detection_count=n_det)
