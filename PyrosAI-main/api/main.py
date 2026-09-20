"""FastAPI service over the pipeline output.

Reads results.json, falling back to the PostGIS events table if the file is
missing. The payload is cached in module memory and only re-read when the file's
mtime changes, so a dashboard polling these endpoints never pays for repeated
JSON parsing of a multi-megabyte file.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND = os.path.join(ROOT, "frontend")
RESULTS = os.path.join(FRONTEND, "results.json")

app = FastAPI(title="Pyros-AI", version="2.0",
              description="Thermal source detection & classification — PS 26162")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# ---- cached payload ------------------------------------------------------
_CACHE: dict[str, Any] | None = None
_MTIME: float = 0.0


def _from_db() -> dict:
    """Rebuild a payload from PostGIS when results.json is absent."""
    try:
        from pipeline.db_setup import get_dict_conn
        with get_dict_conn() as conn, conn.cursor() as cur:
            cur.execute("""SELECT event_id AS id, latitude AS lat, longitude AS lon,
                                  name, classification, confidence, risk_score,
                                  anomaly_flag, deviation_mult, reason,
                                  frp_history, hour_histogram, dow_histogram,
                                  class_scores, detection_count, active_days,
                                  frp_mean, frp_peak, first_seen, last_seen
                           FROM events ORDER BY risk_score DESC""")
            rows = [dict(r) for r in cur.fetchall()]
    except Exception:                                   # noqa: BLE001
        return {"meta": {"event_count": 0, "source": "unavailable"}, "events": []}

    for i, row in enumerate(rows, start=1):
        row["rank"] = i
        for key in ("first_seen", "last_seen"):
            if row.get(key):
                row[key] = str(row[key])
    return {"meta": {"event_count": len(rows), "source": "postgis",
                     "anomaly_count": sum(1 for r in rows if r["anomaly_flag"])},
            "events": rows}


def data() -> dict:
    """Cached payload; re-read only when results.json actually changes."""
    global _CACHE, _MTIME
    try:
        mtime = os.path.getmtime(RESULTS)
    except OSError:
        if _CACHE is None:
            _CACHE, _MTIME = _from_db(), 0.0
        return _CACHE

    if _CACHE is None or mtime != _MTIME:
        with open(RESULTS, encoding="utf-8") as fh:
            _CACHE = json.load(fh)
        _MTIME = mtime
    return _CACHE


# ---- endpoints -----------------------------------------------------------

@app.get("/health")
def health():
    payload = data()
    return {"status": "ok", "events": len(payload.get("events", [])),
            "source": payload.get("meta", {}).get("source", "results.json")}


@app.get("/api/meta")
def meta():
    return data().get("meta", {})


@app.get("/api/stats")
def stats():
    events = data().get("events", [])
    by_class: dict[str, int] = {}
    for e in events:
        by_class[e["classification"]] = by_class.get(e["classification"], 0) + 1
    anomalies = [e for e in events if e.get("anomaly_flag")]
    return {
        "total_events": len(events),
        "by_classification": by_class,
        "anomalies": len(anomalies),
        "high_risk": sum(1 for e in events if e.get("risk_score", 0) >= 60),
        "top_risk": [{"id": e["id"], "name": e.get("name"),
                      "risk_score": e.get("risk_score"),
                      "classification": e["classification"]}
                     for e in events[:10]],
    }


@app.get("/api/events")
def list_events(
    classification: str | None = Query(None),
    anomaly_only: bool = Query(False),
    min_risk: float = Query(0.0),
    limit: int = Query(1000, ge=1, le=10000),
):
    events = data().get("events", [])
    if classification:
        events = [e for e in events if e["classification"] == classification]
    if anomaly_only:
        events = [e for e in events if e.get("anomaly_flag")]
    if min_risk > 0:
        events = [e for e in events if e.get("risk_score", 0) >= min_risk]
    return {"count": len(events), "events": events[:limit]}


@app.get("/api/events/{event_id}")
def get_event(event_id: str):
    for e in data().get("events", []):
        if e["id"] == event_id:
            return e
    raise HTTPException(status_code=404, detail=f"event {event_id} not found")


def _run_pipeline(fetch: bool) -> None:
    cmd = [sys.executable, os.path.join(ROOT, "run_pipeline.py")]
    if fetch:
        cmd.append("--fetch")
    subprocess.run(cmd, cwd=ROOT, check=False)


@app.post("/api/refresh")
def refresh(background: BackgroundTasks, fetch: bool = Query(True)):
    background.add_task(_run_pipeline, fetch)
    return {"status": "started",
            "detail": "pipeline running in the background; poll /api/meta for generated_at"}


# Mounted last so the API routes above take precedence over the static files.
if os.path.isdir(FRONTEND):
    app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
