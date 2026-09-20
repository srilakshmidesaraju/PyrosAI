#!/usr/bin/env python3
"""Pyros-AI — end-to-end pipeline runner.

    python run_pipeline.py                      # re-analyse what is already in the DB
    python run_pipeline.py --fetch              # pull fresh FIRMS data first
    python run_pipeline.py --fetch --osm        # also refresh OSM facilities
    python run_pipeline.py --fetch --vision     # add CNN imagery verification

Every stage is an importable module and can be run on its own
(`python pipeline/features.py`) to inspect that stage in isolation.
"""

from __future__ import annotations

import argparse
import sys
import time

BANNER = "=" * 72


def stage(n: str, title: str) -> float:
    print(f"\n{BANNER}\n  [{n}]  {title}\n{BANNER}")
    return time.perf_counter()


def done(t0: float) -> None:
    print(f"  ── {time.perf_counter() - t0:.1f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true", help="fetch fresh FIRMS detections")
    ap.add_argument("--historical", action="store_true",
                    help="fetch including the SP archive sources (deeper baselines)")
    ap.add_argument("--osm", action="store_true", help="refresh OSM facilities")
    ap.add_argument("--vision", action="store_true", help="run CNN imagery verification")
    ap.add_argument("--validate-flares", action="store_true",
                    help="(re)load the World Bank flare registry before validating")
    ap.add_argument("--days", type=int, default=60, help="window length (default 60)")
    args = ap.parse_args()

    overall = time.perf_counter()
    print(f"\n{BANNER}\n  PYROS-AI  ·  PS 26162  ·  thermal source detection & classification"
          f"\n{BANNER}")

    # -- 0 --------------------------------------------------------------
    t = stage("0/6", "DATABASE")
    from pipeline.db_setup import create_schema, table_counts
    create_schema(verbose=False)
    print("  schema ready:", ", ".join(f"{k}={v:,}" for k, v in table_counts().items()))
    done(t)

    # -- 1 --------------------------------------------------------------
    if args.fetch or args.historical:
        t = stage("1/6", "FETCH — NASA FIRMS")
        from data.fetch_firms import fetch
        fetch(days=args.days, historical=args.historical)
        done(t)
    else:
        print("\n  [1/6]  FETCH skipped (use --fetch to pull fresh detections)")

    if args.osm:
        t = stage("1b", "FETCH — OpenStreetMap facilities")
        from data.fetch_osm import fetch as fetch_osm
        fetch_osm()
        done(t)

    # -- 2 --------------------------------------------------------------
    t = stage("2/6", "CLUSTER — DBSCAN")
    from pipeline.cluster import load_detections
    from pipeline.cluster import run as cluster_run
    events = cluster_run()
    detection_count = len(load_detections())
    done(t)

    if not events:
        print("\nNo events. Run with --fetch to pull detections first.")
        return 1

    # -- 3 --------------------------------------------------------------
    t = stage("3/6", "FEATURES — temporal, intensity, spatial")
    from pipeline.features import run as features_run
    events = features_run(events, window_days=args.days)
    done(t)

    # -- 4 --------------------------------------------------------------
    t = stage("4/6", "CLASSIFY — rule-based scoring")
    from pipeline.classify import run as classify_run
    events = classify_run(events)
    done(t)

    # -- 5 --------------------------------------------------------------
    t = stage("5/6", "ANOMALY — per-site baselines + isolation forest")
    from pipeline.anomaly import run as anomaly_run
    events = anomaly_run(events)
    done(t)

    if args.vision:
        t = stage("5b", "VISION — CNN land-cover verification")
        from pipeline.export import compute_risk
        for e in events:                      # risk gates which events get imagery
            e["risk_score"] = compute_risk(e)
        from vision.cnn_verify import run as vision_run
        events = vision_run(events)
        done(t)

    # -- 5c -------------------------------------------------------------
    # Independent cross-check of our Gas Flare calls against a documented
    # registry. Wrapped so a missing spreadsheet or an unloaded table degrades
    # to "unvalidated" rather than taking the whole run down.
    t = stage("5c", "VALIDATE — World Bank flare registry")
    try:
        from pipeline.validate_flares import annotate_events, load_reference
        from pipeline.db_setup import get_conn as _vf_conn

        if args.validate_flares:
            load_reference()

        conn = _vf_conn()                 # one connection for the whole loop
        try:
            events = annotate_events(events, conn)
        finally:
            conn.close()

        matched = sum(1 for e in events if e.get("wb_match"))
        ok = sum(1 for e in events if e.get("flare_validated") is True)
        no = sum(1 for e in events if e.get("flare_validated") is False)
        rate = ok / (ok + no) if (ok + no) else 0.0
        print(f"  {matched:,} events within 2 km of a documented flare site")
        print(f"  gas flare calls: {ok:,} confirmed, {no:,} unlisted "
              f"({rate:.1%} agreement)")
    except Exception as err:                                  # noqa: BLE001
        print(f"  flare validation skipped: {type(err).__name__}: {err}")
        print("  (run: python pipeline/validate_flares.py  to load the registry)")
        for ev in events:
            ev.setdefault("wb_match", None)
            ev.setdefault("flare_validated", None)
    done(t)

    # -- 6 --------------------------------------------------------------
    t = stage("6/6", "EXPORT — risk scoring, PostGIS, frontend payload")
    from pipeline.export import run as export_run
    payload = export_run(events, detection_count=detection_count,
                         window_days=args.days)
    done(t)

    meta = payload["meta"]
    print(f"\n{BANNER}\n  SUMMARY\n{BANNER}")
    print(f"  region        : {meta['region']}")
    print(f"  window        : {meta['window_start']} → {meta['window_end']}")
    print(f"  detections    : {meta['detection_count']:,}")
    print(f"  events        : {meta['event_count']:,}")
    print(f"  anomalies     : {meta['anomaly_count']:,}")
    for name, n in sorted(meta["class_counts"].items(), key=lambda kv: -kv[1]):
        print(f"      {name:<24} {n:>5,}")
    print(f"  elapsed       : {time.perf_counter() - overall:.1f}s")

    print(f"\n{BANNER}\n  VIEW THE DASHBOARD\n{BANNER}")
    print("  Offline (no server — just double-click):")
    print("      xdg-open frontend/index.html")
    print("\n  With the API (live endpoints + auto-reload):")
    print("      venv/bin/uvicorn api.main:app --reload --port 8000")
    print("      then open http://localhost:8000/\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
