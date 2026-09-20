"""Independent validation against the World Bank Global Gas Flaring database.

Our classifier says "Gas Flare" from behaviour alone: burns at night, burns at
weekends, stable FRP, no spread. That is an inference. This module checks it
against an external registry of documented flare sites, which is the closest
thing to ground truth available for this region.

The registry is the World Bank / NOAA VIIRS global flaring survey, 2012-2024:
156,397 rows, one per site per year. Collapsed to distinct sites inside the
Andhra Pradesh study region it yields seven, operated by ONGC, Reliance and
Oil India.

What a match does and does not prove
------------------------------------
A match is positive evidence: we said flare, and a documented flare is there.

A non-match is *not* proof of error. The registry covers oil and gas sector
flaring surveyed at ~750 m VIIRS resolution; it does not list refinery process
flares, small industrial flares, or sites commissioned after the survey year.
So agreement rate is reported as agreement, never as accuracy, and an
unvalidated flare is flagged for review rather than counted as wrong.

Match radius is 2 km: a VIIRS pixel is 375 m at nadir and the registry
coordinates are themselves derived from satellite detections, so sub-kilometre
agreement is not meaningful precision.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psycopg2.extras import execute_values  # noqa: E402

from pipeline.db_setup import STUDY_BBOX, get_conn  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XLSX = os.path.join(ROOT, "data", "worldbank", "flare_locations.xlsx")
RESULTS_JSON = os.path.join(ROOT, "frontend", "results.json")

MATCH_RADIUS_M = 2000.0
# The survey re-derives each site's coordinates annually, so one physical flare
# appears at slightly different positions year to year. Registry points within
# this distance are treated as the same stack.
SITE_CLUSTER_M = 1500.0
FLARE_CLASS = "Gas Flare"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS known_flares (
    id            BIGSERIAL PRIMARY KEY,
    geom          GEOMETRY(Point, 4326),
    latitude      DOUBLE PRECISION,
    longitude     DOUBLE PRECISION,
    country       TEXT,
    field_name    TEXT,
    operator      TEXT,
    field_type    TEXT,
    location_type TEXT,
    flare_level   TEXT,
    years_active  INTEGER,
    year_first    INTEGER,
    year_last     INTEGER,
    volume_m3     DOUBLE PRECISION,
    UNIQUE (latitude, longitude)
);
CREATE INDEX IF NOT EXISTS known_flares_geom_idx ON known_flares USING GIST (geom);
"""


def ensure_schema() -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
        conn.commit()


# --------------------------------------------------------------------------
# reference loading
# --------------------------------------------------------------------------

def _norm(name: str) -> str:
    """Registry headers carry stray double spaces ('Field  Operator')."""
    return re.sub(r"\s+", " ", str(name)).strip().lower()


def load_reference(path: str = XLSX, verbose: bool = True) -> int:
    """Parse the registry, collapse it to distinct in-region sites, load it."""
    import pandas as pd

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"World Bank registry not found at {path}. Download the Global Gas "
            f"Flaring Reduction dataset and place flare_locations.xlsx there.")

    ensure_schema()
    if verbose:
        print(f"  reading {os.path.basename(path)} …")
    df = pd.read_excel(path, engine="openpyxl")
    cols = {_norm(c): c for c in df.columns}

    def col(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_lat, c_lon = col("latitude"), col("longitude")
    c_year, c_vol = col("year"), col("flaring vol (million m3)", "bcm")
    c_name, c_op = col("field name"), col("field operator")
    c_type, c_loc = col("field type"), col("location")
    c_level, c_country = col("flare level"), col("country")

    total = len(df)
    df = df[df[c_lat].notna() & df[c_lon].notna()]
    inside = df[
        df[c_lat].between(STUDY_BBOX["lat_min"], STUDY_BBOX["lat_max"])
        & df[c_lon].between(STUDY_BBOX["lon_min"], STUDY_BBOX["lon_max"])
    ].copy()

    if verbose:
        print(f"  {total:,} global records → {len(inside):,} inside the study bbox")

    if inside.empty:
        return 0

    # One row per site per year — and the survey re-derives coordinates every
    # year, so the same physical flare drifts by a couple of hundred metres
    # between rows. Grouping on exact coordinates therefore yields eight
    # separate "Gopavaram" entries spread over 16.5090-16.5111 for what is one
    # stack. Sites are clustered spatially instead, so the reference is a count
    # of physical flares rather than a count of survey rows.
    import numpy as np
    from sklearn.cluster import DBSCAN

    coords = np.radians(inside[[c_lat, c_lon]].to_numpy(dtype=float))
    inside["_site"] = DBSCAN(
        eps=SITE_CLUSTER_M / 6371000.0, min_samples=1,
        metric="haversine", algorithm="ball_tree",
    ).fit_predict(coords)

    if verbose:
        print(f"  {len(inside):,} survey rows → "
              f"{inside['_site'].nunique()} distinct physical sites "
              f"(clustered within {SITE_CLUSTER_M:.0f} m)")

    def _mode(series):
        vals = [str(v).strip() for v in series.dropna() if str(v).strip()
                and str(v).strip().lower() != "nan"]
        if not vals:
            return None
        return max(set(vals), key=vals.count)

    rows = []
    for _, grp in inside.groupby("_site"):
        # Representative position: the mean of every year's survey fix.
        lat = float(grp[c_lat].mean())
        lon = float(grp[c_lon].mean())
        years = sorted(int(y) for y in grp[c_year].dropna().unique())
        volume_m3 = (float(grp[c_vol].fillna(0.0).sum()) * 1e6) if c_vol else 0.0

        level = None
        if c_level is not None:
            # Report the worst level ever recorded, not the most common one:
            # a site that reached Large matters even if most years were Small.
            seen = {str(v).strip().title() for v in grp[c_level].dropna()}
            for candidate in ("Large", "Medium", "Small"):
                if candidate in seen:
                    level = candidate
                    break

        rows.append((
            lon, lat, lat, lon,
            _mode(grp[c_country]) if c_country else None,
            _mode(grp[c_name]) if c_name else None,
            _mode(grp[c_op]) if c_op else None,
            _mode(grp[c_type]) if c_type else None,
            _mode(grp[c_loc]) if c_loc else None,
            level,
            len(years), (years[0] if years else None), (years[-1] if years else None),
            round(volume_m3, 2),
        ))

    sql = """
    INSERT INTO known_flares
        (geom, latitude, longitude, country, field_name, operator, field_type,
         location_type, flare_level, years_active, year_first, year_last, volume_m3)
    VALUES %s
    ON CONFLICT (latitude, longitude) DO UPDATE SET
        field_name = EXCLUDED.field_name, operator = EXCLUDED.operator,
        flare_level = EXCLUDED.flare_level, years_active = EXCLUDED.years_active,
        year_last = EXCLUDED.year_last, volume_m3 = EXCLUDED.volume_m3
    """
    template = ("(ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s, %s, %s, %s, %s, %s, "
                "%s, %s, %s, %s, %s, %s)")
    with get_conn() as conn, conn.cursor() as cur:
        execute_values(cur, sql, rows, template=template, page_size=500)
        conn.commit()

    if verbose:
        print(f"  {len(rows)} distinct flare sites loaded into known_flares")
        for site in list_reference():
            name = site["field_name"] or "(unnamed site)"
            op = site["operator"] or "unknown operator"
            print(f"    {name:<22} {op:<18} "
                  f"{site['year_first']}–{site['year_last']} "
                  f"({site['years_active']:>2}y)  {site['flare_level'] or '?':<7} "
                  f"{site['volume_m3']/1e6:>7.2f} Mm³  "
                  f"{site['latitude']:.4f}, {site['longitude']:.4f}")
    return len(rows)


def list_reference() -> list[dict]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""SELECT field_name, operator, flare_level, years_active,
                              year_first, year_last, volume_m3, latitude, longitude
                       FROM known_flares ORDER BY field_name NULLS LAST""")
        keys = ["field_name", "operator", "flare_level", "years_active",
                "year_first", "year_last", "volume_m3", "latitude", "longitude"]
        return [dict(zip(keys, r)) for r in cur.fetchall()]


# --------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------

def _coords(event: dict) -> tuple[float, float]:
    """Pipeline events use latitude/longitude; exported ones use lat/lon."""
    lat = event.get("latitude", event.get("lat"))
    lon = event.get("longitude", event.get("lon"))
    return float(lat), float(lon)


MATCH_SQL = """
WITH pts(idx, lon, lat) AS (VALUES %s)
SELECT p.idx, f.field_name, f.operator, f.flare_level, f.years_active,
       f.volume_m3, f.year_last,
       ST_Distance(f.geom::geography,
                   ST_SetSRID(ST_MakePoint(p.lon, p.lat), 4326)::geography) AS dist_m
FROM pts p
CROSS JOIN LATERAL (
    SELECT field_name, operator, flare_level, years_active, volume_m3, year_last, geom
    FROM known_flares
    ORDER BY geom <-> ST_SetSRID(ST_MakePoint(p.lon, p.lat), 4326)
    LIMIT 1
) f
"""


def annotate_events(events: list[dict], conn=None,
                    radius_m: float = MATCH_RADIUS_M) -> list[dict]:
    """Attach `wb_match` and `flare_validated` to every event.

    `conn` is accepted so the caller owns one connection for the whole loop.
    The nearest-site lookup itself is issued as a single batched LATERAL query
    rather than one round trip per event — same index path, one round trip
    instead of several hundred.
    """
    if not events:
        return events

    owns = conn is None
    conn = conn or get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.known_flares')")
            if cur.fetchone()[0] is None:
                raise RuntimeError("known_flares table does not exist — "
                                   "run: python pipeline/validate_flares.py")
            cur.execute("SELECT COUNT(*) FROM known_flares")
            if cur.fetchone()[0] == 0:
                raise RuntimeError("known_flares is empty — "
                                   "run: python pipeline/validate_flares.py")

            values = ",".join(
                cur.mogrify("(%s,%s,%s)", (i, lon, lat)).decode()
                for i, (lat, lon) in enumerate(_coords(e) for e in events))
            cur.execute(MATCH_SQL % values)
            nearest = {r[0]: r for r in cur.fetchall()}
    finally:
        if owns:
            conn.close()

    for i, event in enumerate(events):
        row = nearest.get(i)
        match = None
        if row and row[7] is not None and float(row[7]) <= radius_m:
            match = {
                "field_name": row[1] or "(unnamed site)",
                "operator": row[2] or "unknown operator",
                "flare_level": row[3],
                "years_active": int(row[4]) if row[4] is not None else None,
                "volume_m3": float(row[5]) if row[5] is not None else None,
                "year_last": int(row[6]) if row[6] is not None else None,
                "distance_m": round(float(row[7]), 1),
            }
        event["wb_match"] = match
        # Only a flare classification can be validated or contradicted by a
        # flare registry. Everything else records the match and stays None.
        event["flare_validated"] = (
            bool(match) if event.get("classification") == FLARE_CLASS else None)

    return events


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def report(events: list[dict]) -> dict:
    flares = [e for e in events if e.get("classification") == FLARE_CLASS]
    validated = [e for e in flares if e.get("flare_validated") is True]
    unvalidated = [e for e in flares if e.get("flare_validated") is False]
    cross = [e for e in events
             if e.get("flare_validated") is None and e.get("wb_match")]

    rate = (len(validated) / len(flares)) if flares else 0.0

    bar = "=" * 72
    print(f"\n{bar}\n  WORLD BANK FLARE REGISTRY — VALIDATION REPORT\n{bar}")

    sites = list_reference()
    print(f"\n  Reference: {len(sites)} distinct documented flare sites in the study region")
    for s in sites:
        print(f"    {(s['field_name'] or '(unnamed)'): <22} "
              f"{(s['operator'] or 'unknown'): <18} "
              f"{s['year_first']}–{s['year_last']}  "
              f"{(s['flare_level'] or '?'): <7} "
              f"{s['volume_m3'] / 1e6:>8.2f} Mm³ total")

    print(f"\n  Match radius: {MATCH_RADIUS_M:.0f} m "
          f"(VIIRS pixel 375 m; registry coordinates are satellite-derived)")

    print(f"\n{bar}\n  AGREEMENT\n{bar}")
    print(f"  Events classified as Gas Flare : {len(flares):>5,}")
    print(f"    confirmed by registry        : {len(validated):>5,}")
    print(f"    not listed in registry       : {len(unvalidated):>5,}")
    print(f"  Agreement rate                 : {rate:>5.1%}")

    if validated:
        print("\n  Confirmed:")
        for e in sorted(validated, key=lambda x: x["wb_match"]["distance_m"]):
            m = e["wb_match"]
            print(f"    {e.get('id', '?'): <16} {m['field_name']: <22} "
                  f"{m['operator']: <18} {m['distance_m']:>7.0f} m")

    if cross:
        print(f"\n  Registry flare nearby but classified otherwise ({len(cross)}) — "
              f"these warrant review:")
        for e in sorted(cross, key=lambda x: x["wb_match"]["distance_m"])[:10]:
            m = e["wb_match"]
            print(f"    {e.get('id', '?'): <16} {e.get('classification', '?'): <22} "
                  f"{m['field_name']: <22} {m['distance_m']:>7.0f} m")

    print(f"\n{bar}\n  INTERPRETATION\n{bar}")
    print("  A match is positive evidence: we inferred a flare from behaviour")
    print("  alone and a documented flare is there.")
    print()
    print("  A non-match is NOT proof of misclassification. The registry covers")
    print("  oil and gas sector flaring surveyed at ~750 m resolution. It does not")
    print("  list refinery process flares, small industrial flares, or sites")
    print("  commissioned after the last survey year. Unvalidated flares are")
    print("  therefore flagged for review, not counted as errors.")
    print()
    print("  This is an agreement rate, not an accuracy figure.")
    print(f"{bar}\n")

    return {"flares": len(flares), "validated": len(validated),
            "unvalidated": len(unvalidated), "agreement_rate": round(rate, 4),
            "cross_class": len(cross)}


def _report_from_results() -> int:
    if not os.path.exists(RESULTS_JSON):
        print(f"{RESULTS_JSON} not found — run: python run_pipeline.py")
        return 1
    with open(RESULTS_JSON, encoding="utf-8") as fh:
        payload = json.load(fh)
    events = payload.get("events", [])
    print(f"  loaded {len(events):,} events from {os.path.basename(RESULTS_JSON)}")

    conn = get_conn()
    try:
        annotate_events(events, conn)
    finally:
        conn.close()
    report(events)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true",
                    help="match frontend/results.json against the registry and "
                         "print the agreement report")
    ap.add_argument("--xlsx", default=XLSX, help="path to flare_locations.xlsx")
    args = ap.parse_args()

    if args.report:
        return _report_from_results()

    print("Loading World Bank flare reference data …")
    n = load_reference(args.xlsx)
    print(f"\ndone — {n} sites in known_flares")
    return 0


if __name__ == "__main__":
    sys.exit(main())
