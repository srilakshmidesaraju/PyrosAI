"""Fetch real NASA FIRMS active-fire detections into PostGIS.

Two things about the FIRMS area API shape this module:

  * Every source is capped at 5 days per request, NRT and SP alike. A 60-day
    baseline window is therefore 12 paged requests per source, not one call.
  * Each source has its own availability window. VIIRS_SNPP_SP stops months
    back, MODIS_NRT only starts a few months ago. Asking outside a source's
    range returns an error page, not data, so the range is fetched up front
    from /api/data_availability and every chunk is clipped to it.

Endpoint:
    /api/area/csv/{KEY}/{SOURCE}/{BBOX}/{DAYS}/{START_DATE}

MODIS and VIIRS disagree on column names (brightness/bright_t31 vs
bright_ti4/bright_ti5) and on how they encode confidence (0-100 vs l/n/h).
Both are normalised here so that nothing downstream has to care.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import requests
from dotenv import load_dotenv
from psycopg2.extras import execute_values

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.db_setup import FIRMS_BBOX, get_conn  # noqa: E402

load_dotenv()

MAP_KEY = os.getenv("FIRMS_MAP_KEY", "").strip()
BASE = "https://firms.modaps.eosdis.nasa.gov/api"
MAX_DAYS_PER_REQUEST = 5          # hard API limit, all sources
REQUEST_TIMEOUT = 90
WORKERS = 4                       # polite concurrency against NASA's API

# Near-real-time: current data, available within ~3 hours of overpass.
NRT_SOURCES = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT"]
# Standard processing: better geolocation, but lags NRT by months. Useful for
# deep historical baselines, useless for "what happened this week".
SP_SOURCES = ["VIIRS_SNPP_SP", "MODIS_SP"]


# --------------------------------------------------------------------------
# availability
# --------------------------------------------------------------------------

def get_availability() -> dict[str, tuple[date, date]]:
    """Per-source min/max date, so chunks outside a source's range are skipped."""
    url = f"{BASE}/data_availability/csv/{MAP_KEY}/ALL"
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as err:
        print(f"  ! availability lookup failed ({err}); proceeding unclipped")
        return {}

    out = {}
    for row in csv.DictReader(io.StringIO(resp.text)):
        try:
            out[row["data_id"]] = (
                datetime.strptime(row["min_date"], "%Y-%m-%d").date(),
                datetime.strptime(row["max_date"], "%Y-%m-%d").date(),
            )
        except (ValueError, KeyError):
            continue
    return out


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------

def _chunks(start: date, end: date):
    """Split [start, end] into <=5-day windows the API will accept."""
    cur = start
    while cur <= end:
        span = min(MAX_DAYS_PER_REQUEST, (end - cur).days + 1)
        yield cur, span
        cur += timedelta(days=span)


def _fetch_chunk(source: str, start: date, span: int) -> list[dict]:
    url = f"{BASE}/area/csv/{MAP_KEY}/{source}/{FIRMS_BBOX}/{span}/{start:%Y-%m-%d}"
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                time.sleep(1.5 * (attempt + 1))
                continue
            text = resp.text.lstrip()
            # Errors come back as plain prose with a 200, not as JSON or a 4xx.
            if not text.lower().startswith("latitude"):
                return []
            return list(csv.DictReader(io.StringIO(text)))
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    return []


def fetch_source(source: str, start: date, end: date,
                 availability: dict) -> list[dict]:
    """All detections for one source across the window, paged and clipped."""
    if source in availability:
        lo, hi = availability[source]
        start, end = max(start, lo), min(end, hi)
        if start > end:
            print(f"  {source:<18} no overlap with its availability window — skipped")
            return []

    windows = list(_chunks(start, end))
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for chunk in pool.map(lambda w: _fetch_chunk(source, w[0], w[1]), windows):
            rows.extend(chunk)
    print(f"  {source:<18} {len(rows):>7,} detections  "
          f"({start} → {end}, {len(windows)} requests)")
    return rows


# --------------------------------------------------------------------------
# normalise + load
# --------------------------------------------------------------------------

def _norm(row: dict) -> tuple | None:
    """One FIRMS CSV row -> one detections tuple, or None if unusable."""
    try:
        lat = float(row["latitude"])
        lon = float(row["longitude"])
    except (KeyError, ValueError, TypeError):
        return None

    # VIIRS calls it bright_ti4; MODIS calls it brightness.
    bright = row.get("bright_ti4") or row.get("brightness")
    try:
        bright = float(bright) if bright not in (None, "") else None
    except ValueError:
        bright = None

    try:
        frp = float(row.get("frp") or 0.0)
    except ValueError:
        frp = 0.0

    acq_date = (row.get("acq_date") or "").strip()
    if not acq_date:
        return None
    # HHMM, sometimes without the leading zero ("714" means 07:14).
    acq_time = str(row.get("acq_time") or "0").strip().zfill(4)

    return (
        lat, lon, lon, lat,                       # last two feed ST_MakePoint
        bright, frp,
        str(row.get("confidence", "")).strip(),
        acq_date, acq_time,
        (row.get("satellite") or "").strip(),
        (row.get("instrument") or "").strip(),
        (row.get("daynight") or "").strip(),
        (row.get("version") or "").strip(),
    )


INSERT_SQL = """
INSERT INTO detections
    (latitude, longitude, geom, bright_ti4, frp, confidence,
     acq_date, acq_time, satellite, instrument, daynight, version)
VALUES %s
ON CONFLICT ON CONSTRAINT detections_natural_key DO NOTHING
"""
TEMPLATE = ("(%s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s, %s, %s, "
            "%s, %s, %s, %s, %s, %s)")


def insert_detections(rows: list[dict]) -> int:
    """Batch insert. Returns the number of genuinely new rows."""
    tuples = [t for t in (_norm(r) for r in rows) if t is not None]
    if not tuples:
        return 0
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM detections")
        before = cur.fetchone()[0]
        execute_values(cur, INSERT_SQL, tuples, template=TEMPLATE, page_size=5000)
        cur.execute("SELECT COUNT(*) FROM detections")
        after = cur.fetchone()[0]
        conn.commit()
    return after - before


def fetch(days: int = 60, historical: bool = False,
          end: date | None = None) -> int:
    """Fetch `days` of detections ending today (or at `end`) and load them."""
    if not MAP_KEY:
        print("FIRMS_MAP_KEY is not set in .env — cannot fetch.")
        return 0

    end = end or date.today()
    start = end - timedelta(days=days - 1)
    sources = NRT_SOURCES + (SP_SOURCES if historical else [])

    print(f"window     : {start} → {end}  ({days} days)")
    print(f"bbox       : {FIRMS_BBOX}")
    print(f"sources    : {', '.join(sources)}")
    print("fetching (FIRMS caps every request at 5 days, so these are paged):")

    availability = get_availability()
    all_rows: list[dict] = []
    for source in sources:
        all_rows.extend(fetch_source(source, start, end, availability))

    if not all_rows:
        print("\nno detections returned for this window.")
        return 0

    inserted = insert_detections(all_rows)
    print(f"\n{len(all_rows):,} fetched → {inserted:,} new rows "
          f"({len(all_rows) - inserted:,} already present)")
    return inserted


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch NASA FIRMS detections into PostGIS.")
    ap.add_argument("--days", type=int, default=60,
                    help="length of the window in days (default 60)")
    ap.add_argument("--historical", action="store_true",
                    help="also pull the SP archive sources for deeper baselines")
    ap.add_argument("--end", help="last day of the window, YYYY-MM-DD (default today)")
    args = ap.parse_args()

    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else None
    fetch(days=args.days, historical=args.historical, end=end)
    return 0


if __name__ == "__main__":
    sys.exit(main())
