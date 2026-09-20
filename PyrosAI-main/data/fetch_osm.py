"""Fetch industrial facilities from OpenStreetMap into the facilities table.

Spatial context is what separates "a hot pixel" from "a hot pixel 200 m inside
a refinery". That context comes from OSM, and the expensive way to get it would
be an Overpass query per hotspot. This module does the opposite: one bounding
box query for the whole study region, cached in PostGIS, after which every
lookup is a local GiST index hit.

Overpass is a shared free service that rate-limits and times out under load, so
requests rotate across mirrors and retry before giving up.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import requests
from psycopg2.extras import execute_values

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.db_setup import STUDY_BBOX, get_conn  # noqa: E402

MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
TIMEOUT = 240

# Ordered most- to least-specific: the first tag that matches decides the type,
# so "oil refinery" wins over the generic "industrial landuse" it sits inside.
RISK_WEIGHTS = {
    "oil_refinery": 5.0, "petroleum": 5.0, "lng": 5.0,
    "gas": 4.5, "chemical": 4.5, "pharmaceutical": 4.0,
    "steel": 3.5, "power_plant": 3.5,
    "cement": 3.0, "mine": 3.0, "quarry": 2.5,
    "industrial": 2.0, "factory": 2.0, "warehouse": 1.0,
}


def _bbox() -> str:
    """Overpass wants south,west,north,east."""
    return (f"{STUDY_BBOX['lat_min']},{STUDY_BBOX['lon_min']},"
            f"{STUDY_BBOX['lat_max']},{STUDY_BBOX['lon_max']}")


def build_query() -> str:
    b = _bbox()
    clauses = []
    for kv in [
        ('"man_made"="works"'), ('"man_made"="petroleum_well"'),
        ('"man_made"="flare"'), ('"man_made"="storage_tank"'),
        ('"power"="plant"'), ('"landuse"="industrial"'),
        ('"landuse"="quarry"'), ('"industrial"'),
        ('"building"="industrial"'), ('"building"="warehouse"'),
        ('"man_made"="mineshaft"'), ('"resource"'),
    ]:
        clauses.append(f'  node[{kv}]({b});')
        clauses.append(f'  way[{kv}]({b});')
    return "[out:json][timeout:240];\n(\n" + "\n".join(clauses) + "\n);\nout center tags;"


def classify_facility(tags: dict) -> tuple[str, float]:
    """Map OSM tags onto a facility type and its risk weight."""
    blob = " ".join(f"{k}={v}".lower() for k, v in tags.items())

    def has(*words):
        return any(w in blob for w in words)

    if has("refinery", "oil_refinery"):            kind = "oil_refinery"
    elif has("petroleum", "oil_well", "petroleum_well"): kind = "petroleum"
    elif has("lng", "liquefied"):                  kind = "lng"
    elif has("gas"):                               kind = "gas"
    elif has("chemical", "petrochemical"):         kind = "chemical"
    elif has("pharmaceut", "bulk_drug", "drug"):   kind = "pharmaceutical"
    elif has("steel", "smelter", "foundry", "metallurg"): kind = "steel"
    elif has("power=plant", "power_plant", "generator"):  kind = "power_plant"
    elif has("cement", "lime", "clinker"):         kind = "cement"
    elif has("mine", "mineshaft", "colliery"):     kind = "mine"
    elif has("quarry"):                            kind = "quarry"
    elif has("warehouse"):                         kind = "warehouse"
    elif has("factory", "works", "manufactur"):    kind = "factory"
    else:                                          kind = "industrial"

    return kind, RISK_WEIGHTS.get(kind, 1.0)


def run_query(query: str) -> list[dict]:
    for mirror in MIRRORS:
        try:
            print(f"  querying {mirror.split('/')[2]} …", flush=True)
            resp = requests.post(mirror, data={"data": query}, timeout=TIMEOUT)
            if resp.status_code == 200:
                return resp.json().get("elements", [])
            print(f"    HTTP {resp.status_code}, trying next mirror")
        except (requests.RequestException, ValueError) as err:
            print(f"    {type(err).__name__}, trying next mirror")
        time.sleep(2)
    return []


INSERT_SQL = """
INSERT INTO facilities (osm_id, geom, latitude, longitude, name, facility_type, risk_weight)
VALUES %s
ON CONFLICT (osm_id) DO UPDATE
   SET facility_type = EXCLUDED.facility_type,
       risk_weight   = EXCLUDED.risk_weight,
       name          = EXCLUDED.name
"""
TEMPLATE = "(%s, ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s, %s, %s, %s, %s)"


def load(elements: list[dict]) -> int:
    rows = []
    for el in elements:
        # `out center` gives ways a centroid; nodes carry lat/lon directly.
        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lon = el.get("lon") or (el.get("center") or {}).get("lon")
        if lat is None or lon is None:
            continue
        tags = el.get("tags") or {}
        kind, weight = classify_facility(tags)
        name = tags.get("name") or tags.get("operator") or kind.replace("_", " ").title()
        rows.append((f"{el.get('type','?')}/{el.get('id')}", lon, lat,
                     lat, lon, name[:200], kind, weight))

    if not rows:
        return 0
    with get_conn() as conn, conn.cursor() as cur:
        execute_values(cur, INSERT_SQL, rows, template=TEMPLATE, page_size=2000)
        conn.commit()
    return len(rows)


def fetch() -> int:
    print(f"bbox       : {_bbox()}  (one query for the whole region, then cached)")
    elements = run_query(build_query())
    if not elements:
        print("\nOverpass returned nothing. The facilities table stays as it is;\n"
              "the pipeline still runs, it just has no spatial context to use.")
        return 0
    print(f"  {len(elements):,} OSM elements returned")
    n = load(elements)

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""SELECT facility_type, COUNT(*), MAX(risk_weight)
                       FROM facilities GROUP BY facility_type
                       ORDER BY MAX(risk_weight) DESC, COUNT(*) DESC""")
        print(f"\n{n:,} facilities loaded:")
        for kind, count, weight in cur.fetchall():
            print(f"    {kind:<16} {count:>6,}   risk weight {weight}")
    return n


def main() -> int:
    argparse.ArgumentParser(description="Fetch OSM industrial facilities.").parse_args()
    fetch()
    return 0


if __name__ == "__main__":
    sys.exit(main())
