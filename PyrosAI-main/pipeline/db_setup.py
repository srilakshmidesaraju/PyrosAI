"""PostGIS schema and the shared database helpers every other stage imports.

Three tables:

  detections  one row per satellite hot pixel, straight from FIRMS
  events      one row per DBSCAN cluster, carrying the full analytical result
  facilities  industrial sites from OpenStreetMap, used for spatial context

Geometry is stored as GEOMETRY(Point, 4326) alongside plain lat/lon columns.
The duplication is deliberate: PostGIS does the spatial work, and the plain
columns keep pandas and JSON export from having to parse WKB.
"""

from __future__ import annotations

import os
import sys

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

load_dotenv()

DB = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "pyrosai"),
    "user": os.getenv("DB_USER", "pyros"),
    "password": os.getenv("DB_PASS", "pyros123"),
}

STUDY_BBOX = {
    "lat_min": float(os.getenv("STUDY_LAT_MIN", "12.6")),
    "lat_max": float(os.getenv("STUDY_LAT_MAX", "19.2")),
    "lon_min": float(os.getenv("STUDY_LON_MIN", "76.7")),
    "lon_max": float(os.getenv("STUDY_LON_MAX", "84.8")),
}
# FIRMS wants west,south,east,north
FIRMS_BBOX = (f"{STUDY_BBOX['lon_min']},{STUDY_BBOX['lat_min']},"
              f"{STUDY_BBOX['lon_max']},{STUDY_BBOX['lat_max']}")

REGION_NAME = "Andhra Pradesh, India"
# FIRMS timestamps are UTC. Every hour-of-day and day-of-week feature in this
# project is computed in IST, because "does this site run at night?" is a
# question about local time, not about Greenwich.
IST_OFFSET_HOURS = 5.5


def get_conn(**kwargs):
    """psycopg2 connection using the .env credentials."""
    return psycopg2.connect(**DB, **kwargs)


def get_dict_conn():
    return psycopg2.connect(**DB, cursor_factory=RealDictCursor)


def get_engine():
    """SQLAlchemy engine, so pandas.read_sql gets a supported connectable."""
    from sqlalchemy import create_engine
    return create_engine(
        f"postgresql+psycopg2://{DB['user']}:{DB['password']}"
        f"@{DB['host']}:{DB['port']}/{DB['dbname']}",
        pool_pre_ping=True,
    )


SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS postgis;

-- ------------------------------------------------------------- detections --
CREATE TABLE IF NOT EXISTS detections (
    id          BIGSERIAL PRIMARY KEY,
    latitude    DOUBLE PRECISION NOT NULL,
    longitude   DOUBLE PRECISION NOT NULL,
    geom        GEOMETRY(Point, 4326),
    bright_ti4  DOUBLE PRECISION,
    frp         DOUBLE PRECISION,
    confidence  TEXT,
    acq_date    DATE NOT NULL,
    acq_time    TEXT,
    satellite   TEXT,
    instrument  TEXT,
    daynight    TEXT,
    version     TEXT,
    fetched_at  TIMESTAMPTZ DEFAULT NOW(),
    -- One physical pixel is one row. Re-running a fetch over an overlapping
    -- window must not duplicate it, which is what ON CONFLICT DO NOTHING keys on.
    CONSTRAINT detections_natural_key
        UNIQUE (latitude, longitude, acq_date, acq_time, satellite, instrument)
);

CREATE INDEX IF NOT EXISTS detections_geom_idx ON detections USING GIST (geom);
CREATE INDEX IF NOT EXISTS detections_date_idx ON detections (acq_date);

-- ----------------------------------------------------------------- events --
CREATE TABLE IF NOT EXISTS events (
    id              BIGSERIAL PRIMARY KEY,
    event_id        TEXT UNIQUE NOT NULL,
    geom            GEOMETRY(Point, 4326),
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    name            TEXT,
    first_seen      TIMESTAMP,
    last_seen       TIMESTAMP,
    detection_count INTEGER,
    active_days     INTEGER,
    frp_mean        DOUBLE PRECISION,
    frp_peak        DOUBLE PRECISION,
    frp_variance    DOUBLE PRECISION,
    spread_rate     DOUBLE PRECISION,
    footprint_km    DOUBLE PRECISION,
    persistence     DOUBLE PRECISION,
    night_share     DOUBLE PRECISION,
    weekend_share   DOUBLE PRECISION,
    classification  TEXT,
    confidence      DOUBLE PRECISION,
    anomaly_flag    BOOLEAN DEFAULT FALSE,
    deviation_mult  DOUBLE PRECISION,
    risk_score      DOUBLE PRECISION,
    reason          TEXT,
    frp_history     JSONB,
    hour_histogram  JSONB,
    dow_histogram   JSONB,
    class_scores    JSONB,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS events_geom_idx  ON events USING GIST (geom);
CREATE INDEX IF NOT EXISTS events_risk_idx  ON events (risk_score DESC);

-- ------------------------------------------------------------- facilities --
CREATE TABLE IF NOT EXISTS facilities (
    id            BIGSERIAL PRIMARY KEY,
    osm_id        TEXT UNIQUE,
    geom          GEOMETRY(Point, 4326),
    latitude      DOUBLE PRECISION,
    longitude     DOUBLE PRECISION,
    name          TEXT,
    facility_type TEXT,
    risk_weight   DOUBLE PRECISION DEFAULT 1.0
);

CREATE INDEX IF NOT EXISTS facilities_geom_idx ON facilities USING GIST (geom);
CREATE INDEX IF NOT EXISTS facilities_type_idx ON facilities (facility_type);
"""


def create_schema(verbose: bool = True) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
        conn.commit()
    if verbose:
        print("schema ready (detections, events, facilities)")


def table_counts() -> dict[str, int]:
    out = {}
    with get_conn() as conn, conn.cursor() as cur:
        for table in ("detections", "events", "facilities"):
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            out[table] = cur.fetchone()[0]
    return out


def main() -> int:
    print(f"connecting to {DB['user']}@{DB['host']}:{DB['port']}/{DB['dbname']} …")
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT version(), postgis_version()")
            pg, gis = cur.fetchone()
    except psycopg2.OperationalError as err:
        print(f"\nCANNOT CONNECT: {err}".rstrip())
        print("\nCheck that PostgreSQL is running and that the role exists:")
        print("  sudo systemctl start postgresql")
        print(f"  sudo -u postgres psql -c \"CREATE ROLE {DB['user']} LOGIN PASSWORD '{DB['password']}';\"")
        print(f"  sudo -u postgres createdb -O {DB['user']} {DB['dbname']}")
        print(f"  sudo -u postgres psql -d {DB['dbname']} -c 'CREATE EXTENSION postgis;'")
        return 1

    print(f"  PostgreSQL : {pg.split(',')[0]}")
    print(f"  PostGIS    : {gis}")
    create_schema(verbose=False)
    print("  schema     : detections, events, facilities (+ GiST indexes)")
    for table, n in table_counts().items():
        print(f"    {table:<12} {n:>8,} rows")
    print(f"  study area : {REGION_NAME}  bbox {FIRMS_BBOX}")
    print("\ndatabase OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
