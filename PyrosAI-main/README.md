# Pyros-AI

**AI-based detection and classification of industrial fires and persistent thermal sources
from satellite thermal data.**
Smart India Hackathon 2026 · Problem Statement **26162** · NTRO

Running on **real NASA FIRMS data** for Andhra Pradesh, with real OpenStreetMap
industrial facilities, in PostGIS.

---

## The problem, and what is actually new here

NASA's FIRMS detects thermal anomalies worldwide, but it only tells you that a
location is hot. It does not tell you *what is burning*. A gas flare, a steel
plant, a forest fire and a farmer clearing stubble arrive as the same row of the
same CSV.

Worse, NASA's **Static Thermal Anomalies** layer tags known industrial sites and
filters them out of alerting permanently. That removes the noise — and it
removes the signal with it. A refinery that starts burning five times hotter at
two in the morning is a known source, therefore suppressed, therefore invisible.

Pyros-AI does two things instead:

1. **Classifies** every thermal event as an industrial fire, gas flare, wildfire
   or agricultural burn — with a stated confidence and the evidence behind it.
2. **Detects behavioural anomalies** by scoring every site against *its own*
   operating history. The question is never "is this location hot?" but
   "is this location behaving unlike itself?"

The second is the capability the existing system structurally cannot have.

### The core idea

*What* a thermal source is shows up in *how it behaves over time*, not in how hot
it is at one instant. A flare and a wildfire can both read 40 MW. Only one of
them is still burning at 02:00 next Tuesday.

| | Persistence | Hours | Weekends | FRP variability | Spread |
|---|---|---|---|---|---|
| **Gas flare** | near-continuous | all, incl. night | active | very low | none |
| **Industrial** | recurring | working hours | idle | moderate | none |
| **Wildfire** | short, continuous | all, incl. night | active | high | growing |
| **Agricultural** | 1–2 days | daytime | either | low | none, clustered |

---

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. verify the database (creates the schema if needed)
python pipeline/db_setup.py

# 2. full run: fetch real FIRMS + OSM data, analyse, export
python run_pipeline.py --fetch --osm

# 3. view it
xdg-open frontend/index.html          # offline, no server needed
```

With the API instead:

```bash
uvicorn api.main:app --reload --port 8000
# http://localhost:8000/        dashboard
# http://localhost:8000/docs    interactive API docs
```

### PostgreSQL setup, if you need it

```bash
sudo -u postgres psql -c "CREATE ROLE pyros LOGIN PASSWORD 'pyros123';"
sudo -u postgres createdb -O pyros pyrosai
sudo -u postgres psql -d pyrosai -c "CREATE EXTENSION postgis;"
```

### Pipeline flags

| Flag | Effect |
|---|---|
| *(none)* | Re-analyse what is already in the database |
| `--fetch` | Pull fresh FIRMS detections first |
| `--historical` | Also pull the SP archive sources, for deeper baselines |
| `--osm` | Refresh the OSM facilities cache |
| `--vision` | Run CNN imagery verification on high-risk anomalies |
| `--days N` | Window length, default 60 |

Every stage also runs standalone for inspection:

```bash
python pipeline/cluster.py    # detections -> events
python pipeline/features.py   # the feature table
python pipeline/classify.py   # labels, confidences, refusals
python pipeline/anomaly.py    # baselines and flagged deviations
```

---

## Architecture

```
Pyros-AI/
├── data/
│   ├── fetch_firms.py     NASA FIRMS -> PostGIS
│   └── fetch_osm.py       OSM industrial facilities -> PostGIS
├── pipeline/
│   ├── db_setup.py        PostGIS schema + shared DB helpers
│   ├── cluster.py         DBSCAN: hot pixels -> thermal events
│   ├── features.py        temporal / intensity / spatial features
│   ├── classify.py        scored rule engine  → Phase 3: RF + XGBoost + SHAP
│   ├── anomaly.py         z-score + hour novelty + isolation forest
│   └── export.py          risk scoring, PostGIS, frontend payload
├── vision/cnn_verify.py   EfficientNet-B0 land cover on Sentinel-2 / Esri imagery
├── api/main.py            FastAPI, serves the dashboard and the JSON API
├── frontend/              Leaflet dashboard, runs from file://
└── run_pipeline.py        orchestrator
```

```
FIRMS ──┐
        ├─> cluster ─> features ─> classify ─┐
OSM   ──┘                    └──> anomaly ───┴─> export ─> results.json ─> dashboard
                                                        └─> PostGIS events
```

### Database

Three tables, geometry as `GEOMETRY(Point, 4326)` with GiST indexes:
`detections` (raw FIRMS pixels), `events` (analysed clusters),
`facilities` (OSM industrial sites).

---

## Decisions worth defending

**Two FRP variability measures.** `frp_variance` is the plain variance;
`frp_robust_var` is IQR-over-mean. Classification reads the robust one, anomaly
detection reads the raw one. Without the split, a single 5× excursion inflates a
factory's variance into wildfire territory and the spike silently reclassifies
the site it happened at.

**The baseline is measured after IQR cleaning.** The excursion being detected is
itself in the history. Left in, it drags the mean up and inflates the standard
deviation, so the event partly masks its own z-score.

**Proximity is necessary evidence for "Industrial Fire", not one criterion of
three.** As originally specified, "weekday daytime" plus "fixed footprint"
normalise to 0.50 and win outright with no industrial evidence at all — on the
live data that put 263 of 295 industrial calls more than a kilometre from any
mapped facility, the worst 45 km out. A distance-decay prior now gates the class.
It is 31 events, all of them within 1 km of a real facility.

**Hour-novelty needs more history than intensity does.** To claim "this site has
never been active at 12:00" you need enough prior passes that you would expect
to have seen 12:00. Satellite overpasses cluster at a handful of local times, so
a site observed four times has barely sampled the clock. The bar is 12 prior
detections; the z-score bar is 6.

**The classifier is allowed to refuse.** A class is asserted only if it clears an
absolute score bar and beats the runner-up by a margin. 126 of 673 events come
back as *Insufficient Evidence* rather than being guessed at. A wrong label on a
government alerting queue costs more than an honest abstention.

**An isolation-forest flag is not a deviation multiple.** It is a population
comparison, not a departure from a site's own baseline, so those events show
"Outlier" and their isolation score rather than a meaningless "1.0×".

**All times are IST.** FIRMS publishes UTC. "Does this site run at night?" is a
question about local time, so every hour-of-day and day-of-week feature is
computed at UTC+05:30.

---

## Validation — World Bank flare registry

Our classifier infers "Gas Flare" from behaviour alone: burns at night, burns at
weekends, recurs across many days, steady output. That is an inference, so it is
cross-checked against an external registry of documented flare sites.

**Reference.** World Bank / NOAA Global Gas Flaring survey, 2012–2024 —
156,397 global records, one row per site per year. Inside the Andhra Pradesh
study region that is 80 survey rows, which collapse to **26 distinct physical
sites** operated by ONGC, Reliance, GAIL and Oil India (Tatipaka, Gopavaram,
Nagayalanka, Kesanapalli West, Malleswaram, Mandapeta, Padmavati and others).

Deduplication is spatial, not by coordinate equality: the survey re-derives each
site's position annually, so one physical stack drifts a few hundred metres
between years. Grouping on exact coordinates produced eight separate
"Gopavaram" entries spread over 16.5090–16.5111 for what is one flare.

**Match radius is 2 km.** A VIIRS pixel is 375 m at nadir and the registry
coordinates are themselves satellite-derived, so sub-kilometre agreement would
be false precision.

```bash
python pipeline/validate_flares.py            # load the registry
python run_pipeline.py --validate-flares      # reload, then annotate every event
python pipeline/validate_flares.py --report   # agreement report
```

### What the cross-check found

It immediately exposed a real defect. Before this was wired in, 341 of 673
events were classified Gas Flare — and **289 of them were single detections**.
A lone pixel scored a perfect 1.00: `persistence` is `unique_days / date_range`,
which is 1/1 for a one-day event, `frp_robust_var` is 0 with one sample, and one
night-time weekend pixel gives `night_share = weekend_share = 1.00`. One hot
pixel at 02:00 on a Sunday read as a textbook flare.

The thresholds were also calibrated on Phase-1 synthetic data. Real persistent
sources in this region run `persistence` 0.26–0.52 and robust FRP variation
0.44–0.82, because cloud cover and overpass geometry make real detection sparse
and noisy — nothing like the >0.70 and <0.20 an idealised flare would show.

Both are fixed: recurrence is now counted in **separate days observed**, each
criterion is gated on enough evidence to support the specific claim it makes,
and observation volume scales **confidence** rather than score — so brevity,
which is the whole signature of an agricultural burn, is not penalised.

### Current agreement: 0 of 18

Honestly reported, and the reason is informative rather than embarrassing:

- The 4 events that *do* sit within 2 km of a documented flare have **1–4
  detections each**. Our classifier refuses to call them anything from that
  little evidence, and the registry says a flare is there. Those render as a
  grey "registry note — warrants review" panel.
- The 18 events we *do* classify as Gas Flare are **14–424 km from any
  documented flare**, but **0.1–2.9 km from a cement plant, steel works, mine or
  factory**. The registry covers the Krishna–Godavari oil and gas fields; these
  sources are spread statewide across heavy industry.

That is a finding, not a failure: **continuous night-and-weekend burning is
characteristic of continuous-process industry in general, not of gas flaring
specifically.** A cement kiln runs 24/7 exactly like a flare stack. Temporal
behaviour alone cannot separate them; what separates them is what is underneath.
The classifier now uses that — hydrocarbon facility context supports a flare
reading, heavy-industry context argues against it — and Industrial Fire now
recognises continuous-process plants instead of only shift-based ones.

### How to read the result

A match is positive evidence: we inferred a flare from behaviour and a
documented flare is there.

A non-match is **not** proof of misclassification. The registry covers oil and
gas sector flaring surveyed at ~750 m resolution. It does not list refinery
process flares, small industrial flares, or sites commissioned after the last
survey year. Unvalidated flares are therefore flagged for review, never counted
as errors.

**This is an agreement rate, not an accuracy figure.** The registry is also
deliberately *not* fed into classification — using it as an input would make the
validation circular.

## Performance

The dashboard is driven live while people interact with it, so the render path is
built for it and measured, not assumed. Measured in headless Chrome with software
rasterisation (pessimistic — real GPU compositing is faster), on the full 673-event
payload:

| | Measured | Budget |
|---|---|---|
| Time to interactive, from `file://` | **282 ms** | 2000 ms |
| Detail panel render | **< 1 ms** | 100 ms |
| Queue rebuild, 673 rows | **< 1 ms** | — |
| Animated pan / zoom, frame p50 | **0.0 ms** | 16.7 ms |
| Animated zoom, frame p95 | **10.1 ms** | 16.7 ms |
| Long tasks > 50 ms | **none** | none |

How:

- **Everything per-event is precomputed once** at load — queue row HTML, SVG
  chart path strings, histogram bars, score bars, evidence lists — and cached on
  the event object. Selecting an event is string concatenation plus one
  `innerHTML` assignment, never numeric work.
- **Markers are clustered** above 100 events, with `chunkedLoading` so adding
  several hundred never blocks the main thread, and cluster split/merge
  animation disabled.
- **The anomaly pulse pauses during pan and zoom.** This was the single biggest
  frame-time win: p95 during a rapid-pan stress test went from 134 ms to 70 ms.
  `will-change` was *removed* for the same reason — pinning a compositor layer
  per marker cost more than it saved.
- Below-the-fold detail sections use `content-visibility: auto`, so the browser
  skips their layout and paint until they scroll into view.
- One delegated listener for the whole queue; filter clicks debounced 150 ms;
  map-driven DOM updates batched through `requestAnimationFrame`.
- Animations only ever touch `transform` and `opacity`, and `prefers-reduced-motion`
  disables the pulse outright.

## Offline behaviour

The dashboard must not depend on the venue's network.

- Leaflet and MarkerCluster are **vendored** in `frontend/vendor/`.
- `export.py` writes `results.data.js` alongside `results.json`, because
  browsers block `fetch()` against `file://` URLs — a double-clicked page cannot
  read a local JSON file, but it can load a script. `app.js` tries
  `window.PYROS_DATA`, then `./results.json`, then `/api/events`.
- Basemaps degrade in three tiers: Esri tiles → OpenStreetMap → a local vector
  basemap (graticule plus the state boundary) drawn with no network at all. The
  status light in the header turns amber. Only the tile imagery is ever lost.

**Note on the dark basemap:** it is Esri Dark Gray Canvas, *not* CartoDB Dark
Matter. CartoDB still answers HTTP 200 without an API key, so it looks healthy in
a network log, but the PNG it returns is stamped "API KEY REQUIRED" diagonally
across every tile. Verified by fetching a tile directly. If you obtain a Carto
key, add it as one entry in the `BASEMAPS` registry in `app.js`.

---

## Current results

Live data, Andhra Pradesh, 60-day window:

| | |
|---|---|
| Detections ingested | 2,155 (VIIRS S-NPP / NOAA-20 / NOAA-21 + MODIS) |
| OSM facilities cached | 16,160 (incl. 11 refineries, 715 power plants) |
| Thermal events | 673 |
| Industrial fires | 31 — all within 1 km of a mapped facility |
| Gas flares / agricultural / wildfire | 341 / 155 / 20 |
| Refused as insufficient evidence | 126 |
| Behavioural anomalies | 23 |
| Full pipeline runtime | ~3 s (11 s with `--vision`) |

The highest-risk event is a real named pharmaceutical plant flagged for
deviating from its own baseline — exactly the case NASA's static layer filters
away.

**These are not accuracy claims.** There is no labelled ground truth for this
region and window, so what the pipeline produces is a ranked, explained,
reproducible operating picture — not a measured precision figure. Quantitative
validation needs labels, and that is Phase 4 work.

---

## Vision layer

`--vision` fetches imagery for events that are both anomalous and high-risk, and
runs EfficientNet-B0 over it. It degrades rather than failing:

- Sentinel-2 quicklook via Copernicus (needs `SENTINEL_*` in `.env`)
  → Esri World Imagery tile (keyless) → no image.
- `models/efficientnet_eurosat.pth` → real 10-class EuroSAT land cover.
  Without that checkpoint the imagery is still fetched and attached, but **no
  land-cover class is claimed** — an untrained 10-class head would emit a
  confident-looking label with nothing behind it.

## Roadmap

| Phase | Work |
|---|---|
| 2 ✅ | Real FIRMS + OSM, PostGIS, rule classifier, anomaly detection, dashboard |
| 3 | Weak supervision, RF + XGBoost ensemble, Platt calibration, SHAP |
| 4 | Validation against reference data, confusion matrix, measured accuracy |

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness + event count |
| `GET /api/meta` | window, counts, generation time |
| `GET /api/stats` | counts by classification, top risk |
| `GET /api/events` | `classification`, `anomaly_only`, `min_risk`, `limit` |
| `GET /api/events/{id}` | one event, full detail |
| `POST /api/refresh` | re-run the pipeline in the background |

The payload is cached in module memory and re-read only when `results.json`
changes on disk.
