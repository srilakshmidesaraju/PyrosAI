"""Behavioural anomaly detection - the core innovation.

NASA's Static Thermal Anomalies layer tags known industrial sites and filters
them out of alerting permanently. That removes the noise and it also removes the
signal: a refinery that starts burning five times hotter at 02:00 is a known
source, therefore suppressed, therefore invisible.

This module inverts that. Every site is scored against *its own* history, so the
question is never "is this location hot?" but "is this location behaving unlike
itself?" Three independent detectors, because industrial incidents show up
differently depending on what went wrong:

  z-score           intensity departure from the site's own cleaned baseline
  hour anomaly      activity in hours the site has never previously been active
  isolation forest  events that are odd relative to the whole population

A baseline needs history. Sites with fewer than MIN_HISTORY detections are
reported as having no baseline and are never z-flagged: with three readings the
standard deviation is meaningless and every site looks anomalous.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

Z_THRESHOLD = 3.0
MIN_HISTORY = 6              # detections needed before a baseline means anything
# Novelty needs a higher bar than intensity. To say "this site has never been
# active at 12:00" you need enough prior passes that you would expect to have
# seen 12:00 by now. Satellite overpasses cluster at a handful of local times,
# so a site seen four times has barely sampled the clock and almost every hour
# looks novel. Twelve prior detections is roughly three days of full coverage.
MIN_PRIOR_FOR_HOUR = 12
RECENT_WINDOW = 5            # how many trailing detections count as "current"
HOUR_SUPPORT = 0.02          # hour-histogram share below which an hour is novel
DEV_MULT_FOR_HOUR_FLAG = 2.0
ISO_CONTAMINATION = 0.1
ISO_THRESHOLD = 0.75

ISO_FEATURES = ["frp_mean", "frp_robust_var", "persistence", "night_share",
                "weekend_share", "spread_rate", "dist_to_facility_m"]


def _iqr_clean(values: np.ndarray) -> np.ndarray:
    """Drop outliers before measuring the baseline.

    The excursion we are trying to detect is itself in the history. Leaving it
    in drags the mean up and inflates the standard deviation, so the event
    partly masks its own z-score. Cleaning first, then measuring, keeps the
    baseline a description of normal behaviour.
    """
    if len(values) < 4:
        return values
    q25, q75 = np.percentile(values, [25, 75])
    iqr = q75 - q25
    keep = values[(values >= q25 - 1.5 * iqr) & (values <= q75 + 1.5 * iqr)]
    return keep if len(keep) >= 2 else values


def z_baseline(event: dict) -> dict:
    """Per-site intensity baseline and the departure from it."""
    frp = np.asarray(event["frp_history"], dtype=float)
    n = len(frp)
    if n < MIN_HISTORY:
        return {
            "baseline_established": False,
            "baseline_mean": round(float(np.median(frp)), 3) if n else 0.0,
            "baseline_std": 0.0,
            "z_score": 0.0,
            "deviation_mult": 1.0,
            "z_anomaly": False,
            "current_frp": round(float(frp[-1]), 3) if n else 0.0,
        }

    cleaned = _iqr_clean(frp)
    mean = float(np.median(cleaned))
    std = float(np.std(cleaned))
    denom = max(std, 0.01)

    # "Current" is the trailing window, not just the last pixel: a spike three
    # passes ago is still the thing an operator needs to see.
    recent = frp[-min(RECENT_WINDOW, n):]
    zs = (recent - mean) / denom
    idx = int(np.argmax(np.abs(zs)))
    z = float(zs[idx])
    current = float(recent[idx])

    return {
        "baseline_established": True,
        "baseline_mean": round(mean, 3),
        "baseline_std": round(std, 3),
        "z_score": round(z, 2),
        "deviation_mult": round(current / max(mean, 0.1), 2),
        "z_anomaly": bool(abs(z) > Z_THRESHOLD),
        "current_frp": round(current, 3),
    }


def hour_anomaly(event: dict, deviation_mult: float) -> dict:
    """Activity in hours this site has no operating history in."""
    hist = np.asarray(event["hour_histogram"], dtype=float)
    hours = np.asarray(event["hours"], dtype=int)
    n = len(hours)
    active = sorted(int(h) for h in np.flatnonzero(hist > HOUR_SUPPORT))

    recent_n = min(RECENT_WINDOW, n)
    if n - recent_n < MIN_PRIOR_FOR_HOUR:
        # Too little prior history for "never seen at this hour" to mean anything.
        return {"hour_anomaly": False, "offhours": [], "active_hours": active}

    recent = hours[-recent_n:]

    # Judged against prior history only. Scored against the whole record, a
    # burst of simultaneous pixels at 02:00 counts as evidence that 02:00 is
    # normal for the site, and the excursion vouches for itself.
    prior = hours[:-len(recent)] if n > len(recent) else hours
    prior_hours = set(int(h) for h in prior)
    offhours = sorted({int(h) for h in recent if h not in prior_hours})

    return {
        "hour_anomaly": bool(offhours) and deviation_mult > DEV_MULT_FOR_HOUR_FLAG,
        "offhours": offhours,
        "active_hours": active,
    }


def isolation_scores(events: list[dict]) -> np.ndarray:
    """Population-level outlier score in [0,1]; higher is more unusual."""
    if len(events) < 10:
        return np.zeros(len(events))

    X = np.array([[float(e.get(f, 0.0) or 0.0) for f in ISO_FEATURES]
                  for e in events], dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Xs = StandardScaler().fit_transform(X)

    forest = IsolationForest(contamination=ISO_CONTAMINATION, random_state=42,
                            n_estimators=200, n_jobs=-1)
    forest.fit(Xs)
    raw = -forest.score_samples(Xs)          # higher = more anomalous
    lo, hi = raw.min(), raw.max()
    return (raw - lo) / (hi - lo) if hi > lo else np.zeros(len(events))


def _describe(event: dict) -> str:
    """Plain-English account of why this event is or is not flagged.

    The three detectors are independent and the isolation forest does not need
    per-site history, so an event can legitimately be flagged with no baseline
    at all. Saying only "no baseline" on such an event reads as a contradiction
    next to an ANOMALY badge, so the reason given is always the detector that
    actually fired.
    """
    parts = []
    if event["z_anomaly"]:
        parts.append(
            f"FRP reached {event['current_frp']:.1f} MW against this site's own "
            f"baseline of {event['baseline_mean']:.1f} MW — "
            f"{event['deviation_mult']:.1f}x normal, z = {event['z_score']:+.1f}")
    if event["hour_anomaly"]:
        hrs = ", ".join(f"{h:02d}:00" for h in event["offhours"])
        parts.append(
            f"activity at {hrs} IST, hours in which this site has no prior "
            f"operating history")
    if event["iso_anomaly"]:
        if event["baseline_established"]:
            parts.append(
                f"behavioural profile is an outlier against the whole event "
                f"population (isolation score {event['iso_score']:.2f})")
        else:
            parts.append(
                f"behavioural profile is an outlier against the whole event "
                f"population (isolation score {event['iso_score']:.2f}); this is "
                f"a population comparison, not a departure from its own history, "
                f"which {event['detection_count']} detection(s) cannot yet establish")

    if parts:
        text = "; ".join(parts)
        return text[0].upper() + text[1:] + "."

    if not event["baseline_established"]:
        return (f"Not flagged. No operating baseline either: only "
                f"{event['detection_count']} detection(s), below the {MIN_HISTORY} "
                f"needed before this site's own history means anything.")

    return (f"Consistent with its own baseline of {event['baseline_mean']:.1f} MW "
            f"(z = {event['z_score']:+.1f}); no departure from normal operating "
            f"hours and not a population outlier.")


def run(events: list[dict], verbose: bool = True) -> list[dict]:
    if not events:
        return events

    iso = isolation_scores(events)

    for i, event in enumerate(events):
        event.update(z_baseline(event))
        event.update(hour_anomaly(event, event["deviation_mult"]))
        event["iso_score"] = round(float(iso[i]), 4)
        event["iso_anomaly"] = bool(iso[i] > ISO_THRESHOLD)

        event["anomaly_flag"] = bool(
            event["z_anomaly"]
            or (event["hour_anomaly"] and event["deviation_mult"] > DEV_MULT_FOR_HOUR_FLAG)
            or event["iso_anomaly"]
        )
        event["anomaly_desc"] = _describe(event)

    if verbose:
        n_z = sum(1 for e in events if e["z_anomaly"])
        n_h = sum(1 for e in events if e["hour_anomaly"])
        n_i = sum(1 for e in events if e["iso_anomaly"])
        n_a = sum(1 for e in events if e["anomaly_flag"])
        n_b = sum(1 for e in events if e["baseline_established"])
        print(f"  {n_b:,}/{len(events):,} events have enough history for a baseline")
        print(f"  z-score {n_z:,} | off-hours {n_h:,} | isolation forest {n_i:,}")
        print(f"  {n_a:,} events flagged anomalous overall")
    return events


if __name__ == "__main__":
    from pipeline.classify import run as classify_run
    from pipeline.cluster import run as cluster_run
    from pipeline.features import run as features_run
    run(classify_run(features_run(cluster_run())))
