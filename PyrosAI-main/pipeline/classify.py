"""Rule-based scoring classifier (Phase 2).

Phase 3 swaps this for a Random Forest + XGBoost ensemble with SHAP. The class
list, the output schema and the refuse-rather-than-guess contract stay
identical, so only this file changes.

Rules are *scored*, not branched. Each class owns weighted criteria; a class
score is the sum of the criteria it satisfies, divided by the most that class
could possibly score. That per-class normalisation matters: the gas flare rules
can award 1.05 points and the agricultural rules only 0.60, so without it an
agricultural burn could satisfy every rule it has and still lose to a flare that
satisfied half of its own. Normalising makes "how much of your own case did you
prove" the comparable quantity.

Evidence strings are generated from the same criteria that produce the score,
so the explanation cannot drift away from the decision.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INDUSTRIAL = "Industrial Fire"
GAS_FLARE = "Gas Flare"
WILDFIRE = "Wildfire"
AGRICULTURAL = "Agricultural Burn"
INSUFFICIENT = "Insufficient Evidence"

CLASS_COLORS = {
    INDUSTRIAL: "#ef4444",
    GAS_FLARE: "#3b82f6",
    WILDFIRE: "#f97316",
    AGRICULTURAL: "#eab308",
    INSUFFICIENT: "#64748b",
}
CLASSES = [INDUSTRIAL, GAS_FLARE, WILDFIRE, AGRICULTURAL]

MIN_SCORE_TO_ASSERT = 0.30
MIN_CONFIDENCE_TO_ASSERT = 0.45

# Behavioural claims need a window long enough to observe the behaviour.
# "Persistent" measured over a single day is 1.0 by construction
# (unique_dates / date_range = 1/1), and "no weekly shutdown" cannot be
# asserted without seeing more than one weekend. Without these floors a single
# hot pixel at 02:00 on a Sunday scores a perfect 1.00 as a gas flare — which
# is exactly what the World Bank registry cross-check caught: 289 of 341 flare
# calls were single detections, and only 3 were near a documented flare.
MIN_SPAN_FOR_WEEKLY = 14          # days: at least two weekends
MIN_DETECTIONS_FOR_RHYTHM = 4     # passes before a day/night ratio is a rhythm
MIN_ACTIVE_DAYS_RECURRING = 8     # separate days before a source is "recurring"
MIN_ACTIVE_DAYS_STEADY = 5

# Thresholds below are calibrated against the real FIRMS distribution for this
# region, not against the synthetic Phase-1 data. That matters: the persistent
# night-burning sources actually present here run persistence 0.26-0.52 and
# robust FRP variation 0.44-0.82, because cloud cover and overpass geometry
# make real detection sparse and noisy. The original >0.70 persistence and
# <0.20 variation gates describe an idealised flare that real VIIRS data over a
# monsoon-season window never produces.
STEADY_ROBUST_VAR = 0.70
NIGHT_SHARE_MIN = 0.35
WEEKEND_SHARE_MIN = 0.15

MAX_ACTIVE_DAYS_AGRI = 3          # stubble burning is over within days
MAX_SPAN_AGRI = 5
AGRI_FRP_MAX = 5.0                # regional median FRP is ~2 MW, not ~15


def _flare_rules(f: dict) -> list[tuple]:
    """Continuous combustion: always on, all hours, same size every time."""
    night, weekend = f["night_share"], f["weekend_share"]
    rvar, pers = f["frp_robust_var"], f["persistence"]
    span, count, days = f["date_range"], f["detection_count"], f["active_days"]
    return [
        # Recurrence is counted in separate days observed, not as a ratio.
        # active_days/date_range is 1.0 for a single detection on a single day,
        # which is how one hot pixel used to score a perfect flare.
        ("recurring", 0.30, days >= MIN_ACTIVE_DAYS_RECURRING,
         f"detected on {days} separate days across a {span}-day window"),
        ("night_burning", 0.30,
         night > NIGHT_SHARE_MIN and count >= MIN_DETECTIONS_FOR_RHYTHM,
         f"burns through the night ({night:.0%} of {count} detections between "
         f"20:00 and 07:00 IST)"),
        ("no_weekly_cycle", 0.25,
         weekend > WEEKEND_SHARE_MIN and span >= MIN_SPAN_FOR_WEEKLY,
         f"no weekly shutdown ({weekend:.0%} of detections fall on weekends, "
         f"observed across {span} days)"),
        ("steady_output", 0.15,
         rvar < STEADY_ROBUST_VAR and days >= MIN_ACTIVE_DAYS_STEADY,
         f"output steady for a sparse thermal source (robust FRP variation "
         f"{rvar:.2f} over {days} active days)"),
    ]


def _industrial_rules(f: dict) -> list[tuple]:
    """Shift-based plant: on a mapped site, daytime, weekdays, fixed footprint."""
    dist = f["dist_to_facility_m"]
    weight = f["facility_risk_weight"]
    night, weekend, spread = f["night_share"], f["weekend_share"], f["spread_rate"]
    fac = f.get("facility_name") or "a mapped industrial site"

    # Proximity is scaled by what kind of site it is: 200 m from a refinery is
    # a very different prior from 200 m from a warehouse.
    near = dist < 1000
    prox_points = 0.40 * min(weight / 5.0, 1.0) if near else 0.0
    return [
        ("on_industrial_site", 0.40, near,
         f"{dist:.0f} m from {fac} ({(f.get('facility_type') or 'industrial').replace('_', ' ')}, "
         f"risk weight {weight:.1f})", prox_points),
        # Two legitimate industrial signatures, not one. Shift-based plants run
        # weekday daylight hours; continuous-process plants — cement kilns,
        # steel furnaces, mine operations — run around the clock and never shut
        # down at weekends. Treating only the first as industrial is what
        # pushed every 24/7 plant into the Gas Flare class.
        ("operating_pattern", 0.25,
         (night < 0.30 and weekend < 0.25) or (night > 0.35 and weekend > 0.15
                                               and f["active_days"] >= 8),
         (f"weekday daytime operating pattern ({night:.0%} night, {weekend:.0%} weekend)"
          if night < 0.30 else
          f"continuous round-the-clock operation ({night:.0%} night, "
          f"{weekend:.0%} weekend, {f['active_days']} active days)")),
        ("fixed_footprint", 0.15, abs(spread) < 0.05,
         f"fixed footprint, no measurable spread ({spread:+.2f} km/day)"),
    ]


def _wildfire_rules(f: dict) -> list[tuple]:
    """Uncontrolled vegetation fire: spreading, hot, erratic, away from industry."""
    spread, dist = f["spread_rate"], f["dist_to_facility_m"]
    peak, rvar = f["frp_peak"], f["frp_robust_var"]
    return [
        ("spreading", 0.45, spread > 0.30,
         f"footprint growing {spread:.2f} km/day across consecutive passes"),
        # No landcover layer in Phase 2, so "in vegetation" is inferred from
        # distance to mapped industry. The CNN layer and Phase 3 land cover
        # replace this proxy with an actual observation.
        ("away_from_industry", 0.30, dist > 3000,
         f"{dist / 1000:.1f} km from the nearest mapped industrial site"),
        ("hot_and_erratic", 0.20, peak > 20 and rvar > 0.40,
         f"high and erratic intensity (peak {peak:.0f} MW, robust variation {rvar:.2f})"),
    ]


def _agricultural_rules(f: dict) -> list[tuple]:
    """Stubble burning: brief, cool, compact, seasonal."""
    mean_frp, night = f["frp_mean"], f["night_share"]
    spread, count = f["spread_rate"], f["detection_count"]
    days, span = f["active_days"], f["date_range"]
    neighbours = f.get("neighbour_count", 0)
    month = _peak_month(f)
    return [
        # Stubble burning is over within a day or two. The previous test,
        # persistence < 0.50 and frp_mean < 15, was satisfied by almost every
        # real event in the region — FRP here runs 1-7 MW, so the intensity
        # bound never bit — and a source detected on 26 separate nights was
        # being called a crop fire.
        ("brief", 0.35, days <= MAX_ACTIVE_DAYS_AGRI and span <= MAX_SPAN_AGRI,
         f"burned on {days} day(s) within a {span}-day span, then stopped"),
        ("daytime_low_intensity", 0.25,
         night < 0.25 and mean_frp < AGRI_FRP_MAX,
         f"daytime burn at low intensity ({night:.0%} night, mean FRP {mean_frp:.1f} MW)"),
        ("clustered", 0.20, neighbours >= 1,
         f"{neighbours} other thermal event(s) within 5 km, consistent with "
         f"coordinated field burning"),
        ("burn_season", 0.10, month in (9, 10, 11),
         "falls in the September-November crop residue burning season"),
        ("compact", 0.10, abs(spread) < 0.05 and f["footprint_km"] < 2.0,
         f"compact footprint ({f['footprint_km']:.2f} km), no spread"),
    ]


def _evidence_factor(f: dict) -> float:
    """How much observation stands behind the call. Scales confidence, not score.

    The distinction matters. Score answers "how well does this event fit the
    class", and evidence volume must not enter it — brevity is the whole
    signature of an agricultural burn, so penalising short observation there
    would distort the taxonomy. Confidence answers "how much should anyone
    trust this", and observation volume is exactly what that depends on.

    Keeping them separate is what stopped 369 single-pixel events being
    reported as Agricultural Burn at 97% confidence.
    """
    count = float(f.get("detection_count", 0))
    return 0.55 + 0.45 * min(max((count - 1.0) / 4.0, 0.0), 1.0)   # 1 -> .55, 5+ -> 1


def _industrial_prior(f: dict) -> float:
    """Distance decay applied to the Industrial Fire score.

    Without this the class is asserted on generic evidence. "Weekday daytime"
    and "fixed footprint" describe almost any small, short-lived detection, and
    together they normalise to 0.50 - enough to win outright with no industrial
    evidence whatsoever. On the live data that put 263 of 295 Industrial Fire
    calls more than a kilometre from any mapped facility, the worst 45 km out.

    Calling a fire industrial is a claim that it is happening at an industrial
    site, so proximity is treated as necessary evidence rather than as one
    optional criterion among three.
    """
    dist = f["dist_to_facility_m"]
    if dist < 1000:
        return 1.00
    if dist < 3000:
        return 0.55
    if dist < 10000:
        return 0.30
    return 0.12


def _peak_month(f: dict) -> int:
    try:
        return datetime.strptime(f["last_seen"][:10], "%Y-%m-%d").month
    except (ValueError, KeyError, TypeError):
        return 0


# Multiplicative priors, applied after a class scores its own criteria.
# OSM facility types that make a gas-flare reading plausible, and those that
# make it far less so.
HYDROCARBON_TYPES = {"oil_refinery", "petroleum", "lng", "gas", "chemical"}
HEAVY_INDUSTRY_TYPES = {"cement", "steel", "mine", "quarry", "factory",
                        "power_plant", "warehouse"}


def _flare_prior(f: dict) -> float:
    """Spatial context for a gas-flare claim.

    The registry cross-check showed every flare call sitting 14-424 km from any
    documented gas field and 0.1-2.9 km from a cement plant, steel works or
    mine. Continuous night-and-weekend burning is characteristic of
    continuous-process industry in general, not of gas flaring specifically, so
    temporal behaviour alone cannot separate a flare stack from a kiln. What
    separates them is what is underneath.
    """
    ftype = (f.get("facility_type") or "").lower()
    dist = f["dist_to_facility_m"]
    if ftype in HYDROCARBON_TYPES and dist < 3000:
        return 1.0                       # oil & gas context: supported
    if ftype in HEAVY_INDUSTRY_TYPES and dist < 1500:
        return 0.45                      # sitting on a kiln or furnace instead
    return 0.80                          # no informative context either way


def _agricultural_prior(f: dict) -> float:
    """Stubble burning happens in daylight; farmers do not burn fields at 2am.

    Without this, a night-dominant event could still satisfy "brief" plus
    "burn season" plus "compact" and win the class outright — 65 events with a
    night share above 0.5, several at 1.00, were being labelled crop fires.
    """
    return 0.40 if f["night_share"] > 0.50 else 1.0


def _wildfire_prior(f: dict) -> float:
    """A wildfire claim needs actual fire evidence, not just rural surroundings.

    "Away from mapped industry" normalises to 0.32 by itself, which was enough
    to win outright for any isolated rural pixel. Being 5 km from a factory is
    not evidence of a spreading fire, so the class is scaled down unless
    something positive — measurable spread or real intensity — supports it.
    """
    if f["spread_rate"] > 0.10 or f["frp_peak"] > 15.0:
        return 1.0
    return 0.35


CLASS_PRIORS = {INDUSTRIAL: _industrial_prior, WILDFIRE: _wildfire_prior,
                GAS_FLARE: _flare_prior, AGRICULTURAL: _agricultural_prior}

RULES = {
    GAS_FLARE: _flare_rules,
    INDUSTRIAL: _industrial_rules,
    WILDFIRE: _wildfire_rules,
    AGRICULTURAL: _agricultural_rules,
}


def _score(rules: list[tuple]) -> tuple[float, list[str]]:
    """Sum satisfied criteria, normalised by the most this class could score."""
    earned = 0.0
    possible = 0.0
    evidence = []
    for rule in rules:
        key, points, passed, phrase = rule[0], rule[1], rule[2], rule[3]
        graded = rule[4] if len(rule) > 4 else (points if passed else 0.0)
        possible += points
        if passed:
            earned += graded
            evidence.append(phrase)
    return (earned / possible if possible else 0.0), evidence


def classify_event(f: dict) -> dict:
    scored = {}
    for name, rule in RULES.items():
        raw, evidence = _score(rule(f))
        prior = CLASS_PRIORS.get(name, lambda _f: 1.0)(f)
        scored[name] = (raw * prior, evidence)
    scores = {name: round(val, 4) for name, (val, _) in scored.items()}

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, best_score), (_, runner_score) = ranked[0], ranked[1]
    margin = best_score - runner_score

    # The specified formula, 0.5 + margin*1.5, has a floor of 0.5 because the
    # margin is never negative - so its own "confidence < 0.45" refusal test
    # could never fire. Separation alone is also the wrong measure: winning a
    # weak field by a hair is not the same as proving a case. The separation
    # term is kept and scaled by how much of the winning class's case was
    # actually made, which puts the refusal path back in play.
    separation = 0.5 + margin * 1.5
    completeness = 0.45 + 0.55 * best_score
    confidence = min(0.97, separation * completeness * _evidence_factor(f))
    evidence = scored[best][1]

    if best_score < MIN_SCORE_TO_ASSERT or confidence < MIN_CONFIDENCE_TO_ASSERT:
        classification = INSUFFICIENT
        confidence = round(confidence * 0.6, 3)
        reason = (f"Evidence is too weak or too evenly split to name a type: "
                  f"best candidate {best} scored {best_score:.2f} with only "
                  f"{margin:.2f} separating it from the runner-up. "
                  + ("Observed: " + "; ".join(evidence[:2]) + "." if evidence
                     else "No criterion for any class was satisfied."))
    else:
        classification = best
        confidence = round(confidence, 3)
        reason = (evidence[0][0].upper() + evidence[0][1:] + "; "
                  + "; ".join(evidence[1:]) + ".") if len(evidence) > 1 else (
                  evidence[0][0].upper() + evidence[0][1:] + "." if evidence else
                  f"Classified {best} on aggregate score {best_score:.2f}.")

    scores[INSUFFICIENT] = round(1.0 - best_score, 4)

    return {
        "classification": classification,
        "confidence": confidence,
        "class_scores": scores,
        "best_candidate": best,
        "margin": round(margin, 4),
        "evidence": evidence,
        "reason": reason,
        "type_color": CLASS_COLORS[classification],
    }


def run(events: list[dict], verbose: bool = True) -> list[dict]:
    for event in events:
        event.update(classify_event(event))
    if verbose:
        counts: dict[str, int] = {}
        for e in events:
            counts[e["classification"]] = counts.get(e["classification"], 0) + 1
        for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"    {name:<22} {n:>5,}")
    return events


if __name__ == "__main__":
    from pipeline.cluster import run as cluster_run
    from pipeline.features import run as features_run
    run(features_run(cluster_run()))
