"""
Spill-to-vessel attribution.

WHAT THIS DOES -- AND WHAT IT DELIBERATELY DOES NOT
---------------------------------------------------
This module correlates two independent model outputs -- a segmented spill from
the SAR net and per-ping anomaly flags from the AIS autoencoder -- against a
common place and time, and ranks the vessels that could account for the slick.

It does NOT hindcast. There is no ocean-current or wind-drift model anywhere in
this project, so the slick is treated as lying at or near where it was
observed. That assumption is sound for a fresh release from a stationary hull
and degrades badly as a slick ages: after a day of drift, the vessel that made
it may be tens of kilometres from the oil. The `drift_caveat` field carries
this to the operator on every single result rather than burying it in docs.

The ranking is a transparent weighted sum of four measurable quantities, not a
learned model. Every component is returned alongside the total so an analyst
can see precisely why a vessel ranked where it did, and disagree with it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .inference.trajectory import haversine_km

# Weights for the four evidence components. They sum to 1.0 and are exposed in
# the UI because an attribution weighting is a policy choice, not a fact.
WEIGHTS = {
    "proximity": 0.40,      # how close the vessel came to the observed slick
    "anomaly": 0.30,        # how anomalous its behaviour was while nearby
    "dwell": 0.20,          # how long it lingered in the area
    "deviation": 0.10,      # how far it departed from its predicted track
}

DEFAULT_SEARCH_RADIUS_KM = 15.0
DEFAULT_TIME_WINDOW_H = 6.0


def _proximity_score(min_distance_km: float, radius_km: float) -> float:
    """1.0 at the slick, decaying to 0 at the edge of the search radius."""
    if min_distance_km >= radius_km:
        return 0.0
    return float(np.clip(1.0 - (min_distance_km / radius_km) ** 0.5, 0.0, 1.0))


def _anomaly_score(max_score: float, threshold: float) -> float:
    """Saturating map of reconstruction error onto 0..1."""
    if max_score <= 0:
        return 0.0
    ratio = max_score / threshold
    return float(np.clip(np.log1p(ratio) / np.log1p(6.0), 0.0, 1.0))


def _dwell_score(minutes_in_radius: float, observed_minutes: float) -> float:
    """
    Fraction of the vessel's *observed* time that it spent inside the radius.

    Normalising against the nominal search window instead would punish every
    vessel for coverage gaps in the AIS feed rather than for its own behaviour:
    a vessel tracked for two hours inside a six-hour window could never score
    above a third however long it loitered at the scene.
    """
    if observed_minutes <= 0:
        return 0.0
    return float(np.clip(minutes_in_radius / observed_minutes, 0.0, 1.0))


def _deviation_score(max_deviation_km: Optional[float]) -> float:
    """Scaled against the trajectory model's p90 held-out error (0.63 km)."""
    if max_deviation_km is None or not np.isfinite(max_deviation_km):
        return 0.0
    return float(np.clip(max_deviation_km / (5 * 0.63), 0.0, 1.0))


@dataclass
class VesselAttribution:
    mmsi: int
    vessel_name: str
    flag: str
    vessel_type: str
    min_distance_km: float
    closest_time: Optional[datetime]
    minutes_in_radius: float
    pings_in_window: int
    flagged_pings: int
    max_anomaly_score: float
    max_deviation_km: Optional[float]
    components: Dict[str, float] = field(default_factory=dict)
    total_score: float = 0.0
    evidence: List[str] = field(default_factory=list)

    @property
    def confidence_band(self) -> str:
        if self.total_score >= 0.70:
            return "primary suspect"
        if self.total_score >= 0.45:
            return "person of interest"
        if self.total_score >= 0.20:
            return "in the area"
        return "cleared by proximity"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mmsi": self.mmsi,
            "vessel_name": self.vessel_name,
            "flag": self.flag,
            "vessel_type": self.vessel_type,
            "score": round(self.total_score, 4),
            "band": self.confidence_band,
            "min_distance_km": round(self.min_distance_km, 3),
            "minutes_in_radius": round(self.minutes_in_radius, 1),
            "flagged_pings": self.flagged_pings,
            "max_anomaly_score": round(self.max_anomaly_score, 4),
            "max_deviation_km": (round(self.max_deviation_km, 3)
                                 if self.max_deviation_km is not None else None),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "evidence": self.evidence,
        }


def attribute(
    scored_ais: pd.DataFrame,
    spill_lat: float,
    spill_lon: float,
    observed_at: Optional[datetime] = None,
    radius_km: float = DEFAULT_SEARCH_RADIUS_KM,
    window_hours: float = DEFAULT_TIME_WINDOW_H,
    threshold: float = 1.104481,
    deviations: Optional[Dict[int, float]] = None,
) -> List[VesselAttribution]:
    """
    Rank vessels against an observed slick position.

    `scored_ais` is the output of `inference.ais.score_frame`. `deviations` maps
    MMSI to that vessel's worst trajectory-prediction deviation, when the
    trajectory model has been run over the same tracks.
    """
    if scored_ais.empty:
        return []

    df = scored_ais.copy()
    has_time = "timestamp" in df.columns and observed_at is not None
    if has_time:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        anchor = pd.Timestamp(observed_at)
        if anchor.tzinfo is None:
            anchor = anchor.tz_localize("UTC")
        half = pd.Timedelta(hours=window_hours)
        df = df[(ts >= anchor - half) & (ts <= anchor + half)].copy()
        if df.empty:
            return []

    df["distance_km"] = [
        haversine_km(spill_lat, spill_lon, la, lo)
        for la, lo in zip(df["latitude"], df["longitude"])
    ]

    results: List[VesselAttribution] = []

    for mmsi, group in df.groupby("mmsi"):
        near = group[group["distance_km"] <= radius_km]
        min_dist = float(group["distance_km"].min())

        closest_row = group.loc[group["distance_km"].idxmin()]
        closest_time = (pd.to_datetime(closest_row["timestamp"]).to_pydatetime()
                        if "timestamp" in group.columns else None)

        # Dwell time, measured from the actual ping cadence rather than assumed.
        if len(group) > 1 and "timestamp" in group.columns:
            gts = pd.to_datetime(group["timestamp"]).sort_values()
            cadence_s = gts.diff().dt.total_seconds().median()
            cadence_s = cadence_s if np.isfinite(cadence_s) else 60.0
            dwell_min = len(near) * cadence_s / 60.0
            observed_min = len(group) * cadence_s / 60.0
        else:
            dwell_min, observed_min = float(len(near)), float(len(group))

        # Anomalies only count as evidence if they happened near the slick.
        anomaly_source = near if len(near) else group.iloc[0:0]
        max_anom = float(anomaly_source["anomaly_score"].max()) if len(anomaly_source) else 0.0
        flagged = int(anomaly_source["is_anomaly"].sum()) if len(anomaly_source) else 0

        dev = deviations.get(int(mmsi)) if deviations else None

        components = {
            "proximity": _proximity_score(min_dist, radius_km),
            "anomaly": _anomaly_score(max_anom, threshold),
            "dwell": _dwell_score(dwell_min, observed_min),
            "deviation": _deviation_score(dev),
        }
        total = sum(WEIGHTS[k] * v for k, v in components.items())

        att = VesselAttribution(
            mmsi=int(mmsi),
            vessel_name=str(group["vessel_name"].iloc[0]) if "vessel_name" in group else str(mmsi),
            flag=str(group["flag"].iloc[0]) if "flag" in group else "-",
            vessel_type=str(group["vessel_type"].iloc[0]) if "vessel_type" in group else "-",
            min_distance_km=min_dist,
            closest_time=closest_time,
            minutes_in_radius=dwell_min,
            pings_in_window=len(group),
            flagged_pings=flagged,
            max_anomaly_score=max_anom,
            max_deviation_km=dev,
            components=components,
            total_score=total,
        )
        att.evidence = _build_evidence(att, radius_km)
        results.append(att)

    results.sort(key=lambda a: a.total_score, reverse=True)
    return results


def _build_evidence(att: VesselAttribution, radius_km: float) -> List[str]:
    """Plain-language reasons, so the ranking is auditable rather than oracular."""
    ev: List[str] = []

    if att.min_distance_km < 0.5:
        ev.append(f"Came within {att.min_distance_km * 1000:.0f} m of the observed slick.")
    elif att.min_distance_km <= radius_km:
        ev.append(f"Closest approach {att.min_distance_km:.1f} km.")
    else:
        ev.append(f"Never came closer than {att.min_distance_km:.1f} km "
                  f"(outside the {radius_km:.0f} km search radius).")

    if att.flagged_pings:
        ev.append(f"{att.flagged_pings} AIS ping(s) flagged anomalous near the slick "
                  f"(peak reconstruction error {att.max_anomaly_score:.2f}).")
    else:
        ev.append("No anomalous AIS behaviour detected in the search area.")

    if att.minutes_in_radius >= 30:
        ev.append(f"Remained in the area for about {att.minutes_in_radius:.0f} minutes.")

    if att.max_deviation_km is not None and att.max_deviation_km > 0.63:
        ev.append(f"Departed from its predicted track by up to "
                  f"{att.max_deviation_km:.2f} km, above the model's 90th-percentile error.")

    return ev


def drift_caveat(observed_at: Optional[datetime] = None,
                 released_at: Optional[datetime] = None) -> str:
    """
    The standing limitation on every attribution this module produces.

    Stated in terms of elapsed time when we know it, because that is what
    governs how far the oil could have moved.
    """
    base = ("Attribution assumes the slick lies where it was observed. No ocean-current "
            "or wind-drift model is applied, so a slick that has been on the water for "
            "some time may have moved far from its source.")
    if observed_at and released_at:
        hours = abs((observed_at - released_at).total_seconds()) / 3600.0
        if hours >= 1:
            base += (f" Roughly {hours:.0f} h separate release from observation here; "
                     f"at a typical 3% wind factor that is easily tens of kilometres "
                     f"of uncertainty.")
    return base


def summarise(results: List[VesselAttribution]) -> Dict[str, Any]:
    """Headline framing for the incident view."""
    if not results:
        return {"verdict": "No AIS traffic in the search window.", "top": None}

    top = results[0]
    runner = results[1] if len(results) > 1 else None
    margin = (top.total_score - runner.total_score) if runner else top.total_score

    if top.total_score < 0.45:
        verdict = ("No vessel in the search window accounts for this slick with any "
                   "confidence. Widen the radius or the time window.")
    elif margin >= 0.20:
        verdict = (f"{top.vessel_name} separates clearly from all other traffic in the "
                   f"window on proximity and behaviour.")
    else:
        verdict = (f"{top.vessel_name} ranks highest, but {runner.vessel_name} is close "
                   f"behind -- treat this as unresolved between them.")

    return {
        "verdict": verdict,
        "top": top.as_dict(),
        "margin": round(margin, 4),
        "candidates": len(results),
    }
