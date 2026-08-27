"""
Reconstructed AIS scenario: the MV Wakashio grounding, Mauritius, 25 July 2020.

WHAT THIS IS
------------
No real AIS feed was supplied with this project, and the trajectory model is
normalised to a Mauritius bounding box for July 2020 -- so the demonstration
scenario is rebuilt over that same water, around the incident the project brief
uses as its case study.

The vessel particulars, the voyage, the grounding position, the timing and the
causal account below are all drawn from the public record of the casualty. The
per-second kinematics -- the individual AIS pings -- are a physically
consistent *reconstruction*, not recovered signal. Tracks are generated from
waypoints with a speed profile, and course, rate-of-turn and the positional
deltas are all derived from the resulting geometry, so every field the models
consume agrees with every other one.

Everything the UI renders from this module is labelled as reconstructed. It
exists to exercise the pipeline end to end; it is not evidence, and no number
produced from it should be quoted as a measurement of the real casualty.

Sources for the documented facts: Panama Maritime Authority casualty
investigation, Mauritius National Crisis Committee statements, and ITOPF /
IMO incident reporting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0
KN_TO_KMH = 1.852

# Below this speed a hull has no steerage way, so course over ground stops being
# a measured quantity. Paired with a displacement floor comfortably above the
# metres-scale positional dither applied to every fix.
STEERAGE_SPEED_KN = 0.5
STEERAGE_MIN_DISPLACEMENT_M = 12.0

# --------------------------------------------------------------------------
# Documented incident facts
# --------------------------------------------------------------------------
GROUNDING_LAT = -20.4372
GROUNDING_LON = 57.7433
GROUNDING_UTC = datetime(2020, 7, 25, 15, 25, tzinfo=timezone.utc)   # 19:25 local (UTC+4)

INCIDENT: Dict[str, Any] = {
    "name": "MV Wakashio grounding",
    "location": "Pointe d'Esny reef, south-east Mauritius",
    "grounding_utc": GROUNDING_UTC,
    "grounding_local": "25 July 2020, 19:25 (UTC+4)",
    "position": (GROUNDING_LAT, GROUNDING_LON),
    "vessel": {
        "name": "MV WAKASHIO",
        "mmsi": 371284000,
        "imo": 9337119,
        "flag": "Panama",
        "type": "Bulk carrier (Capesize)",
        "length_m": 203,
        "dwt": 101932,
        "owner": "Nagashiki Shipping (Japan)",
    },
    "voyage": "Lianyungang, China -> Tubarao, Brazil, in ballast",
    "bunkers_t": 3894,
    "diesel_t": 207,
    "oil_released_t": 1000,
    "leak_began": "6 August 2020",
    "hull_failure": "Vessel broke in two, 15 August 2020",
    "cause_summary": (
        "The vessel altered course toward the coast to obtain mobile telephone "
        "signal, closed the reef at full service speed with no effective "
        "navigational watch, and struck it at approximately 11 knots without "
        "any avoiding action being taken."
    ),
    "why_it_matters": (
        "The behavioural signature -- a sustained inshore deviation followed by an "
        "abrupt loss of way -- is exactly what an AIS anomaly detector is meant to "
        "surface, and it precedes the visible spill by twelve days."
    ),
}


def grounding_point() -> Tuple[float, float]:
    return GROUNDING_LAT, GROUNDING_LON


# --------------------------------------------------------------------------
# Geodesy
# --------------------------------------------------------------------------
def destination_point(lat: float, lon: float, bearing_deg: float, distance_km: float):
    """Forward geodesic: where you end up steering `bearing` for `distance`."""
    br = np.radians(bearing_deg)
    ang = distance_km / EARTH_RADIUS_KM
    lat1, lon1 = np.radians(lat), np.radians(lon)
    lat2 = np.arcsin(np.sin(lat1) * np.cos(ang) + np.cos(lat1) * np.sin(ang) * np.cos(br))
    lon2 = lon1 + np.arctan2(
        np.sin(br) * np.sin(ang) * np.cos(lat1),
        np.cos(ang) - np.sin(lat1) * np.sin(lat2),
    )
    return float(np.degrees(lat2)), float((np.degrees(lon2) + 540) % 360 - 180)


def bearing_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return float((np.degrees(np.arctan2(y, x)) + 360.0) % 360.0)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


# --------------------------------------------------------------------------
# Track construction
# --------------------------------------------------------------------------
@dataclass
class Leg:
    """One steered segment of a passage."""
    bearing: float
    minutes: float
    speed_kn: float
    status: int = 0
    label: str = ""


@dataclass
class VesselSpec:
    mmsi: int
    name: str
    flag: str
    vessel_type: str
    start: Tuple[float, float]
    legs: List[Leg]
    length_m: int = 180
    role: str = "background"
    course_jitter: float = 0.8
    speed_jitter: float = 0.12
    note: str = ""
    aground_after: bool = False
    aground_minutes: float = 0.0
    tags: List[str] = field(default_factory=list)


def _build_positions(spec: VesselSpec, interval_s: float) -> List[Dict[str, Any]]:
    """Walk the legs, emitting one raw position fix per ping interval."""
    lat, lon = spec.start
    fixes: List[Dict[str, Any]] = []
    step_h = interval_s / 3600.0

    for leg in spec.legs:
        n = max(1, int(round(leg.minutes * 60.0 / interval_s)))
        for _ in range(n):
            fixes.append({"latitude": lat, "longitude": lon,
                          "speed": leg.speed_kn, "status": leg.status,
                          "phase": leg.label})
            lat, lon = destination_point(lat, lon, leg.bearing, leg.speed_kn * KN_TO_KMH * step_h)

    if spec.aground_after:
        n = max(1, int(round(spec.aground_minutes * 60.0 / interval_s)))
        for _ in range(n):
            fixes.append({"latitude": lat, "longitude": lon,
                          "speed": 0.0, "status": 6, "phase": "aground"})
    return fixes


def _to_frame(spec: VesselSpec, fixes: List[Dict[str, Any]],
              start_utc: datetime, interval_s: float, rng: np.random.Generator) -> pd.DataFrame:
    """
    Turn raw fixes into an AIS frame.

    Course and rate-of-turn are derived from the realised geometry rather than
    declared independently, so the kinematic fields cannot contradict the
    positions the models also consume.
    """
    df = pd.DataFrame(fixes)
    n = len(df)
    df["mmsi"] = spec.mmsi
    df["vessel_name"] = spec.name
    df["flag"] = spec.flag
    df["vessel_type"] = spec.vessel_type
    df["timestamp"] = [start_utc + timedelta(seconds=interval_s * i) for i in range(n)]

    # Receiver noise: metres-scale positional dither, small speed error.
    df["latitude"] += rng.normal(0, 3e-5, n)
    df["longitude"] += rng.normal(0, 3e-5, n)
    df["speed"] = np.clip(df["speed"] + rng.normal(0, spec.speed_jitter, n), 0.0, None)
    df.loc[df["speed"] < 0.15, "speed"] = 0.0

    # Course over ground from the realised track. Below steerage way the fix-to-fix
    # displacement is receiver dither rather than motion, and deriving a bearing
    # from it would manufacture a vessel spinning on its anchor; real AIS holds the
    # last steerage course instead. The final fix inherits the previous course
    # because there is no next position to measure against.
    courses: List[float] = []
    for i in range(n):
        j = min(i + 1, n - 1)
        moved_m = 0.0 if i == j else haversine_km(
            df.latitude[i], df.longitude[i], df.latitude[j], df.longitude[j]) * 1000.0
        if moved_m < STEERAGE_MIN_DISPLACEMENT_M:
            courses.append(courses[-1] if courses else spec.legs[0].bearing)
        else:
            courses.append(bearing_between(df.latitude[i], df.longitude[i],
                                           df.latitude[j], df.longitude[j]))

    # Heading noise only applies while the vessel is actually making way.
    under_way = df["speed"].to_numpy() >= STEERAGE_SPEED_KN
    course = np.array(courses) + rng.normal(0, spec.course_jitter, n) * under_way
    df["course"] = course % 360.0

    # Rate of turn, degrees/min, from the shortest signed course change.
    delta = ((np.diff(df["course"].to_numpy(), prepend=df["course"].iloc[0]) + 180) % 360) - 180
    df["rot"] = np.clip(delta * (60.0 / interval_s), -127, 127).round(0)

    df["msg_type"] = 1
    df["accuracy"] = 1
    df["heading"] = df["course"].round(0) % 360
    return df


def _apply_grounding_signature(df: pd.DataFrame, interval_s: float) -> pd.DataFrame:
    """
    Impose the physics of striking a reef at service speed.

    A 100,000 DWT hull that grounds does not decelerate smoothly: way is lost
    over a couple of ship-lengths, the head swings as the bow bites and the
    stern continues, and the vessel then reports aground and stops reporting
    meaningful course over ground.
    """
    idx = df.index[df["phase"] == "aground"]
    if len(idx) == 0:
        return df
    impact = idx[0]

    # Deceleration across the two pings spanning impact.
    if impact - 1 in df.index:
        df.loc[impact - 1, "speed"] = 6.4
        df.loc[impact - 1, "rot"] = -68.0
        df.loc[impact - 1, "course"] = (df.loc[impact - 1, "course"] - 9.0) % 360

    df.loc[impact, "speed"] = 0.4
    df.loc[impact, "rot"] = 127.0
    df.loc[impact, "course"] = (df.loc[impact, "course"] + 34.0) % 360
    df.loc[impact, "status"] = 6

    # Settled on the reef: no way on, heading pinned, occasional swell-driven yaw.
    after = idx[1:]
    if len(after):
        df.loc[after, "speed"] = 0.0
        df.loc[after, "status"] = 6
        pinned = float(df.loc[impact, "course"])
        df.loc[after, "course"] = pinned
        df.loc[after, "rot"] = 0.0
        if len(after) > 1:
            df.loc[after[0], "rot"] = -127.0      # head snaps back after the strike
    return df


def build_track(spec: VesselSpec, start_utc: datetime, interval_s: float,
                rng: np.random.Generator) -> pd.DataFrame:
    fixes = _build_positions(spec, interval_s)
    df = _to_frame(spec, fixes, start_utc, interval_s, rng)
    if spec.aground_after:
        df = _apply_grounding_signature(df, interval_s)
    return df


# --------------------------------------------------------------------------
# The scenario itself
# --------------------------------------------------------------------------
def _wakashio_spec(interval_s: float) -> VesselSpec:
    """
    Back-solve the approach so the track terminates exactly on the reef.

    Two legs: the planned passage that would have cleared Mauritius to the
    south, then the inshore deviation that put the vessel on the reef.
    """
    transit_min, deviation_min = 50.0, 40.0
    transit_brg, deviation_brg = 236.0, 248.0
    speed = 11.2

    dev_km = speed * KN_TO_KMH * deviation_min / 60.0
    tr_km = speed * KN_TO_KMH * transit_min / 60.0

    turn_lat, turn_lon = destination_point(GROUNDING_LAT, GROUNDING_LON,
                                           (deviation_brg + 180) % 360, dev_km)
    start_lat, start_lon = destination_point(turn_lat, turn_lon,
                                             (transit_brg + 180) % 360, tr_km)

    return VesselSpec(
        mmsi=371284000,
        name="MV WAKASHIO",
        flag="Panama",
        vessel_type="Bulk carrier",
        length_m=203,
        start=(start_lat, start_lon),
        role="suspect",
        course_jitter=0.5,
        speed_jitter=0.10,
        legs=[
            Leg(transit_brg, transit_min, speed, 0, "planned passage"),
            Leg(deviation_brg, deviation_min, speed, 0, "inshore deviation"),
        ],
        aground_after=True,
        aground_minutes=35.0,
        note=("Altered course inshore, then struck the reef at service speed "
              "with no avoiding action."),
        tags=["deviation", "grounding"],
    )


def vessel_registry(interval_s: float = 60.0) -> List[VesselSpec]:
    """
    The Wakashio plus contemporaneous traffic in the same AOI.

    The background vessels are what make the demonstration honest: attribution
    is only meaningful if the detector had innocent traffic available to blame
    and did not blame it.
    """
    return [
        _wakashio_spec(interval_s),

        VesselSpec(
            mmsi=419002731, name="MT KAVERI PRIDE", flag="India",
            vessel_type="Products tanker", length_m=183,
            start=(-20.1450, 58.3200), role="background",
            legs=[Leg(228.0, 125.0, 12.6, 0, "transit")],
            note="Routine south-west transit through the outer lane.",
        ),

        VesselSpec(
            mmsi=256891004, name="MSC CAP FLORES", flag="Malta",
            vessel_type="Container ship", length_m=294,
            start=(-20.5300, 57.8100), role="background",
            legs=[Leg(58.0, 125.0, 15.4, 0, "transit")],
            note="North-east bound container service, clear of the reef line.",
        ),

        VesselSpec(
            mmsi=563114900, name="BULK PIONEER", flag="Singapore",
            vessel_type="Bulk carrier", length_m=190,
            start=(-20.2100, 58.2600), role="background",
            legs=[
                Leg(232.0, 70.0, 10.8, 0, "transit"),
                Leg(244.0, 55.0, 10.4, 0, "transit"),
            ],
            note="Comparable bulk carrier on a comparable heading -- the control case.",
        ),

        VesselSpec(
            mmsi=645079210, name="FV SEA HARVESTER 7", flag="Mauritius",
            vessel_type="Fishing vessel", length_m=24,
            start=(-20.4750, 57.9400), role="background",
            course_jitter=3.0, speed_jitter=0.5,
            legs=[
                Leg(120.0, 14.0, 4.2, 7, "working"),
                Leg(15.0, 12.0, 3.4, 7, "working"),
                Leg(250.0, 13.0, 4.6, 7, "working"),
                Leg(340.0, 11.0, 2.8, 7, "working"),
                Leg(95.0, 15.0, 4.0, 7, "working"),
                Leg(200.0, 13.0, 3.6, 7, "working"),
                Leg(30.0, 12.0, 4.4, 7, "working"),
                Leg(265.0, 15.0, 3.2, 7, "working"),
                Leg(150.0, 20.0, 4.1, 7, "working"),
            ],
            note=("Trawling inshore on tight repeated turns. Kinematically erratic "
                  "by design -- the honest false-positive case for this detector."),
            tags=["erratic"],
        ),

        VesselSpec(
            mmsi=352001899, name="MV BLUE BAY TRADER", flag="Panama",
            vessel_type="General cargo", length_m=142,
            start=(-20.2650, 57.8300), role="background",
            legs=[
                Leg(196.0, 40.0, 8.2, 0, "approach"),
                Leg(196.0, 85.0, 0.0, 1, "at anchor"),
            ],
            note="Anchored awaiting a berth -- stationary but entirely normal.",
        ),
    ]


def build_scenario(interval_s: float = 60.0, seed: int = 20200725,
                   lead_minutes: float = 0.0) -> Dict[str, Any]:
    """
    Generate the full scenario.

    All tracks share a clock anchored so that the Wakashio's grounding falls at
    the documented 15:25 UTC, which lets the whole AOI be replayed against a
    single incident timeline.
    """
    rng = np.random.default_rng(seed)
    specs = vessel_registry(interval_s)

    wakashio = specs[0]
    underway_min = sum(leg.minutes for leg in wakashio.legs)
    start_utc = GROUNDING_UTC - timedelta(minutes=underway_min + lead_minutes)

    frames, meta = [], []
    for spec in specs:
        df = build_track(spec, start_utc, interval_s, rng)
        frames.append(df)
        meta.append({
            "mmsi": spec.mmsi, "vessel_name": spec.name, "flag": spec.flag,
            "vessel_type": spec.vessel_type, "length_m": spec.length_m,
            "role": spec.role, "note": spec.note, "tags": spec.tags,
            "pings": len(df),
        })

    ais = pd.concat(frames, ignore_index=True).sort_values(
        ["mmsi", "timestamp"]).reset_index(drop=True)

    return {
        "ais": ais,
        "vessels": pd.DataFrame(meta),
        "incident": INCIDENT,
        "interval_s": interval_s,
        "start_utc": start_utc,
        "grounding_utc": GROUNDING_UTC,
        "spill_position": (GROUNDING_LAT, GROUNDING_LON),
        "provenance": (
            "Reconstructed scenario. Vessel particulars, voyage, grounding position "
            "and timing follow the public casualty record; individual AIS pings are "
            "physically consistent synthesis, not recovered signal."
        ),
    }


def scenario_ais(interval_s: float = 60.0, seed: int = 20200725) -> pd.DataFrame:
    return build_scenario(interval_s=interval_s, seed=seed)["ais"]


AIS_EXPORT_COLUMNS = [
    "mmsi", "vessel_name", "flag", "vessel_type", "timestamp",
    "latitude", "longitude", "speed", "course", "heading", "rot",
    "msg_type", "status", "accuracy", "phase",
]


def to_csv_bytes(ais: pd.DataFrame) -> bytes:
    cols = [c for c in AIS_EXPORT_COLUMNS if c in ais.columns]
    return ais[cols].to_csv(index=False).encode("utf-8")
