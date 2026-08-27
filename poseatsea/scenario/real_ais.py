"""
Real AIS — Mauritius AOI, July 2020.

This is the genuine article: 27,979 satellite and terrestrial AIS records
covering the Mauritius area of interest for 1–31 July 2020, the same water and
month the trajectory model was normalised for. **The MV Wakashio's own
grounding is in this feed**, broadcast by the ship itself.

What the record shows, straight from the data:

  * The vessel transits south-west at 11.6–12.0 kn on courses 245–247°.
  * Its last ping with way on is 2020-07-25 15:27:22 UTC at −20.44421,
    57.74328, already down to 1.7 kn on course 230.
  * Eighty-four seconds later it reports 0.2 kn, and it never moves again.
  * From there it broadcasts navigational status **6 — Aground** for six days.

15:27 UTC is 19:27 local, which matches the casualty record's 19:25 to within
two minutes. Nothing here is reconstructed.

Everything the console shows now derives from this file. The synthetic
scenario this replaced is gone.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .wakashio import INCIDENT as _DOCUMENTED_INCIDENT

CSV_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "ais" / "mauritius_aoi_202007.csv"

# The casualty, as it identifies itself on air.
WAKASHIO_MMSI = 372711000

# AIS sentinel: rate of turn "not available". Left raw it reads as a hard
# port swing to every downstream consumer, so it is mapped to zero.
ROT_NOT_AVAILABLE = -128

# Position reports. Type 5 is static/voyage data and carries no position.
POSITION_MSG_TYPES = (1, 2, 3, 18, 19, 27)

# ITU-R M.1371 ship-and-cargo-type codes, collapsed to something readable.
SHIP_TYPE_NAMES = {
    30: "Fishing vessel", 31: "Towing", 32: "Towing (large)",
    35: "Military", 36: "Sailing", 37: "Pleasure craft",
    52: "Tug", 60: "Passenger ship", 70: "Cargo ship", 71: "Cargo (hazardous A)",
    72: "Cargo (hazardous B)", 73: "Cargo (hazardous C)", 74: "Cargo (hazardous D)",
    79: "Bulk carrier", 80: "Tanker", 81: "Tanker (hazardous A)",
    82: "Tanker (hazardous B)", 83: "Tanker (hazardous C)", 84: "Tanker (hazardous D)",
    89: "Tanker (other)",
}

FLAG_NAMES = {
    "PA": "Panama", "SG": "Singapore", "MH": "Marshall Islands", "HK": "Hong Kong",
    "LR": "Liberia", "JP": "Japan", "IN": "India", "MT": "Malta", "MU": "Mauritius",
    "CY": "Cyprus", "GR": "Greece", "CN": "China", "KR": "South Korea",
    "BS": "Bahamas", "MY": "Malaysia", "GB": "United Kingdom", "NO": "Norway",
}

# The vessels the console presents. Chosen for track density on the day of the
# casualty: the Wakashio plus the five best-covered contemporaneous transits.
# Every one is a real ship that was really there.
DEMO_MMSIS = [
    WAKASHIO_MMSI,   # WAKASHIO      — the casualty
    564796000,       # KOTA SURIA    — Singapore, 312 fixes
    538006057,       # VERY MARIA    — Marshall Islands, 158 fixes
    477007600,       # DHT EDELWEISS — Hong Kong tanker, 101 fixes
    477848500,       # PALONA        — Hong Kong, 55 fixes
    371282000,       # AQUAVITA SOL  — Panama, 43 fixes
]

INCIDENT_DAY = "2020-07-25"


def ship_type_name(code) -> str:
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "Unknown"
    if code in SHIP_TYPE_NAMES:
        return SHIP_TYPE_NAMES[code]
    if 70 <= code <= 79:
        return "Cargo ship"
    if 80 <= code <= 89:
        return "Tanker"
    return f"Type {code}"


def flag_name(code) -> str:
    if not isinstance(code, str):
        return "Unknown"
    return FLAG_NAMES.get(code.strip().upper(), code.strip().upper())


# --------------------------------------------------------------------------
# Loading and cleaning
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_raw() -> pd.DataFrame:
    """The full extract, with timestamps parsed and identities resolved."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"AIS extract not found at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # About 4% of rows carry fractional seconds; a single inferred format drops
    # them silently, which is exactly the kind of quiet data loss that ruins a
    # track. format="mixed" parses all of them.
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce",
                                     format="mixed", utc=True)
    df = df.dropna(subset=["timestamp"])

    # Static (type 5) rows carry the name/flag/type for an MMSI but no position.
    # Resolve identity across the whole file, then attach it to position rows.
    for col, target in [("name", "vessel_name"), ("flag", "flag_code"),
                        ("imo", "imo"), ("call_sign", "call_sign"),
                        ("ship_and_cargo_type", "type_code"),
                        ("length", "length_m"), ("destination", "destination")]:
        if col in df.columns:
            resolved = df.dropna(subset=[col]).groupby("mmsi")[col].first()
            df[target] = df["mmsi"].map(resolved)

    return df


def clean_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Position reports only, with the fields the models consume made usable."""
    pos = df[df["msg_type"].isin(POSITION_MSG_TYPES)].copy()
    pos = pos.dropna(subset=["latitude", "longitude", "speed", "course"])

    # Rate of turn: -128 means "not available", not a hard turn.
    pos["rot"] = pos["rot"].replace(ROT_NOT_AVAILABLE, 0.0).fillna(0.0)
    pos["status"] = pos["status"].fillna(0).astype(int)
    pos["accuracy"] = pos["accuracy"].fillna(0).astype(int)
    pos["msg_type"] = pos["msg_type"].astype(int)
    pos["heading"] = pos["heading"].fillna(pos["course"])

    pos["vessel_name"] = (pos["vessel_name"].fillna("MMSI " + pos["mmsi"].astype(str))
                          .astype(str).str.strip())
    pos["flag"] = pos["flag_code"].map(flag_name)
    pos["vessel_type"] = pos["type_code"].map(ship_type_name)

    # One fix per vessel per instant; satellite and terrestrial receivers
    # frequently report the same ping twice.
    pos = (pos.sort_values(["mmsi", "timestamp"])
              .drop_duplicates(subset=["mmsi", "timestamp"], keep="first")
              .reset_index(drop=True))
    return pos


# --------------------------------------------------------------------------
# The incident
# --------------------------------------------------------------------------
@dataclass
class GroundingEvent:
    last_moving: pd.Series
    first_stopped: pd.Series

    @property
    def utc(self) -> datetime:
        return self.first_stopped["timestamp"].to_pydatetime()

    @property
    def position(self):
        return float(self.first_stopped["latitude"]), float(self.first_stopped["longitude"])

    @property
    def gap_seconds(self) -> float:
        return (self.first_stopped["timestamp"] - self.last_moving["timestamp"]).total_seconds()

    def as_dict(self) -> Dict[str, Any]:
        lat, lon = self.position
        return {
            "utc": self.utc,
            "local": self.utc.astimezone(timezone.utc).strftime("%d %B %Y, %H:%M UTC"),
            "position": [lat, lon],
            "last_speed_kn": float(self.last_moving["speed"]),
            "last_course": float(self.last_moving["course"]),
            "stopped_within_s": self.gap_seconds,
        }


def detect_grounding(track: pd.DataFrame, moving_kn: float = 1.0) -> Optional[GroundingEvent]:
    """
    Find where a vessel lost way and never regained it.

    Defined from the data rather than asserted: the last fix above `moving_kn`
    that is not followed by any later movement, and the fix immediately after it.
    """
    track = track.sort_values("timestamp").reset_index(drop=True)
    moving = track.index[track["speed"] > moving_kn]
    if not len(moving):
        return None
    last = int(moving[-1])
    if last + 1 >= len(track):
        return None
    return GroundingEvent(track.loc[last], track.loc[last + 1])


# --------------------------------------------------------------------------
# Scenario assembly
# --------------------------------------------------------------------------
@lru_cache(maxsize=4)
def build_scenario(day: str = INCIDENT_DAY, mmsis: tuple = tuple(DEMO_MMSIS)) -> Dict[str, Any]:
    """
    The console's dataset: real tracks for the casualty and its contemporaries.

    Returns the same shape the rest of the application already consumes, so
    nothing downstream needed to change when this replaced the synthetic feed.
    """
    from ..inference.ais import add_derived_features

    pos = clean_positions(load_raw())

    start = pd.Timestamp(day, tz="UTC")
    end = start + pd.Timedelta(days=1)
    window = pos[(pos["timestamp"] >= start) & (pos["timestamp"] < end)]
    ais = window[window["mmsi"].isin(list(mmsis))].copy()

    # Label the casualty's own phases from its speed profile, so the UI can
    # narrate the track without any of it being asserted by hand.
    ais["phase"] = "transit"
    wak = ais[ais["mmsi"] == WAKASHIO_MMSI]
    event = detect_grounding(wak) if len(wak) else None
    if event is not None:
        aground = (ais["mmsi"] == WAKASHIO_MMSI) & (ais["timestamp"] >= event.first_stopped["timestamp"])
        ais.loc[aground, "phase"] = "aground"
        approach = ((ais["mmsi"] == WAKASHIO_MMSI)
                    & (ais["timestamp"] >= event.last_moving["timestamp"] - pd.Timedelta(minutes=40))
                    & (~aground))
        ais.loc[approach, "phase"] = "final approach"

    ais = add_derived_features(ais).reset_index(drop=True)

    meta = (ais.groupby("mmsi")
              .agg(vessel_name=("vessel_name", "first"), flag=("flag", "first"),
                   vessel_type=("vessel_type", "first"), pings=("speed", "size"),
                   mean_speed=("speed", "mean"), first_seen=("timestamp", "min"),
                   last_seen=("timestamp", "max"))
              .reset_index())
    meta["role"] = np.where(meta["mmsi"] == WAKASHIO_MMSI, "casualty", "background")

    grounding_utc = event.utc if event else None
    spill = event.position if event else (
        _DOCUMENTED_INCIDENT["position"][0], _DOCUMENTED_INCIDENT["position"][1])

    incident = dict(_DOCUMENTED_INCIDENT)
    incident["vessel"] = dict(incident["vessel"])
    incident["vessel"]["mmsi"] = WAKASHIO_MMSI          # as actually broadcast
    incident["position"] = spill
    if grounding_utc:
        incident["grounding_utc"] = grounding_utc
        incident["grounding_local"] = grounding_utc.strftime("%d %B %Y, %H:%M UTC")
        incident["observed"] = event.as_dict()

    return {
        "ais": ais,
        "vessels": meta,
        "incident": incident,
        "interval_s": float(ais.groupby("mmsi")["timestamp"].diff()
                            .dt.total_seconds().median() or 60.0),
        "start_utc": ais["timestamp"].min().to_pydatetime(),
        "grounding_utc": grounding_utc,
        "spill_position": spill,
        "event": event,
        "provenance": (
            "Real AIS. Mauritius AOI extract, 1-31 July 2020 — the same region and "
            "month the trajectory model was normalised for. The MV Wakashio's "
            "grounding is recorded in this feed by the vessel itself, including six "
            "days of navigational status 6 (aground). No track on this page is "
            "synthetic."
        ),
    }


def scenario_ais(day: str = INCIDENT_DAY) -> pd.DataFrame:
    return build_scenario(day)["ais"]


def fleet_overview(day: str = INCIDENT_DAY) -> pd.DataFrame:
    """Every vessel seen in the AOI that day, not just the demo set."""
    pos = clean_positions(load_raw())
    start = pd.Timestamp(day, tz="UTC")
    window = pos[(pos["timestamp"] >= start) & (pos["timestamp"] < start + pd.Timedelta(days=1))]
    return (window.groupby("mmsi")
            .agg(vessel_name=("vessel_name", "first"), flag=("flag", "first"),
                 vessel_type=("vessel_type", "first"), pings=("speed", "size"),
                 mean_speed=("speed", "mean"))
            .reset_index().sort_values("pings", ascending=False))


AIS_EXPORT_COLUMNS = [
    "mmsi", "vessel_name", "flag", "vessel_type", "timestamp",
    "latitude", "longitude", "speed", "course", "heading", "rot",
    "msg_type", "status", "accuracy", "phase",
]


def to_csv_bytes(ais: pd.DataFrame) -> bytes:
    cols = [c for c in AIS_EXPORT_COLUMNS if c in ais.columns]
    return ais[cols].to_csv(index=False).encode("utf-8")
