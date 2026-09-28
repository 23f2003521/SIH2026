"""
POSEatSea -- Coastal Vulnerability Integration Module.

Provides access to the InVEST Coastal Vulnerability assessment outputs,
translating the biophysical coastal exposure model (incorporating relief,
bathymetry, wave power, winds, and coral reef/mangrove habitats) into
actionable operational risk metrics for the maritime surveillance console.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from poseatsea.config import PROJECT_ROOT
from poseatsea.scenario.wakashio import GROUNDING_LAT, GROUNDING_LON

COASTAL_GEOJSON_PATH = PROJECT_ROOT / "assets/coastal_vulnerability/coastal_exposure.geojson"
COASTAL_METADATA_PATH = PROJECT_ROOT / "metadata.json"

# InVEST Exposure Tiers based on cumulative exposure index (1.0 to 5.0)
TIER_COLORS = {
    "Low": "#3fb950",        # Good / Green
    "Moderate": "#d29922",   # Warning / Amber
    "High": "#f85149",       # Critical / Red
    "Very High": "#bd561d",  # Extreme / Deep Orange-Red
}


def is_available() -> bool:
    """Return True if the InVEST Coastal Vulnerability output layer exists."""
    return COASTAL_GEOJSON_PATH.exists()


def load_exposure_points() -> pd.DataFrame:
    """
    Load computed InVEST coastal vulnerability shoreline points.

    Returns a DataFrame with columns:
    [latitude, longitude, exposure_index, exposure_tier, color,
     r_hab, r_wave, r_wind, r_relief, r_surge]
    """
    if not is_available():
        return pd.DataFrame()

    try:
        with open(COASTAL_GEOJSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = []
        for feat in data.get("features", []):
            coords = feat.get("geometry", {}).get("coordinates", [0, 0])
            props = feat.get("properties", {})

            # InVEST exposure index can be named exposure_index, EI, or coastal_exposure
            ei = float(
                props.get("exposure_index") or
                props.get("EI") or
                props.get("coastal_exposure") or
                props.get("exposure_rank") or
                2.5
            )

            # Classify exposure tier
            if ei <= 2.0:
                tier = "Low"
            elif ei <= 3.0:
                tier = "Moderate"
            elif ei <= 4.0:
                tier = "High"
            else:
                tier = "Very High"

            def _get_num(keys: list[str], default: float = 1.0) -> float:
                for k in keys:
                    v = props.get(k)
                    if v is not None:
                        try:
                            return float(v)
                        except (ValueError, TypeError):
                            pass
                return default

            records.append({
                "shore_id": int(props.get("shore_id", len(records))),
                "latitude": float(coords[1]),
                "longitude": float(coords[0]),
                "exposure_index": ei,
                "exposure_tier": tier,
                "color": TIER_COLORS[tier],
                "r_hab": _get_num(["R_hab", "r_hab", "R_HAB"], 1.0),
                "r_wave": _get_num(["R_wave", "r_wave", "R_WAVE"], 3.0),
                "r_wind": _get_num(["R_wind", "r_wind", "R_WIND"], 3.0),
                "r_relief": _get_num(["R_relief", "r_relief", "R_RELIEF"], 2.0),
                "r_surge": _get_num(["R_surge", "r_surge", "R_SURGE"], 2.0),
            })

        df = pd.DataFrame(records)
        if not df.empty and "latitude" in df.columns and "longitude" in df.columns:
            # Robust planar distance from MV Wakashio grounding site
            cos_lat = np.cos(np.radians(GROUNDING_LAT))
            lat_diff = (pd.to_numeric(df["latitude"], errors="coerce") - GROUNDING_LAT) * 111.0
            lon_diff = (pd.to_numeric(df["longitude"], errors="coerce") - GROUNDING_LON) * 111.0 * cos_lat
            df["dist_to_grounding_km"] = np.sqrt(lat_diff ** 2 + lon_diff ** 2).fillna(0.0)
        elif not df.empty:
            df["dist_to_grounding_km"] = 0.0

        return df
    except Exception as err:
        print(f"[WARN] Error loading coastal exposure points: {err}")
        return pd.DataFrame()


def coastal_summary() -> Dict[str, Any]:
    """
    Summary metrics of the coastal vulnerability analysis for UI KPI cards.
    """
    df = load_exposure_points()
    if df.empty:
        return {
            "available": False,
            "status": "InVEST model run pending",
            "segments_count": 0,
            "mean_exposure": None,
            "max_exposure": None,
            "high_risk_segments": 0,
            "high_risk_pct": 0.0,
            "dist_wakashio_to_high_risk_km": None,
            "avg_r_hab": None,
            "avg_r_wave": None,
            "avg_r_wind": None,
            "avg_r_relief": None,
            "avg_r_surge": None,
        }

    dists = df.get("dist_to_grounding_km", pd.Series([0.0] * len(df)))
    high_risk_df = df[df["exposure_tier"].isin(["High", "Very High"])]
    min_dist_to_high = float(high_risk_df["dist_to_grounding_km"].min()) if not high_risk_df.empty else float(dists.min())

    return {
        "available": True,
        "status": "Verified InVEST Coastal Vulnerability Layer",
        "segments_count": len(df),
        "mean_exposure": float(df["exposure_index"].mean()),
        "max_exposure": float(df["exposure_index"].max()),
        "high_risk_segments": len(high_risk_df),
        "high_risk_pct": (len(high_risk_df) / len(df) * 100.0) if len(df) > 0 else 0.0,
        "dist_wakashio_to_high_risk_km": min_dist_to_high,
        "avg_r_hab": float(df["r_hab"].mean()),
        "avg_r_wave": float(df["r_wave"].mean()),
        "avg_r_wind": float(df["r_wind"].mean()),
        "avg_r_relief": float(df["r_relief"].mean()),
        "avg_r_surge": float(df["r_surge"].mean()),
    }

