"""
Dark Vessel & AIS Transponder Gap Detection Engine.

Cross-references Sentinel-1 SAR radar ship returns against active AIS broadcasts
to isolate non-cooperative targets ("dark vessels") operating without active
transponders, and detects suspicious AIS blackout gaps near sensitive maritime zones.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DEFAULT_PIXEL_RESOLUTION_M
from .inference.sar import SHIP_CLASS, SarResult, _components
from .inference.trajectory import haversine_km
from .scenario.sar_scenes import SarScene, by_key as get_sar_scene


# --------------------------------------------------------------------------
# Georeferencing / Coordinate Projection
# --------------------------------------------------------------------------
def sar_pixel_to_latlon(
    x: float,
    y: float,
    center_lat: float,
    center_lon: float,
    width_px: int,
    height_px: int,
    pixel_res_m: float = DEFAULT_PIXEL_RESOLUTION_M,
) -> Tuple[float, float]:
    """
    Project a 2D pixel coordinate (x, y) on a Sentinel-1 frame into (lat, lon).

    x: 0 (left) to width_px (right) -> East/West axis
    y: 0 (top) to height_px (bottom) -> North/South axis
    """
    # Offset in physical meters from image center
    dx_meters = (float(x) - (width_px / 2.0)) * pixel_res_m
    dy_meters = -((float(y) - (height_px / 2.0)) * pixel_res_m)  # top is north

    # Earth radius in meters
    r_earth = 6371000.0

    delta_lat = (dy_meters / r_earth) * (180.0 / math.pi)
    center_lat_rad = math.radians(center_lat)
    cos_lat = math.cos(center_lat_rad)
    cos_lat = max(abs(cos_lat), 1e-6)  # avoid division by zero near poles
    delta_lon = (dx_meters / (r_earth * cos_lat)) * (180.0 / math.pi)

    return float(center_lat + delta_lat), float(center_lon + delta_lon)


def latlon_to_sar_pixel(
    lat: float,
    lon: float,
    center_lat: float,
    center_lon: float,
    width_px: int,
    height_px: int,
    pixel_res_m: float = DEFAULT_PIXEL_RESOLUTION_M,
) -> Tuple[float, float]:
    """Inverse of sar_pixel_to_latlon."""
    r_earth = 6371000.0
    delta_lat = lat - center_lat
    delta_lon = lon - center_lon

    dy_meters = math.radians(delta_lat) * r_earth
    center_lat_rad = math.radians(center_lat)
    dx_meters = math.radians(delta_lon) * r_earth * math.cos(center_lat_rad)

    x = (dx_meters / pixel_res_m) + (width_px / 2.0)
    y = (width_px / 2.0)  # placeholder to initialize
    y = (-dy_meters / pixel_res_m) + (height_px / 2.0)

    return float(x), float(y)


# --------------------------------------------------------------------------
# Data Structures
# --------------------------------------------------------------------------
@dataclass
class RadarTarget:
    target_id: int
    centroid_xy: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    pixels: int
    latitude: float
    longitude: float
    estimated_length_m: float
    is_matched: bool = False
    matched_mmsi: Optional[int] = None
    matched_vessel_name: Optional[str] = None
    matched_distance_m: Optional[float] = None
    matched_speed_kn: Optional[float] = None
    matched_flag: Optional[str] = None
    matched_vessel_type: Optional[str] = None
    distance_to_spill_km: Optional[float] = None
    risk_level: str = "normal"  # normal | suspicious | dark_vessel_near_slick
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "centroid_xy": [round(c, 1) for c in self.centroid_xy],
            "bbox": self.bbox,
            "pixels": self.pixels,
            "latitude": round(self.latitude, 5),
            "longitude": round(self.longitude, 5),
            "estimated_length_m": round(self.estimated_length_m, 1),
            "is_matched": self.is_matched,
            "status": "COOPERATIVE (AIS ACTIVE)" if self.is_matched else "DARK VESSEL (NO AIS)",
            "matched_mmsi": self.matched_mmsi,
            "matched_vessel_name": self.matched_vessel_name,
            "matched_distance_m": round(self.matched_distance_m, 1) if self.matched_distance_m is not None else None,
            "matched_speed_kn": round(self.matched_speed_kn, 1) if self.matched_speed_kn is not None else None,
            "matched_flag": self.matched_flag,
            "matched_vessel_type": self.matched_vessel_type,
            "distance_to_spill_km": round(self.distance_to_spill_km, 2) if self.distance_to_spill_km is not None else None,
            "risk_level": self.risk_level,
            "notes": self.notes,
        }


@dataclass
class TransponderGap:
    mmsi: int
    vessel_name: str
    flag: str
    vessel_type: str
    gap_start: datetime
    gap_end: datetime
    gap_minutes: float
    last_latitude: float
    last_longitude: float
    last_speed_kn: float
    last_course: float
    resume_latitude: float
    resume_longitude: float
    resume_speed_kn: float
    resume_course: float
    distance_km: float
    implied_speed_kn: float
    min_dist_to_spill_km: float
    crossed_hazard_zone: bool
    risk_score: float
    risk_label: str  # nominal | elevated | high_risk_gap

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mmsi": self.mmsi,
            "vessel_name": self.vessel_name,
            "flag": self.flag,
            "vessel_type": self.vessel_type,
            "gap_start": self.gap_start.isoformat(),
            "gap_end": self.gap_end.isoformat(),
            "gap_minutes": round(self.gap_minutes, 1),
            "last_position": [round(self.last_latitude, 5), round(self.last_longitude, 5)],
            "last_speed_kn": round(self.last_speed_kn, 1),
            "last_course": round(self.last_course, 1),
            "resume_position": [round(self.resume_latitude, 5), round(self.resume_longitude, 5)],
            "resume_speed_kn": round(self.resume_speed_kn, 1),
            "resume_course": round(self.resume_course, 1),
            "distance_km": round(self.distance_km, 2),
            "implied_speed_kn": round(self.implied_speed_kn, 1),
            "min_dist_to_spill_km": round(self.min_dist_to_spill_km, 2),
            "crossed_hazard_zone": self.crossed_hazard_zone,
            "risk_score": round(self.risk_score, 3),
            "risk_label": self.risk_label,
        }


# --------------------------------------------------------------------------
# Dark Vessel Cross-Referencing Logic
# --------------------------------------------------------------------------
def extract_radar_targets_from_mask(
    mask: np.ndarray,
    center_lat: float,
    center_lon: float,
    pixel_res_m: float = DEFAULT_PIXEL_RESOLUTION_M,
    top_k: int = 30,
) -> List[RadarTarget]:
    """Extract and georeference class 3 (Ship) connected components from a SAR mask."""
    h, w = mask.shape[:2]
    comps = _components(mask, SHIP_CLASS, top_k=top_k)
    targets: List[RadarTarget] = []

    for i, c in enumerate(comps, start=1):
        cx, cy = c["centroid_xy"]
        lat, lon = sar_pixel_to_latlon(cx, cy, center_lat, center_lon, w, h, pixel_res_m)

        # Approximate length from bounding box extent in physical meters
        _bx, _by, bw, bh = c["bbox"]
        length_m = max(bw, bh) * pixel_res_m

        targets.append(RadarTarget(
            target_id=i,
            centroid_xy=(cx, cy),
            bbox=c["bbox"],
            pixels=c["pixels"],
            latitude=lat,
            longitude=lon,
            estimated_length_m=length_m,
        ))

    return targets


def match_sar_ships_to_ais(
    targets: List[RadarTarget],
    ais_df: pd.DataFrame,
    acquisition_time: Optional[datetime] = None,
    spill_lat: Optional[float] = None,
    spill_lon: Optional[float] = None,
    gating_radius_m: float = 2000.0,
    time_window_minutes: float = 45.0,
) -> List[RadarTarget]:
    """
    Cross-reference detected SAR radar targets against active AIS broadcasts.

    Targets matched within gating_radius_m of an AIS vessel are tagged cooperative.
    Unmatched radar reflections are tagged DARK VESSELS.
    """
    if not targets:
        return []

    # Ensure timestamp parsing
    df = ais_df.copy()
    if "timestamp" in df.columns:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], utc=True)
    else:
        df["timestamp_dt"] = datetime.now(timezone.utc)

    # Filter AIS to temporal window around satellite overpass
    if acquisition_time is not None:
        anchor = pd.Timestamp(acquisition_time)
        if anchor.tzinfo is None:
            anchor = anchor.tz_localize("UTC")
        window = pd.Timedelta(minutes=time_window_minutes)
        candidates_df = df[(df["timestamp_dt"] >= anchor - window) &
                           (df["timestamp_dt"] <= anchor + window)].copy()
        if candidates_df.empty:
            candidates_df = df.copy()  # fallback to active frame if window too tight
    else:
        candidates_df = df.copy()

    # Get one representative fix per MMSI closest to the anchor time
    vessel_fixes = []
    for mmsi, group in candidates_df.groupby("mmsi"):
        if acquisition_time is not None:
            anchor = pd.Timestamp(acquisition_time)
            if anchor.tzinfo is None:
                anchor = anchor.tz_localize("UTC")
            time_diffs = (group["timestamp_dt"] - anchor).abs()
            best_idx = time_diffs.idxmin()
            best_row = group.loc[best_idx]
        else:
            best_row = group.iloc[len(group) // 2]

        vessel_fixes.append({
            "mmsi": int(mmsi),
            "vessel_name": str(best_row.get("vessel_name", f"MMSI {mmsi}")),
            "flag": str(best_row.get("flag", "-")),
            "vessel_type": str(best_row.get("vessel_type", "-")),
            "latitude": float(best_row["latitude"]),
            "longitude": float(best_row["longitude"]),
            "speed": float(best_row.get("speed", 0.0)),
        })

    # Greedily match targets to closest available AIS fix within gating radius
    matched_mmsis = set()

    for target in targets:
        # Distance to slick if spill coordinates provided
        if spill_lat is not None and spill_lon is not None:
            target.distance_to_spill_km = haversine_km(
                target.latitude, target.longitude, spill_lat, spill_lon
            )

        best_dist_m = float("inf")
        best_fix = None

        for fix in vessel_fixes:
            if fix["mmsi"] in matched_mmsis:
                continue
            dist_km = haversine_km(target.latitude, target.longitude,
                                   fix["latitude"], fix["longitude"])
            dist_m = dist_km * 1000.0
            if dist_m < best_dist_m:
                best_dist_m = dist_m
                best_fix = fix

        if best_fix is not None and best_dist_m <= gating_radius_m:
            target.is_matched = True
            target.matched_mmsi = best_fix["mmsi"]
            target.matched_vessel_name = best_fix["vessel_name"]
            target.matched_distance_m = best_dist_m
            target.matched_speed_kn = best_fix["speed"]
            target.matched_flag = best_fix["flag"]
            target.matched_vessel_type = best_fix["vessel_type"]
            target.risk_level = "normal"
            target.notes = f"Correlated with AIS fix {best_dist_m:.0f} m away."
            matched_mmsis.add(best_fix["mmsi"])
        else:
            target.is_matched = False
            if target.distance_to_spill_km is not None and target.distance_to_spill_km <= 15.0:
                target.risk_level = "dark_vessel_near_slick"
                target.notes = (f"Unmatched radar return {target.estimated_length_m:.0f} m LOA, "
                                f"located {target.distance_to_spill_km:.1f} km from observed slick center.")
            else:
                target.risk_level = "suspicious"
                target.notes = (f"Unmatched radar reflection (~{target.estimated_length_m:.0f} m LOA). "
                                f"No AIS signal within {gating_radius_m:.0f} m.")

    return targets


# --------------------------------------------------------------------------
# AIS Transponder Gap / Blackout Detection
# --------------------------------------------------------------------------
def _point_to_segment_distance_km(
    plat: float, plon: float,
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """Distance from a point (spill) to a line segment (vessel transit vector)."""
    # Sample 10 points along great circle for robust distance computation
    num_samples = 10
    min_d = float("inf")
    for t in np.linspace(0.0, 1.0, num_samples):
        cur_lat = lat1 + t * (lat2 - lat1)
        cur_lon = lon1 + t * (lon2 - lon1)
        d = haversine_km(plat, plon, cur_lat, cur_lon)
        if d < min_d:
            min_d = d
    return min_d


def detect_transponder_gaps(
    ais_df: pd.DataFrame,
    min_gap_minutes: float = 30.0,
    min_speed_knots: float = 2.5,
    spill_lat: Optional[float] = None,
    spill_lon: Optional[float] = None,
    hazard_radius_km: float = 15.0,
) -> List[TransponderGap]:
    """
    Detect transponder silence / blackout gaps for moving vessels.

    Flags vessels that stopped broadcasting while underway, assessing whether
    their silent transit traversed near the spill site.
    """
    if ais_df.empty or "timestamp" not in ais_df.columns:
        return []

    df = ais_df.copy()
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    gaps: List[TransponderGap] = []

    for mmsi, group in df.groupby("mmsi"):
        track = group.sort_values("ts").reset_index(drop=True)
        if len(track) < 2:
            continue

        vname = str(track["vessel_name"].iloc[0]) if "vessel_name" in track else f"MMSI {mmsi}"
        flag = str(track["flag"].iloc[0]) if "flag" in track else "-"
        vtype = str(track["vessel_type"].iloc[0]) if "vessel_type" in track else "-"

        for i in range(1, len(track)):
            r_prev = track.iloc[i - 1]
            r_curr = track.iloc[i]

            gap_sec = (r_curr["ts"] - r_prev["ts"]).total_seconds()
            gap_min = gap_sec / 60.0

            if gap_min >= min_gap_minutes:
                # Was vessel underway prior or upon resumption?
                v_prev = float(r_prev.get("speed", 0.0))
                v_curr = float(r_curr.get("speed", 0.0))
                max_v = max(v_prev, v_curr)

                if max_v >= min_speed_knots:
                    dist_km = haversine_km(
                        float(r_prev["latitude"]), float(r_prev["longitude"]),
                        float(r_curr["latitude"]), float(r_curr["longitude"]),
                    )
                    implied_speed = (dist_km / 1.852) / (gap_min / 60.0) if gap_min > 0 else 0.0

                    # Check proximity to spill site
                    if spill_lat is not None and spill_lon is not None:
                        min_dist_spill = _point_to_segment_distance_km(
                            spill_lat, spill_lon,
                            float(r_prev["latitude"]), float(r_prev["longitude"]),
                            float(r_curr["latitude"]), float(r_curr["longitude"]),
                        )
                        crossed = min_dist_spill <= hazard_radius_km
                    else:
                        min_dist_spill = 999.0
                        crossed = False

                    # Risk scoring: longer gap + high speed + near spill = critical
                    risk = (min(gap_min / 180.0, 1.0) * 0.4 +
                            min(implied_speed / 15.0, 1.0) * 0.2 +
                            (0.4 if crossed else 0.0))

                    if crossed and risk >= 0.6:
                        label = "high_risk_gap"
                    elif risk >= 0.4:
                        label = "elevated"
                    else:
                        label = "nominal"

                    gaps.append(TransponderGap(
                        mmsi=int(mmsi),
                        vessel_name=vname,
                        flag=flag,
                        vessel_type=vtype,
                        gap_start=r_prev["ts"].to_pydatetime(),
                        gap_end=r_curr["ts"].to_pydatetime(),
                        gap_minutes=gap_min,
                        last_latitude=float(r_prev["latitude"]),
                        last_longitude=float(r_prev["longitude"]),
                        last_speed_kn=v_prev,
                        last_course=float(r_prev.get("course", 0.0)),
                        resume_latitude=float(r_curr["latitude"]),
                        resume_longitude=float(r_curr["longitude"]),
                        resume_speed_kn=v_curr,
                        resume_course=float(r_curr.get("course", 0.0)),
                        distance_km=dist_km,
                        implied_speed_kn=implied_speed,
                        min_dist_to_spill_km=min_dist_spill,
                        crossed_hazard_zone=crossed,
                        risk_score=risk,
                        risk_label=label,
                    ))

    gaps.sort(key=lambda g: g.risk_score, reverse=True)
    return gaps


# --------------------------------------------------------------------------
# High-Level Audit Coordinator
# --------------------------------------------------------------------------
def analyze_scene_dark_vessels(
    scene_key: str,
    ais_df: pd.DataFrame,
    spill_lat: Optional[float] = None,
    spill_lon: Optional[float] = None,
) -> Dict[str, Any]:
    """Scan a curated SAR scene for dark vessels by matching its mask against AIS."""
    scene = get_sar_scene(scene_key)
    if scene is None:
        raise ValueError(f"Unknown SAR scene key: {scene_key}")

    mask = scene.load_mask()
    targets = extract_radar_targets_from_mask(
        mask,
        center_lat=scene.position[0],
        center_lon=scene.position[1],
        pixel_res_m=scene.pixel_resolution_m,
    )

    try:
        acq_time = datetime.strptime(scene.captured.replace(" UTC", ""), "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    except Exception:
        acq_time = None

    s_lat = spill_lat if spill_lat is not None else scene.position[0]
    s_lon = spill_lon if spill_lon is not None else scene.position[1]

    matched_targets = match_sar_ships_to_ais(
        targets=targets,
        ais_df=ais_df,
        acquisition_time=acq_time,
        spill_lat=s_lat,
        spill_lon=s_lon,
        gating_radius_m=2000.0,
        time_window_minutes=60.0,
    )

    dark_targets = [t for t in matched_targets if not t.is_matched]
    coop_targets = [t for t in matched_targets if t.is_matched]

    return {
        "scene_key": scene_key,
        "scene_title": scene.title,
        "scene_position": scene.position,
        "captured": scene.captured,
        "total_radar_ships": len(matched_targets),
        "cooperative_ships": len(coop_targets),
        "dark_vessels": len(dark_targets),
        "high_risk_dark_vessels": sum(1 for t in dark_targets if t.risk_level == "dark_vessel_near_slick"),
        "targets": [t.as_dict() for t in matched_targets],
    }

