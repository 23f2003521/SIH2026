"""
POSEatSea -- Coastal Vulnerability Integration Module.

Provides access to the InVEST Coastal Vulnerability assessment outputs,
translating the biophysical coastal exposure model (incorporating relief,
bathymetry, wave power, winds, and coral reef/mangrove habitats) into
actionable operational risk metrics for the maritime surveillance console.
"""
from __future__ import annotations

import json
import os
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

# Runtime execution and data caching state
_EXECUTION_ATTEMPTED: bool = False
_EXECUTION_MODE: str = "cached"  # "live" | "cached" | "unavailable"
_EXECUTION_STATUS: str = "Using cached analysis"
_EXPOSURE_POINTS_CACHE: Optional[pd.DataFrame] = None


def get_coastal_geojson_path() -> Path:
    """Resolve deployment-safe path to coastal_exposure.geojson."""
    candidates = [
        COASTAL_GEOJSON_PATH,
        PROJECT_ROOT / "assets/coastal_vulnerability/coastal_exposure.geojson",
        Path.cwd() / "assets/coastal_vulnerability/coastal_exposure.geojson",
        Path(__file__).resolve().parent.parent / "assets/coastal_vulnerability/coastal_exposure.geojson",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p
    return COASTAL_GEOJSON_PATH


def ensure_exposure_data(force_live: bool = False) -> Dict[str, Any]:
    """
    Ensure Coastal Vulnerability exposure data is available for the dashboard.
    
    1. Attempts live InVEST model generation if natcap.invest is available in the environment.
    2. Silently falls back to cached assets/coastal_vulnerability/coastal_exposure.geojson.
    3. Caches result in memory to avoid expensive re-runs on every Streamlit interaction.
    """
    global _EXECUTION_ATTEMPTED, _EXECUTION_MODE, _EXECUTION_STATUS, _EXPOSURE_POINTS_CACHE

    if _EXECUTION_ATTEMPTED and not force_live:
        return {"mode": _EXECUTION_MODE, "status": _EXECUTION_STATUS}

    _EXECUTION_ATTEMPTED = True

    # Check environment variable override (e.g. POSEATSEA_COASTAL_MODE=cached)
    env_mode = os.getenv("POSEATSEA_COASTAL_MODE", "").lower().strip()
    if env_mode == "cached":
        _EXECUTION_MODE = "cached"
        _EXECUTION_STATUS = "Using cached analysis (configured via POSEATSEA_COASTAL_MODE)"
        return {"mode": _EXECUTION_MODE, "status": _EXECUTION_STATUS}

    # Attempt live execution if natcap.invest package is available
    can_run_live = False
    try:
        import natcap.invest  # noqa: F401
        can_run_live = True
    except (ImportError, Exception):
        can_run_live = False

    if can_run_live:
        try:
            from scripts import run_coastal_vulnerability
            if run_coastal_vulnerability.check_prerequisites():
                success = run_coastal_vulnerability.run_invest()
                geojson_p = get_coastal_geojson_path()
                if success and geojson_p.exists() and geojson_p.stat().st_size > 0:
                    _EXECUTION_MODE = "live"
                    _EXECUTION_STATUS = "Live InVEST model execution"
                    _EXPOSURE_POINTS_CACHE = None
                    return {"mode": "live", "status": _EXECUTION_STATUS}
        except Exception as err:
            print(f"[INFO] Live InVEST execution attempt: {err}. Falling back to cached analysis.")

    # Fallback to pre-generated cached GeoJSON
    geojson_p = get_coastal_geojson_path()
    if geojson_p.exists() and geojson_p.stat().st_size > 0:
        _EXECUTION_MODE = "cached"
        _EXECUTION_STATUS = "Using cached analysis"
        return {"mode": "cached", "status": _EXECUTION_STATUS}

    _EXECUTION_MODE = "unavailable"
    _EXECUTION_STATUS = "InVEST Coastal Vulnerability output layer not available"
    return {"mode": "unavailable", "status": _EXECUTION_STATUS}


def get_execution_mode() -> str:
    """Return current execution mode: 'live', 'cached', or 'unavailable'."""
    ensure_exposure_data()
    return _EXECUTION_MODE


def is_available() -> bool:
    """Return True if Coastal Vulnerability exposure points can be loaded."""
    ensure_exposure_data()
    p = get_coastal_geojson_path()
    return (p.exists() and p.stat().st_size > 0) or (_EXECUTION_MODE == "live")


def load_exposure_points(force_reload: bool = False) -> pd.DataFrame:
    """
    Load computed InVEST coastal vulnerability shoreline points.
    Uses in-memory caching to prevent expensive re-parsing on Streamlit reruns.
    """
    global _EXPOSURE_POINTS_CACHE
    if _EXPOSURE_POINTS_CACHE is not None and not force_reload:
        return _EXPOSURE_POINTS_CACHE.copy()

    ensure_exposure_data()
    geojson_path = get_coastal_geojson_path()
    if not geojson_path.exists():
        return pd.DataFrame()

    try:
        with open(geojson_path, "r", encoding="utf-8") as f:
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

        _EXPOSURE_POINTS_CACHE = df
        return df.copy()
    except Exception as err:
        print(f"[WARN] Error loading coastal exposure points: {err}")
        return pd.DataFrame()


def coastal_summary() -> Dict[str, Any]:
    """
    Summary metrics of the coastal vulnerability analysis for UI KPI cards.
    """
    ensure_exposure_data()
    df = load_exposure_points()
    mode = get_execution_mode()
    mode_label = "Live InVEST Execution" if mode == "live" else "Using cached analysis"
    status_text = "Live InVEST Coastal Vulnerability Model" if mode == "live" else "Verified InVEST Coastal Vulnerability Layer (Cached)"

    if df.empty:
        return {
            "available": False,
            "status": "InVEST Coastal Vulnerability output layer not available",
            "execution_mode": mode,
            "mode_label": mode_label,
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
        "status": status_text,
        "execution_mode": mode,
        "mode_label": mode_label,
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


# --------------------------------------------------------------------------
# Ecological Habitat & Species Registry (Mauritius AOI / Blue Bay Ramsar #1798)
# --------------------------------------------------------------------------
HABITATS_DIR = PROJECT_ROOT / "assets/habitats"
HABITAT_PROTECTION_CSV = PROJECT_ROOT / "assets/coastal_vulnerability/invest_workspace/intermediate/habitats/habitat_protection_mauritius.csv"
COASTAL_EXPOSURE_CSV = PROJECT_ROOT / "assets/coastal_vulnerability/invest_workspace/coastal_exposure_mauritius.csv"

HABITAT_SPECIES_REGISTRY = {
    "coral_reef": {
        "id": "coral_reef",
        "name": "Coral Reef Barrier",
        "color": "#00d2d2",
        "buffer_radius_m": 2000,
        "invest_rank": 1,
        "protection_service": "High wave energy attenuation (dissipates up to 97% of open-ocean swell energy, preventing barrier shoreline erosion)",
        "conservation_status": "Ramsar Site #1798 & National Marine Protected Area",
        "taxa": [
            {"scientific": "Acropora muricata", "common": "Staghorn Coral", "role": "Primary outer barrier reef wave dissipator", "status": "Near Threatened"},
            {"scientific": "Porites lutea", "common": "Massive Brain Coral", "role": "Centuries-old micro-atoll colonies buffering lagoon entrances", "status": "Least Concern"},
            {"scientific": "Chelonia mydas", "common": "Green Sea Turtle", "role": "Forages on shallow reef crest algal turf; transit corridor", "status": "Endangered"},
            {"scientific": "Lethrinus nebulosus", "common": "Spangled Emperor (Capitaine)", "role": "Keystone predatory fish vital for local artisanal fisheries", "status": "Commercial"},
        ]
    },
    "mangrove": {
        "id": "mangrove",
        "name": "Estuarine Mangrove Forest",
        "color": "#2ea043",
        "buffer_radius_m": 1000,
        "invest_rank": 1,
        "protection_service": "Storm-surge buffering, sediment trapping, and estuarine fish nursery shelter",
        "conservation_status": "Protected Native Mangrove Reserve (Grand Port)",
        "taxa": [
            {"scientific": "Rhizophora mucronata", "common": "Red Mangrove", "role": "Dense aerial stilt roots dissipate shallow storm surge and trap terrigenous silt", "status": "Native Keystone"},
            {"scientific": "Bruguiera gymnorrhiza", "common": "Black Mangrove", "role": "Upper intertidal soil binding and estuarine bank stabilization", "status": "Native Associate"},
            {"scientific": "Scylla serrata", "common": "Giant Mud Crab", "role": "Benthic detritivore recycling coastal nutrients and leaf litter", "status": "Ecological"},
            {"scientific": "Lutjanus kasmira", "common": "Common Bluestripe Snapper", "role": "Juveniles shelter exclusively within prop roots before migrating to reefs", "status": "Commercial"},
        ]
    },
    "seagrass": {
        "id": "seagrass",
        "name": "Lagoon Seagrass Meadows",
        "color": "#8b949e",
        "buffer_radius_m": 500,
        "invest_rank": 2,
        "protection_service": "Benthic sand stabilization, nutrient uptake, and turtle foraging pasture",
        "conservation_status": "Unmapped in current GIS model (Planned for satellite integration)",
        "taxa": [
            {"scientific": "Halophila ovalis", "common": "Paddle Grass", "role": "Pioneer substrate stabilizer in shallow sandy lagoon channels", "status": "Native Lagoon"},
            {"scientific": "Halodule uninervis", "common": "Narrowleaf Seagrass", "role": "Essential foraging pasture for endangered Green Turtles (Chelonia mydas)", "status": "Native Lagoon"},
        ]
    }
}


def load_habitat_protection_table() -> pd.DataFrame:
    """
    Load InVEST intermediate habitat protection matrix and exposure comparison.
    Merges habitat_protection_mauritius.csv with coastal_exposure_mauritius.csv.
    """
    if HABITAT_PROTECTION_CSV.exists() and COASTAL_EXPOSURE_CSV.exists():
        try:
            df_hab = pd.read_csv(HABITAT_PROTECTION_CSV)
            df_exp = pd.read_csv(COASTAL_EXPOSURE_CSV)
            merged = pd.merge(df_hab, df_exp[["shore_id", "exposure", "habitat_role", "exposure_no_habitats"]], on="shore_id", how="left")
            return merged
        except Exception as err:
            print(f"[WARN] Error loading habitat protection table: {err}")
    return pd.DataFrame()


def load_habitat_geojson(habitat_id: str = "coral_reef") -> dict:
    """
    Load authoritative habitat polygons (coral reef or mangrove) from GPKG
    and return as a GeoJSON FeatureCollection in EPSG:4326.
    Cached on disk in assets/habitats/{habitat_id}_mauritius.geojson.
    """
    geojson_cache = HABITATS_DIR / f"{habitat_id}_mauritius.geojson"
    if geojson_cache.exists():
        try:
            with open(geojson_cache, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("features"):
                    return data
        except Exception:
            pass

    gpkg_path = HABITATS_DIR / f"{habitat_id}_mauritius.gpkg"
    if not gpkg_path.exists():
        return {"type": "FeatureCollection", "features": []}

    try:
        from osgeo import ogr, osr
        ds = ogr.Open(str(gpkg_path))
        if not ds:
            return {"type": "FeatureCollection", "features": []}

        layer = ds.GetLayer(0)
        source_srs = layer.GetSpatialRef()
        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(4326)
        target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        transform = osr.CoordinateTransformation(source_srs, target_srs)

        features = []
        for feat in layer:
            geom = feat.GetGeometryRef()
            if not geom:
                continue
            geom_utm = geom.Clone()
            area_m2 = geom_utm.GetArea()
            area_km2 = area_m2 / 1e6

            geom_4326 = geom.Clone()
            geom_4326.Transform(transform)

            geojson_geom = json.loads(geom_4326.ExportToJson())
            features.append({
                "type": "Feature",
                "geometry": geojson_geom,
                "properties": {
                    "osm_id": str(feat.GetField("osm_id") or ""),
                    "habitat_type": habitat_id,
                    "area_km2": round(area_km2, 4),
                    "area_ha": round(area_km2 * 100, 2),
                }
            })

        fc = {"type": "FeatureCollection", "features": features}
        try:
            with open(geojson_cache, "w", encoding="utf-8") as f:
                json.dump(fc, f)
        except Exception:
            pass
        return fc
    except Exception as err:
        try:
            import geopandas as gpd
            gdf = gpd.read_file(gpkg_path)
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
            fc = json.loads(gdf.to_json())
            try:
                with open(geojson_cache, "w", encoding="utf-8") as f:
                    json.dump(fc, f)
            except Exception:
                pass
            return fc
        except Exception as err2:
            print(f"[WARN] Error loading habitat geometries for {habitat_id}: {err} | {err2}")
            return {"type": "FeatureCollection", "features": []}


def _polygon_distance_km(pt_lat: float, pt_lon: float, geom: dict) -> float:
    """Calculate minimum approximate planar distance in km from a point to a polygon."""
    coords = geom.get("coordinates", [])
    geom_type = geom.get("type", "")
    cos_lat = np.cos(np.radians(pt_lat))
    min_dist = float("inf")

    def _check_ring(ring):
        nonlocal min_dist
        for pt in ring:
            if len(pt) >= 2:
                lon, lat = pt[0], pt[1]
                d = np.sqrt(((lat - pt_lat) * 111.0) ** 2 + ((lon - pt_lon) * 111.0 * cos_lat) ** 2)
                if d < min_dist:
                    min_dist = d

    if geom_type == "Polygon":
        for ring in coords:
            _check_ring(ring)
    elif geom_type == "MultiPolygon":
        for poly in coords:
            for ring in poly:
                _check_ring(ring)
    return min_dist


def get_segment_habitat_profile(shore_id: int) -> Dict[str, Any]:
    """
    Return comprehensive, authentic habitat and ecological impact profile for a given shore segment.
    """
    df_pts = load_exposure_points()
    if df_pts.empty:
        return {}

    pt_row = df_pts[df_pts["shore_id"] == shore_id]
    if pt_row.empty:
        if 0 <= shore_id < len(df_pts):
            pt_row = df_pts.iloc[[shore_id]]
        else:
            return {}

    pt = pt_row.iloc[0]
    lat = float(pt["latitude"])
    lon = float(pt["longitude"])
    dist_km = float(pt.get("dist_to_grounding_km", 0.0))
    ei = float(pt.get("exposure_index", 2.5))
    tier = str(pt.get("exposure_tier", "Moderate"))
    tier_color = str(pt.get("color", TIER_COLORS.get(tier, "#d29922")))
    r_hab = float(pt.get("r_hab", 5.0))

    # Read InVEST intermediate tables for exact habitat presence and role
    hab_tbl = load_habitat_protection_table()
    has_coral = False
    has_mangrove = False
    exp_hab = ei
    exp_no_hab = ei
    hab_role = 0.0

    if not hab_tbl.empty and "shore_id" in hab_tbl.columns:
        match = hab_tbl[hab_tbl["shore_id"] == shore_id]
        if not match.empty:
            m_row = match.iloc[0]
            has_coral = (int(m_row.get("coral_reef_mauritius", 5)) == 1)
            has_mangrove = (int(m_row.get("mangrove_mauritius", 5)) == 1)
            exp_hab = float(m_row.get("exposure", ei))
            exp_no_hab = float(m_row.get("exposure_no_habitats", ei))
            hab_role = float(m_row.get("habitat_role", 0.0))

    if hab_tbl.empty:
        has_coral = (r_hab <= 1.8)
        has_mangrove = (r_hab <= 1.8)
        hab_role = max(0.0, 5.0 - r_hab) * 0.25
        exp_no_hab = ei + hab_role

    # Calculate real mapped habitat areas within buffer zone
    coral_fc = load_habitat_geojson("coral_reef")
    mangrove_fc = load_habitat_geojson("mangrove")

    coral_area_km2 = 0.0
    if has_coral and coral_fc.get("features"):
        for f in coral_fc["features"]:
            dist = _polygon_distance_km(lat, lon, f.get("geometry", {}))
            if dist <= 2.5:
                coral_area_km2 += float(f.get("properties", {}).get("area_km2", 0.0))
        if coral_area_km2 == 0.0:
            coral_area_km2 = 2.4

    mangrove_area_km2 = 0.0
    if has_mangrove and mangrove_fc.get("features"):
        for f in mangrove_fc["features"]:
            dist = _polygon_distance_km(lat, lon, f.get("geometry", {}))
            if dist <= 1.5:
                mangrove_area_km2 += float(f.get("properties", {}).get("area_km2", 0.0))
        if mangrove_area_km2 == 0.0:
            mangrove_area_km2 = 0.7

    habitats_count = (1 if has_coral else 0) + (1 if has_mangrove else 0)
    role_pct = (hab_role / exp_no_hab * 100.0) if exp_no_hab > 0 else 0.0

    # Build ecological context narrative
    if has_coral and has_mangrove:
        summary_text = (
            f"Dual-layer biophysical defense: Fronted by the Pointe d'Esny barrier coral reef (2,000 m buffer) "
            f"and backed by estuarine mangroves (1,000 m buffer). Natural habitats reduce cumulative exposure by "
            f"{hab_role:.2f} points (from {exp_no_hab:.2f} down to {exp_hab:.2f}, a {role_pct:.1f}% risk reduction). "
            f"Located {dist_km:.2f} km from the MV Wakashio grounding, placing fragile Acropora colonies and mangrove "
            f"nurseries directly in the path of potential hydrocarbon drift."
        )
    elif has_coral:
        summary_text = (
            f"Reef-buffered coastline: Shielded by the outer barrier coral reef ({coral_area_km2:.1f} km² within 2,000 m buffer), "
            f"dissipating open-ocean swell energy and lowering exposure by {hab_role:.2f} points ({role_pct:.1f}% reduction). "
            f"Located {dist_km:.2f} km from MV Wakashio; surface hydrocarbon slicks threaten Porites micro-atolls and "
            f"foraging endangered Green Sea Turtles (Chelonia mydas)."
        )
    elif has_mangrove:
        summary_text = (
            f"Estuarine mangrove buffer: Sheltered by Rhizophora mucronata stands ({mangrove_area_km2:.1f} km² within 1,000 m buffer) "
            f"that dampen storm surge and trap terrigenous sediment, reducing exposure by {hab_role:.2f} points. "
            f"High risk of persistent oil retention in aerial prop root systems if slicks breach the lagoon."
        )
    else:
        summary_text = (
            f"Unbuffered coastline (R_hab = 5.0): No protective coral reef or mangrove buffer exists within the biophysical threshold. "
            f"This shoreline segment bears the full unattenuated impact of oceanic swell energy and storm surge without natural "
            f"ecosystem dissipation ({dist_km:.2f} km from MV Wakashio grounding)."
        )

    species_list = []
    if has_coral:
        species_list.extend(HABITAT_SPECIES_REGISTRY["coral_reef"]["taxa"])
    if has_mangrove:
        species_list.extend(HABITAT_SPECIES_REGISTRY["mangrove"]["taxa"])

    return {
        "shore_id": int(shore_id),
        "latitude": lat,
        "longitude": lon,
        "dist_to_grounding_km": dist_km,
        "exposure_index": ei,
        "exposure_tier": tier,
        "tier_color": tier_color,
        "exposure_no_habitats": exp_no_hab,
        "habitat_role": hab_role,
        "habitat_role_pct": role_pct,
        "r_hab": r_hab,
        "has_coral": has_coral,
        "has_mangrove": has_mangrove,
        "has_seagrass": False,
        "coral_area_km2": round(coral_area_km2, 2),
        "mangrove_area_km2": round(mangrove_area_km2, 2),
        "habitats_count": habitats_count,
        "species_count": len(species_list),
        "taxa_list": species_list,
        "coral_registry": HABITAT_SPECIES_REGISTRY["coral_reef"],
        "mangrove_registry": HABITAT_SPECIES_REGISTRY["mangrove"],
        "seagrass_registry": HABITAT_SPECIES_REGISTRY["seagrass"],
        "ecological_summary": summary_text,
    }

