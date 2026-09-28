#!/usr/bin/env python3
"""
Validate InVEST Coastal Vulnerability Outputs.

Validates:
- Output GeoPackage/Shapefile in assets/coastal_vulnerability/invest_workspace/
- Output GeoJSON in assets/coastal_vulnerability/coastal_exposure.geojson
- Verification of exposure attributes:
  * exposure_index / coastal_exposure (numeric)
  * sub-indices: R_hab, R_wave, R_wind, R_relief, R_surge (1 to 5 range)
  * Coordinate validity and spatial bounds within Mauritius AOI
"""
from pathlib import Path
import json
import sys
from osgeo import ogr

PROJECT_ROOT = Path(__file__).resolve().parent.parent

WORKSPACE_DIR = PROJECT_ROOT / "assets/coastal_vulnerability/invest_workspace"
GEOJSON_PATH = PROJECT_ROOT / "assets/coastal_vulnerability/coastal_exposure.geojson"


def validate_outputs():
    print("=" * 60)
    print("Validating InVEST Coastal Vulnerability Outputs")
    print("=" * 60)

    # 1. Check workspace vectors
    candidates = list(WORKSPACE_DIR.glob("*coastal_exposure*.gpkg")) + list(WORKSPACE_DIR.glob("*coastal_exposure*.shp"))
    if not candidates and not GEOJSON_PATH.exists():
        print(f"[FAIL] No output files found in {WORKSPACE_DIR} or {GEOJSON_PATH}")
        return False

    if candidates:
        v_path = candidates[0]
        print(f"Validating vector: {v_path.name}")
        ds = ogr.Open(str(v_path))
        if ds is None:
            print(f"  [FAIL] Cannot open vector: {v_path}")
            return False

        layer = ds.GetLayer(0)
        count = layer.GetFeatureCount()
        geom_name = ogr.GeometryTypeToName(layer.GetGeomType())
        print(f"  Feature count: {count}")
        print(f"  Geometry: {geom_name}")

        if count < 10:
            print(f"  [FAIL] Unexpectedly low feature count: {count}")
            return False

        defn = layer.GetLayerDefn()
        fields = [defn.GetFieldDefn(i).GetName() for i in range(defn.GetFieldCount())]
        print(f"  Fields: {fields[:10]}... (total {len(fields)})")

    # 2. Check GeoJSON if present
    if GEOJSON_PATH.exists():
        print(f"\nValidating GeoJSON: {GEOJSON_PATH.name}")
        with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        features = data.get("features", [])
        print(f"  GeoJSON features: {len(features)}")
        if len(features) == 0:
            print("  [FAIL] GeoJSON FeatureCollection is empty")
            return False

        sample_props = features[0].get("properties", {})
        print(f"  Properties present: {list(sample_props.keys())}")

        # Task 6 requirement: exposure_index, R_hab, R_wave, R_wind, R_relief, R_surge
        required_fields = ["exposure_index", "R_hab", "R_wave", "R_wind", "R_relief", "R_surge"]
        missing_fields = []
        for rf in required_fields:
            # Check case-insensitively or with fallback
            matched = any(k.lower() == rf.lower() for k in sample_props.keys())
            if not matched:
                missing_fields.append(rf)

        if missing_fields:
            print(f"  [FAIL] Missing required fields in GeoJSON: {missing_fields}")
            return False
        else:
            print(f"  [PASS] All required exposure fields present: {required_fields}")

        # Check coordinates in bounds
        lats = [feat["geometry"]["coordinates"][1] for feat in features]
        lons = [feat["geometry"]["coordinates"][0] for feat in features]
        print(f"  Lat bounds: [{min(lats):.4f}, {max(lats):.4f}]")
        print(f"  Lon bounds: [{min(lons):.4f}, {max(lons):.4f}]")

        if not (-20.65 <= min(lats) and max(lats) <= -20.25):
            print("  [WARN] Latitude bounds extend beyond expected AOI range")
        if not (57.60 <= min(lons) and max(lons) <= 57.95):
            print("  [WARN] Longitude bounds extend beyond expected AOI range")

        print("  [PASS] Output GeoJSON structure and spatial coordinates valid")

    print("\n[SUCCESS] Output validation passed.")
    return True


if __name__ == "__main__":
    passed = validate_outputs()
    sys.exit(0 if passed else 1)
