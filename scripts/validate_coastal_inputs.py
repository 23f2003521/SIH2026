#!/usr/bin/env python3
"""
GDAL/OGR Validation Suite for InVEST Coastal Vulnerability Inputs.

Validates:
1. AOI Vector (mauritius_aoi_final.gpkg)
2. Landmass Vector (mauritius_land.gpkg)
3. Bathymetry Raster (gebco_bathymetry.tif)
4. DEM Raster (Copernicus GLO-30 DEM.tif)
5. Continental Shelf Contour Vector (shelf_contour_200m.gpkg)
6. WWIII Wind/Wave Vector (wwiii_era5_mauritius.gpkg) with all 80 required fields
7. Natural Habitats Table (natural_habitats.csv) and referenced spatial layers
"""
from pathlib import Path
import csv
import sys
from osgeo import gdal, ogr, osr

gdal.UseExceptions()
ogr.UseExceptions()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AOI_PATH = PROJECT_ROOT / "assets/coastal_vulnerability/mauritius_aoi_final.gpkg"
LANDMASS_PATH = PROJECT_ROOT / "assets/landmass/mauritius_land.gpkg"
BATHYMETRY_PATH = PROJECT_ROOT / "assets/derived/gebco_bathymetry.tif"
DEM_PATH = PROJECT_ROOT / (
    "assets/dem_glo30/DEM1_SAR_DGE_30_20110518T014310_20121016T014335_ADS_000000_tQoD_9f0619ea/"
    "Copernicus_DSM_10_S21_00_E057_00/DEM/Copernicus_DSM_10_S21_00_E057_00_DEM.tif"
)
SHELF_PATH = PROJECT_ROOT / "assets/coastal_vulnerability/shelf_contour_200m.gpkg"
WWIII_PATH = PROJECT_ROOT / "assets/coastal_vulnerability/wwiii_era5_mauritius.gpkg"
HABITAT_TABLE_PATH = PROJECT_ROOT / "assets/coastal_vulnerability/natural_habitats.csv"

SECTORS = [
    0, 22, 45, 67, 90, 112, 135, 157,
    180, 202, 225, 247, 270, 292, 315, 337
]

FIELD_PREFIXES = ["REI_PCT", "REI_V", "WavPPCT", "WavP_", "V10PCT_"]


def validate_vector(path: Path, expected_geom: str, expected_epsg: int = None, min_features: int = 1):
    print(f"--- Checking Vector: {path.relative_to(PROJECT_ROOT)} ---")
    if not path.exists():
        print(f"  [FAIL] File does not exist: {path}")
        return False

    ds = ogr.Open(str(path))
    if ds is None:
        print(f"  [FAIL] OGR cannot open data source: {path}")
        return False

    layer_count = ds.GetLayerCount()
    if layer_count == 0:
        print(f"  [FAIL] No layers in data source")
        return False

    layer = ds.GetLayer(0)
    layer_name = layer.GetName()
    feat_count = layer.GetFeatureCount()
    geom_type = ogr.GeometryTypeToName(layer.GetGeomType())
    srs = layer.GetSpatialRef()
    epsg = srs.GetAttrValue("AUTHORITY", 1) if srs else "None"

    print(f"  Layer Name: {layer_name}")
    print(f"  Geometry: {geom_type}")
    print(f"  Feature Count: {feat_count}")
    print(f"  CRS EPSG: {epsg}")

    passed = True
    def _norm(name: str) -> str:
        return name.lower().replace(" ", "").replace("_", "")

    if _norm(expected_geom) not in _norm(geom_type):
        print(f"  [FAIL] Expected geometry {expected_geom}, got {geom_type}")
        passed = False

    # Check for 3D/Z coordinates that cause InVEST failures
    if "25d" in geom_type.lower() or " z" in geom_type.lower():
        print(f"  [WARN] Geometry contains Z-dimension ({geom_type}); InVEST expects 2D geometries")

    if expected_epsg and str(epsg) != str(expected_epsg):
        print(f"  [WARN] Expected EPSG {expected_epsg}, got {epsg}")
    if feat_count < min_features:
        print(f"  [FAIL] Expected >= {min_features} features, got {feat_count}")
        passed = False

    if passed:
        print("  [PASS] Vector geometry and structure valid")
    return passed


def validate_raster(path: Path, expected_epsg: int = None):
    print(f"--- Checking Raster: {path.relative_to(PROJECT_ROOT)} ---")
    if not path.exists():
        print(f"  [FAIL] File does not exist: {path}")
        return False

    ds = gdal.Open(str(path))
    if ds is None:
        print(f"  [FAIL] GDAL cannot open raster: {path}")
        return False

    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    bands = ds.RasterCount
    proj = ds.GetProjection()
    srs = osr.SpatialReference(wkt=proj)
    epsg = srs.GetAttrValue("AUTHORITY", 1) if srs else "None"

    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    stats = band.GetStatistics(True, True)

    print(f"  Dimensions: {xsize} x {ysize}, Bands: {bands}")
    print(f"  CRS EPSG: {epsg}")
    print(f"  NoData: {nodata}")
    print(f"  Stats: Min={stats[0]:.2f}, Max={stats[1]:.2f}, Mean={stats[2]:.2f}, StdDev={stats[3]:.2f}")

    passed = True
    if bands < 1:
        print(f"  [FAIL] Raster has no bands")
        passed = False
    if expected_epsg and str(epsg) != str(expected_epsg):
        print(f"  [WARN] Expected EPSG {expected_epsg}, got {epsg}")

    if passed:
        print("  [PASS] Raster data and statistics valid")
    return passed


def validate_wwiii(path: Path):
    print(f"--- Checking WWIII Vector: {path.relative_to(PROJECT_ROOT)} ---")
    if not path.exists():
        print(f"  [FAIL] File does not exist: {path}")
        return False

    ds = ogr.Open(str(path))
    if ds is None:
        print(f"  [FAIL] OGR cannot open WWIII layer: {path}")
        return False

    layer = ds.GetLayer(0)
    feat_count = layer.GetFeatureCount()
    geom_type = ogr.GeometryTypeToName(layer.GetGeomType())
    srs = layer.GetSpatialRef()
    epsg = srs.GetAttrValue("AUTHORITY", 1) if srs else "None"

    print(f"  Layer Name: {layer.GetName()}")
    print(f"  Geometry: {geom_type}")
    print(f"  Feature Count: {feat_count}")
    print(f"  CRS EPSG: {epsg}")

    defn = layer.GetLayerDefn()
    existing_fields = {defn.GetFieldDefn(i).GetName(): i for i in range(defn.GetFieldCount())}

    # Verify all 80 required directional fields
    missing_fields = []
    required_80 = []
    for prefix in FIELD_PREFIXES:
        for s in SECTORS:
            f_name = f"{prefix}{s}"
            required_80.append(f_name)
            if f_name not in existing_fields:
                missing_fields.append(f_name)

    print(f"  Required InVEST directional fields: {len(required_80)}")
    print(f"  Total fields found in layer: {len(existing_fields)}")

    passed = True
    if "point" not in geom_type.lower():
        print(f"  [FAIL] Geometry must be Point, got {geom_type}")
        passed = False

    if missing_fields:
        print(f"  [FAIL] Missing {len(missing_fields)} directional fields: {missing_fields[:10]}...")
        passed = False
    else:
        print("  [PASS] All 80 required InVEST directional fields are present")

    # Check non-null values across features
    null_issues = 0
    feature = layer.GetNextFeature()
    while feature:
        for f_name in required_80:
            val = feature.GetField(f_name)
            if val is None:
                null_issues += 1
        feature = layer.GetNextFeature()

    if null_issues > 0:
        print(f"  [FAIL] Found {null_issues} NULL values in required fields")
        passed = False
    else:
        print("  [PASS] No NULL values in required fields")

    return passed


def validate_habitats(table_path: Path):
    print(f"--- Checking Natural Habitats Table: {table_path.relative_to(PROJECT_ROOT)} ---")
    if not table_path.exists():
        print(f"  [FAIL] Habitats table does not exist: {table_path}")
        return False

    with open(table_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = [h.strip() for h in reader.fieldnames]
        rows = list(reader)

    print(f"  Headers: {headers}")
    print(f"  Row count: {len(rows)}")

    required_cols = ["id", "path", "rank", "protection distance (m)"]
    missing_cols = [c for c in required_cols if c not in headers]
    if missing_cols:
        print(f"  [FAIL] Missing required CSV columns: {missing_cols}")
        return False

    passed = True
    for row in rows:
        hid = row["id"]
        hpath_str = row["path"]
        hrank = row["rank"]
        hdist = row["protection distance (m)"]

        # Validate rank 1-5
        try:
            r_int = int(hrank)
            if not 1 <= r_int <= 5:
                print(f"  [FAIL] Habitat '{hid}' rank {r_int} outside [1, 5]")
                passed = False
        except ValueError:
            print(f"  [FAIL] Habitat '{hid}' rank '{hrank}' is not an integer")
            passed = False

        # Validate distance > 0
        try:
            d_val = float(hdist)
            if d_val <= 0:
                print(f"  [FAIL] Habitat '{hid}' distance {d_val} <= 0")
                passed = False
        except ValueError:
            print(f"  [FAIL] Habitat '{hid}' distance '{hdist}' is not a number")
            passed = False

        # Resolve path relative to the CSV file's directory (matching InVEST model behavior)
        if Path(hpath_str).is_absolute():
            hpath = Path(hpath_str)
        else:
            hpath = (table_path.parent / hpath_str).resolve()

        if not hpath.exists():
            print(f"  [FAIL] Habitat '{hid}' layer file not found: {hpath}")
            passed = False
        else:
            h_ds = ogr.Open(str(hpath))
            if h_ds is None:
                print(f"  [FAIL] Habitat '{hid}' vector could not be opened by OGR")
                passed = False
            else:
                h_layer = h_ds.GetLayer(0)
                print(f"  [PASS] Habitat '{hid}' (rank={hrank}, dist={hdist}m) -> {h_layer.GetFeatureCount()} features")

    return passed


def main():
    print("=" * 70)
    print("GDAL/OGR InVEST Coastal Vulnerability Input Verification")
    print("=" * 70)

    results = {}
    results["aoi"] = validate_vector(AOI_PATH, "Polygon", expected_epsg=32740)
    results["landmass"] = validate_vector(LANDMASS_PATH, "Polygon", expected_epsg=32740)
    results["bathymetry"] = validate_raster(BATHYMETRY_PATH)
    results["dem"] = validate_raster(DEM_PATH)
    results["shelf_contour"] = validate_vector(SHELF_PATH, "LineString", expected_epsg=32740, min_features=1)
    results["wwiii"] = validate_wwiii(WWIII_PATH)
    results["habitats"] = validate_habitats(HABITAT_TABLE_PATH)

    print("\n" + "=" * 70)
    print("SUMMARY OF INPUT READINESS")
    print("=" * 70)
    all_ready = True
    for k, v in results.items():
        status = "READY" if v else "MISSING / INVALID"
        print(f"  {k:20s}: {status}")
        if not v:
            all_ready = False

    print("=" * 70)
    return 0 if all_ready else 1


if __name__ == "__main__":
    sys.exit(main())
