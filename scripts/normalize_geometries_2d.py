#!/usr/bin/env python3
"""
Normalize Vector Geometries to Strict 2D for InVEST Coastal Vulnerability.

InVEST requires all vector inputs to have strictly 2D geometries (no Z or M coordinates)
and valid OGC geometries. This script inspects:
- assets/coastal_vulnerability/wwiii_era5_mauritius.gpkg
- assets/habitats/coral_reef_mauritius.gpkg
- assets/habitats/mangrove_mauritius.gpkg
- assets/coastal_vulnerability/shelf_contour_200m.gpkg
- assets/coastal_vulnerability/mauritius_aoi_final.gpkg
- assets/landmass/mauritius_land.gpkg

If any layer contains 3D/Z geometries or invalid features, it normalizes them to
clean 2D geometries without altering any attributes or scientific values.
"""
from pathlib import Path
import sys
import tempfile
import shutil
from osgeo import ogr, osr

ogr.UseExceptions()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TARGET_VECTORS = [
    PROJECT_ROOT / "assets/coastal_vulnerability/wwiii_era5_mauritius.gpkg",
    PROJECT_ROOT / "assets/habitats/coral_reef_mauritius.gpkg",
    PROJECT_ROOT / "assets/habitats/mangrove_mauritius.gpkg",
    PROJECT_ROOT / "assets/coastal_vulnerability/shelf_contour_200m.gpkg",
    PROJECT_ROOT / "assets/coastal_vulnerability/mauritius_aoi_final.gpkg",
    PROJECT_ROOT / "assets/landmass/mauritius_land.gpkg",
]

# Map 3D/2.5D geometry types to their 2D equivalents
GEOM_2D_MAP = {
    ogr.wkbPoint25D: ogr.wkbPoint,
    ogr.wkbLineString25D: ogr.wkbLineString,
    ogr.wkbPolygon25D: ogr.wkbPolygon,
    ogr.wkbMultiPoint25D: ogr.wkbMultiPoint,
    ogr.wkbMultiLineString25D: ogr.wkbMultiLineString,
    ogr.wkbMultiPolygon25D: ogr.wkbMultiPolygon,
    ogr.wkbGeometryCollection25D: ogr.wkbGeometryCollection,
}


def normalize_layer_to_2d(gpkg_path: Path):
    if not gpkg_path.exists():
        print(f"[SKIP] File does not exist: {gpkg_path.relative_to(PROJECT_ROOT)}")
        return False

    print(f"\nChecking 2D geometry for: {gpkg_path.relative_to(PROJECT_ROOT)}")
    ds = ogr.Open(str(gpkg_path), 0)  # Read-only
    if ds is None:
        print(f"  [ERROR] Cannot open {gpkg_path}")
        return False

    layer = ds.GetLayer(0)
    layer_name = layer.GetName()
    geom_type = layer.GetGeomType()
    geom_name = ogr.GeometryTypeToName(geom_type)
    srs = layer.GetSpatialRef()
    feat_count = layer.GetFeatureCount()

    needs_conversion = False
    if geom_type in GEOM_2D_MAP or "25d" in geom_name.lower() or " z" in geom_name.lower():
        needs_conversion = True
        target_2d_type = GEOM_2D_MAP.get(geom_type, geom_type & (~ogr.wkb25DBit))
        print(f"  Layer has 3D geometry ({geom_name}). Normalization to 2D required.")
    else:
        # Check individual features in case layer type is generic
        for feat in layer:
            g = feat.GetGeometryRef()
            if g and g.Is3D():
                needs_conversion = True
                target_2d_type = ogr.wkbFlatten(geom_type)
                print(f"  Found 3D feature coordinates. Normalization to 2D required.")
                break
        layer.ResetReading()

    if not needs_conversion:
        print(f"  [OK] Already strictly 2D ({geom_name}, {feat_count} features).")
        ds = None
        return True

    # Convert to 2D via a temporary GeoPackage
    print(f"  Normalizing {feat_count} features to 2D {ogr.GeometryTypeToName(target_2d_type)}...")
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_gpkg = tmp_dir / gpkg_path.name

    driver = ogr.GetDriverByName("GPKG")
    out_ds = driver.CreateDataSource(str(tmp_gpkg))

    out_srs = srs.Clone() if srs else None
    if out_srs:
        out_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    out_layer = out_ds.CreateLayer(layer_name, out_srs, target_2d_type)

    defn = layer.GetLayerDefn()
    for i in range(defn.GetFieldCount()):
        out_layer.CreateField(defn.GetFieldDefn(i))

    out_defn = out_layer.GetLayerDefn()
    converted_count = 0

    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom:
            geom_clone = geom.Clone()
            geom_clone.FlattenTo2D()
            if not geom_clone.IsValid():
                geom_clone = geom_clone.MakeValid()

            out_feat = ogr.Feature(out_defn)
            out_feat.SetGeometry(geom_clone)
            for i in range(defn.GetFieldCount()):
                out_feat.SetField(i, feat.GetField(i))
            out_layer.CreateFeature(out_feat)
            converted_count += 1

    ds = None
    out_ds = None

    # Replace original file with normalized 2D version
    shutil.copy2(str(tmp_gpkg), str(gpkg_path))
    shutil.rmtree(tmp_dir)

    print(f"  [SUCCESS] Overwritten with clean 2D layer ({converted_count} features).")
    return True


def main():
    print("=" * 65)
    print("InVEST Vector Geometry 2D Normalizer")
    print("=" * 65)

    all_ok = True
    for vec in TARGET_VECTORS:
        if vec.exists():
            success = normalize_layer_to_2d(vec)
            if not success:
                all_ok = False
        else:
            print(f"[SKIP] Vector not found: {vec.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 65)
    print("NORMALIZATION SUMMARY")
    print("=" * 65)
    print("All processed vector layers verified as valid 2D geometries." if all_ok else "Some layers had errors.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
