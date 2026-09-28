#!/usr/bin/env python3
"""
Convert ERA5 reanalysis wind and wave timeseries into an InVEST Coastal
Vulnerability-compatible WaveWatch III (WWIII) vector layer.

Output:
    assets/coastal_vulnerability/wwiii_era5_mauritius.gpkg

Layer:
    wwiii_era5 (Point, EPSG:4326)

Fields:
    All 80 required InVEST directional fields across 16 equiangular sectors:
    - REI_PCT{sector}  : percentage of highest-10% wind observations in sector
    - REI_V{sector}    : average wind speed (m/s) of highest-10% observations in sector
    - WavPPCT{sector}  : percentage of highest-10% wave power observations in sector
    - WavP_{sector}    : average wave power (kW/m) of highest-10% observations in sector
    - V10PCT_{sector}  : average highest-10% wind speed (m/s) in sector
    plus provenance fields: wind_n, wave_n, wind_max, wave_power_max.

Assumptions & Methodology:
    1. Meteorological wind direction is computed from eastward (u10) and
       northward (v10) components:
           dir = (270 - arctan2(v10, u10) * 180 / pi) % 360
       representing the direction FROM which the wind blows.
    2. Deep-water wave power is approximated as:
           P = rho * g^2 / (64 * pi) * Hs^2 * Te ~= 0.49 * swh^2 * mwp  [kW/m]
       using ERA5 significant wave height (swh) and mean wave period (mwp).
    3. The highest 10% events are thresholded at the 90th percentile across the
       time series for each grid cell, consistent with InVEST WWIII processing.
    4. Observations are binned into 16 compass sectors centered at:
       [0, 22, 45, 67, 90, 112, 135, 157, 180, 202, 225, 247, 270, 292, 315, 337]
       with a sector half-width of 11.25 degrees.
    5. Sectors with zero high-energy observations are assigned 0.0, avoiding nulls.
    6. Wave statistics are interpolated to match the wind grid positions.
"""
from pathlib import Path
import sys
import numpy as np
from netCDF4 import Dataset
from osgeo import ogr, osr

PROJECT_ROOT = Path(__file__).resolve().parent.parent

WIND_FILE = PROJECT_ROOT / (
    "assets/significant_height_of_combined_wind_waves_swell/"
    "reanalysis-era5-single-levels-timeseries-sfcm_puf_m6.nc"
)

WAVE_FILE = PROJECT_ROOT / (
    "assets/significant_height_of_combined_wind_waves_swell/"
    "reanalysis-era5-single-levels-timeseries-wavx0_vb0k1.nc"
)

OUTPUT_GPKG = PROJECT_ROOT / "assets/coastal_vulnerability/wwiii_era5_mauritius.gpkg"

SECTORS = [
    0, 22, 45, 67, 90, 112, 135, 157,
    180, 202, 225, 247, 270, 292, 315, 337
]


def sector_index(direction: float) -> int:
    """Assign a direction in [0, 360) to the nearest 22.5-degree sector [0..15]."""
    direction = direction % 360.0
    return int(np.floor((direction + 11.25) / 22.5)) % 16


def wind_direction_from_uv(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute meteorological wind direction (direction FROM which wind blows).
    ERA5: u10 is eastward component, v10 is northward component.
    """
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


def wave_power_kw_m(hs: np.ndarray, period: np.ndarray) -> np.ndarray:
    """
    Deep-water wave power approximation in kW/m.
    P = (rho * g^2) / (64 * pi) * Hs^2 * Te ~= 0.49 * Hs^2 * Te
    """
    return 0.49 * (hs ** 2) * period


def directional_statistics(values: np.ndarray, directions: np.ndarray):
    """
    Calculate directional statistics for the highest 10% of observations.

    Returns:
        pct : np.ndarray of shape (16,) with % of highest-10% events in each sector
        avg : np.ndarray of shape (16,) with average value in each sector
    """
    values = np.asarray(values, dtype=float)
    directions = np.asarray(directions, dtype=float)

    valid = np.isfinite(values) & np.isfinite(directions)
    values = values[valid]
    directions = directions[valid]

    result_pct = np.zeros(16, dtype=float)
    result_avg = np.zeros(16, dtype=float)

    if len(values) == 0:
        return result_pct, result_avg

    # 90th percentile threshold for top 10%
    threshold = np.percentile(values, 90.0)
    high = values >= threshold

    high_values = values[high]
    high_directions = directions[high]
    total_high = len(high_values)

    if total_high == 0:
        return result_pct, result_avg

    counts = np.zeros(16, dtype=int)
    sums = np.zeros(16, dtype=float)

    for val, direc in zip(high_values, high_directions):
        idx = sector_index(direc)
        counts[idx] += 1
        sums[idx] += val

    for i in range(16):
        if counts[i] > 0:
            result_pct[i] = 100.0 * counts[i] / total_high
            result_avg[i] = sums[i] / counts[i]

    return result_pct, result_avg


def add_field(layer: ogr.Layer, name: str, width: int = 24, precision: int = 8):
    """Add a real field to an OGR layer."""
    field_defn = ogr.FieldDefn(name, ogr.OFTReal)
    field_defn.SetWidth(width)
    field_defn.SetPrecision(precision)
    layer.CreateField(field_defn)


def build_wwiii_gpkg():
    print("=" * 60)
    print("InVEST WWIII Vector Generation from ERA5 NetCDF")
    print("=" * 60)

    if not WIND_FILE.exists():
        raise FileNotFoundError(f"Missing ERA5 wind file: {WIND_FILE}")
    if not WAVE_FILE.exists():
        raise FileNotFoundError(f"Missing ERA5 wave file: {WAVE_FILE}")

    print(f"Reading wind NetCDF: {WIND_FILE}")
    with Dataset(WIND_FILE) as ds:
        lats = np.asarray(ds.variables["latitude"][:], dtype=float)
        lons = np.asarray(ds.variables["longitude"][:], dtype=float)
        u10 = np.asarray(ds.variables["u10"][:], dtype=float)
        v10 = np.asarray(ds.variables["v10"][:], dtype=float)

    print(f"  Wind grid: {len(lats)} lats x {len(lons)} lons, {u10.shape[0]} time steps")

    print(f"Reading wave NetCDF: {WAVE_FILE}")
    with Dataset(WAVE_FILE) as ds:
        wave_lats = np.asarray(ds.variables["latitude"][:], dtype=float)
        wave_lons = np.asarray(ds.variables["longitude"][:], dtype=float)
        mwd = np.asarray(ds.variables["mwd"][:], dtype=float)
        mwp = np.asarray(ds.variables["mwp"][:], dtype=float)
        swh = np.asarray(ds.variables["swh"][:], dtype=float)

    print(f"  Wave grid: {len(wave_lats)} lats x {len(wave_lons)} lons, {swh.shape[0]} time steps")

    # Compute wave statistics per wave grid cell
    wave_stats = {}
    for wi, wlat in enumerate(wave_lats):
        for wj, wlon in enumerate(wave_lons):
            hs_cell = swh[:, wi, wj]
            period_cell = mwp[:, wi, wj]
            direction_cell = mwd[:, wi, wj]

            power = wave_power_kw_m(hs_cell, period_cell)
            valid = (
                np.isfinite(power) &
                np.isfinite(direction_cell) &
                (hs_cell >= 0) &
                (period_cell > 0) &
                (power >= 0)
            )

            p_valid = power[valid]
            d_valid = direction_cell[valid]

            wp_pct, wp_avg = directional_statistics(p_valid, d_valid)
            wave_stats[(float(wlat), float(wlon))] = {
                "pct": wp_pct,
                "avg": wp_avg,
                "n": int(np.sum(valid)),
                "max": float(np.nanmax(p_valid)) if len(p_valid) > 0 else 0.0
            }

    wave_points = sorted(wave_stats.keys())

    def interpolate_wave_stats(lat: float, lon: float):
        """Interpolate wave statistics along longitude."""
        p_west = wave_stats[wave_points[0]]
        p_east = wave_stats[wave_points[-1]]
        lon_w = wave_points[0][1]
        lon_e = wave_points[-1][1]

        if lon <= lon_w:
            w = 0.0
        elif lon >= lon_e:
            w = 1.0
        else:
            w = (lon - lon_w) / (lon_e - lon_w)

        pct = (1.0 - w) * p_west["pct"] + w * p_east["pct"]
        avg = (1.0 - w) * p_west["avg"] + w * p_east["avg"]
        return pct, avg

    # Prepare output GeoPackage
    OUTPUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_GPKG.exists():
        OUTPUT_GPKG.unlink()

    driver = ogr.GetDriverByName("GPKG")
    ds_out = driver.CreateDataSource(str(OUTPUT_GPKG))

    srs = osr.SpatialReference()
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    srs.ImportFromEPSG(4326)

    layer = ds_out.CreateLayer("wwiii_era5", srs, ogr.wkbPoint)

    # Create all 80 InVEST directional fields
    for sector in SECTORS:
        add_field(layer, f"REI_PCT{sector}")
        add_field(layer, f"REI_V{sector}")
        add_field(layer, f"WavPPCT{sector}")
        add_field(layer, f"WavP_{sector}")
        add_field(layer, f"V10PCT_{sector}")

    # Provenance fields
    add_field(layer, "wind_n")
    add_field(layer, "wave_n")
    add_field(layer, "wind_max")
    add_field(layer, "wave_power_max")

    layer_defn = layer.GetLayerDefn()
    feature_count = 0

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            wind_u = u10[:, i, j]
            wind_v = v10[:, i, j]

            wind_speed = np.sqrt(wind_u ** 2 + wind_v ** 2)
            wind_direction = wind_direction_from_uv(wind_u, wind_v)

            valid = np.isfinite(wind_speed) & np.isfinite(wind_direction)
            speed_valid = wind_speed[valid]
            direction_valid = wind_direction[valid]

            rei_pct, rei_v = directional_statistics(speed_valid, direction_valid)

            # V10PCT: average wind speed (m/s) in top 10% events for each sector
            v10pct = np.zeros(16, dtype=float)
            thresh_wind = np.percentile(speed_valid, 90.0)
            high_mask = speed_valid >= thresh_wind
            h_spd = speed_valid[high_mask]
            h_dir = direction_valid[high_mask]

            for s_idx, sector in enumerate(SECTORS):
                delta = np.abs(((h_dir - sector + 180.0) % 360.0) - 180.0)
                in_sector = delta <= 11.25
                if np.any(in_sector):
                    v10pct[s_idx] = float(np.mean(h_spd[in_sector]))
                else:
                    v10pct[s_idx] = 0.0

            # Wave statistics for this point
            wave_pct, wave_avg = interpolate_wave_stats(float(lat), float(lon))

            feature = ogr.Feature(layer_defn)
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(float(lon), float(lat))
            feature.SetGeometry(point)

            for k, sector in enumerate(SECTORS):
                feature.SetField(f"REI_PCT{sector}", float(rei_pct[k]))
                feature.SetField(f"REI_V{sector}", float(rei_v[k]))
                feature.SetField(f"WavPPCT{sector}", float(wave_pct[k]))
                feature.SetField(f"WavP_{sector}", float(wave_avg[k]))
                feature.SetField(f"V10PCT_{sector}", float(v10pct[k]))

            feature.SetField("wind_n", int(len(speed_valid)))
            feature.SetField(
                "wave_n",
                int(np.mean([wave_stats[p]["n"] for p in wave_points]))
            )
            feature.SetField("wind_max", float(np.max(speed_valid)))
            feature.SetField(
                "wave_power_max",
                float(np.mean([wave_stats[p]["max"] for p in wave_points]))
            )

            layer.CreateFeature(feature)
            feature_count += 1

    ds_out = None
    print(f"\nSuccessfully generated: {OUTPUT_GPKG}")
    print(f"  Feature count: {feature_count} point features")
    print(f"  Field count: {len(SECTORS) * 5} InVEST directional fields + 4 provenance fields")
    print(f"  CRS: EPSG:4326 (WGS 84)")
    return feature_count


if __name__ == "__main__":
    count = build_wwiii_gpkg()
    sys.exit(0 if count > 0 else 1)
