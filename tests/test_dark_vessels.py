"""
Tests for Dark Vessel Detection & AIS Transponder Gap Analysis.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from api.main import app
from poseatsea import dark_vessels
from poseatsea.scenario.real_ais import build_scenario


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_pixel_to_latlon_center_and_invertibility():
    """Verify projection matches center and is perfectly invertible."""
    c_lat, c_lon = -20.4442, 57.7433
    w, h = 1250, 650
    pixel_res = 10.0

    # Center pixel (w/2, h/2) must project to exactly (c_lat, c_lon)
    lat_center, lon_center = dark_vessels.sar_pixel_to_latlon(
        w / 2.0, h / 2.0, c_lat, c_lon, w, h, pixel_res
    )
    assert lat_center == pytest.approx(c_lat, abs=1e-5)
    assert lon_center == pytest.approx(c_lon, abs=1e-5)

    # Invertibility test across arbitrary coordinates
    for test_x, test_y in [(100.0, 150.0), (1050.0, 520.0), (625.0, 325.0)]:
        lat, lon = dark_vessels.sar_pixel_to_latlon(
            test_x, test_y, c_lat, c_lon, w, h, pixel_res
        )
        inv_x, inv_y = dark_vessels.latlon_to_sar_pixel(
            lat, lon, c_lat, c_lon, w, h, pixel_res
        )
        assert inv_x == pytest.approx(test_x, abs=1e-2)
        assert inv_y == pytest.approx(test_y, abs=1e-2)


def test_extract_radar_targets_from_mask():
    """Verify ship components (class 3) are extracted from mask with georeferencing."""
    mask = np.zeros((300, 400), dtype=np.uint8)
    # Target 1: small ship
    mask[50:60, 50:70] = 3
    # Target 2: larger ship
    mask[200:220, 300:330] = 3

    targets = dark_vessels.extract_radar_targets_from_mask(
        mask, center_lat=-20.44, center_lon=57.74, pixel_res_m=10.0
    )
    assert len(targets) == 2
    # Sorted by area descending
    assert targets[0].pixels == 20 * 30  # 600
    assert targets[1].pixels == 10 * 20  # 200
    assert targets[0].estimated_length_m >= 300.0  # 30 px * 10m


def test_match_sar_ships_to_ais_identifies_dark_vessels():
    """Verify cooperative ships match to AIS and unmatched targets flag as dark vessels."""
    c_lat, c_lon = -20.4442, 57.7433

    target_coop = dark_vessels.RadarTarget(
        target_id=1,
        centroid_xy=(625.0, 325.0),
        bbox=(620, 320, 10, 10),
        pixels=100,
        latitude=c_lat,
        longitude=c_lon,
        estimated_length_m=100.0,
    )

    # Dark target 20 km away
    target_dark = dark_vessels.RadarTarget(
        target_id=2,
        centroid_xy=(100.0, 100.0),
        bbox=(95, 95, 10, 10),
        pixels=80,
        latitude=c_lat + 0.18,
        longitude=c_lon + 0.18,
        estimated_length_m=80.0,
    )

    now = datetime(2020, 7, 25, 12, 0, tzinfo=timezone.utc)
    ais_data = pd.DataFrame([
        {
            "mmsi": 999123456,
            "vessel_name": "COOPERATIVE ONE",
            "flag": "Panama",
            "vessel_type": "Tanker",
            "latitude": c_lat + 0.001,  # ~110m away
            "longitude": c_lon + 0.001,
            "speed": 11.5,
            "timestamp": now.isoformat(),
        }
    ])

    matched = dark_vessels.match_sar_ships_to_ais(
        [target_coop, target_dark],
        ais_df=ais_data,
        acquisition_time=now,
        spill_lat=c_lat,
        spill_lon=c_lon,
        gating_radius_m=2000.0,
    )

    # Target 1 should be cooperative
    t1 = next(t for t in matched if t.target_id == 1)
    assert t1.is_matched is True
    assert t1.matched_mmsi == 999123456
    assert t1.matched_vessel_name == "COOPERATIVE ONE"
    assert t1.risk_level == "normal"

    # Target 2 should be dark vessel
    t2 = next(t for t in matched if t.target_id == 2)
    assert t2.is_matched is False
    assert t2.matched_mmsi is None
    assert "DARK VESSEL" in t2.as_dict()["status"]


def test_detect_transponder_gaps():
    """Verify detection of silent gaps for moving vessels."""
    times = [
        datetime(2020, 7, 25, 10, 0, tzinfo=timezone.utc),
        datetime(2020, 7, 25, 10, 5, tzinfo=timezone.utc),
        # 60 minute gap underway at 12 knots
        datetime(2020, 7, 25, 11, 5, tzinfo=timezone.utc),
        datetime(2020, 7, 25, 11, 10, tzinfo=timezone.utc),
    ]
    df = pd.DataFrame([
        {"mmsi": 111222333, "vessel_name": "GHOST VESSEL", "flag": "Liberia",
         "vessel_type": "Cargo", "timestamp": times[0], "latitude": -20.40,
         "longitude": 57.70, "speed": 12.0, "course": 220.0},
        {"mmsi": 111222333, "vessel_name": "GHOST VESSEL", "flag": "Liberia",
         "vessel_type": "Cargo", "timestamp": times[1], "latitude": -20.41,
         "longitude": 57.71, "speed": 12.0, "course": 220.0},
        {"mmsi": 111222333, "vessel_name": "GHOST VESSEL", "flag": "Liberia",
         "vessel_type": "Cargo", "timestamp": times[2], "latitude": -20.55,
         "longitude": 57.85, "speed": 12.0, "course": 220.0},
        {"mmsi": 111222333, "vessel_name": "GHOST VESSEL", "flag": "Liberia",
         "vessel_type": "Cargo", "timestamp": times[3], "latitude": -20.56,
         "longitude": 57.86, "speed": 12.0, "course": 220.0},
    ])

    gaps = dark_vessels.detect_transponder_gaps(
        df,
        min_gap_minutes=30.0,
        min_speed_knots=3.0,
        spill_lat=-20.4442,
        spill_lon=57.7433,
    )
    assert len(gaps) == 1
    g = gaps[0]
    assert g.mmsi == 111222333
    assert g.gap_minutes == pytest.approx(60.0, abs=0.1)
    assert g.distance_km > 10.0
    assert g.implied_speed_kn > 5.0
    assert g.crossed_hazard_zone is True  # transit vector passed near Pointe d'Esny


def test_analyze_scene_dark_vessels_curated_scene():
    """Verify analyze_scene_dark_vessels against packaged scene."""
    sc = build_scenario()
    res = dark_vessels.analyze_scene_dark_vessels("wakashio_reef", sc["ais"])
    assert res["scene_key"] == "wakashio_reef"
    assert res["total_radar_ships"] >= 1
    assert "dark_vessels" in res
    assert isinstance(res["targets"], list)


def test_api_dark_vessels_endpoints(client):
    """Verify FastAPI /dark-vessels endpoints."""
    # 1. Valid scene
    resp = client.get("/dark-vessels/scene/wakashio_reef")
    assert resp.status_code == 200
    data = resp.json()
    assert data["scene_key"] == "wakashio_reef"
    assert "total_radar_ships" in data
    assert "targets" in data

    # 2. Invalid scene
    resp_bad = client.get("/dark-vessels/scene/nonexistent_scene_xyz")
    assert resp_bad.status_code == 404

    # 3. Transponder gaps
    resp_gaps = client.get("/dark-vessels/gaps?min_gap_minutes=30.0&full_month=false")
    assert resp_gaps.status_code == 200
    gaps_data = resp_gaps.json()
    assert "total_gaps" in gaps_data
    assert "gaps" in gaps_data

