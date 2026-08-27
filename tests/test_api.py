"""API contract tests -- exercised through FastAPI's TestClient, no server needed."""
from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402
from poseatsea.config import SAR_INPUT_SIZE, SEQ_LEN  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert len(body["models"]) == 3


def test_models_declares_limitations(client):
    body = client.get("/models").json()
    joined = " ".join(body["limitations"]).lower()
    assert "no drift or hindcast" in joined
    assert "cannot clear one" in joined


def test_score_ping_flags_a_grounding(client):
    grounding = {"speed": 0.3, "course": 246.0, "rot": 127.0, "msg_type": 1,
                 "status": 6, "accuracy": 1, "course_diff": 41.0, "rot_diff": 127.0,
                 "speed_diff": -10.6, "lat_diff": -0.0001, "long_diff": -0.0001}
    body = client.post("/ais/score", json=grounding).json()
    assert body["is_anomaly"] is True
    assert body["severity"] in {"high", "critical"}
    assert body["top_contributors"]


def test_score_ping_clears_normal_transit(client):
    transit = {"speed": 11.2, "course": 225.0, "rot": 0.0, "msg_type": 1,
               "status": 0, "accuracy": 1, "course_diff": 0.4, "rot_diff": 0.0,
               "speed_diff": 0.1, "lat_diff": -0.0008, "long_diff": -0.0011}
    assert client.post("/ais/score", json=transit).json()["is_anomaly"] is False


def test_trajectory_rejects_wrong_sequence_length(client):
    point = {"latitude": -20.4, "longitude": 57.9, "speed": 11.0,
             "course": 240.0, "rot": 0.0}
    r = client.post("/trajectory/predict", json={"history": [point] * 5})
    assert r.status_code == 400


def test_trajectory_rejects_out_of_aoi(client):
    point = {"latitude": 19.0, "longitude": 70.0, "speed": 11.0,
             "course": 225.0, "rot": 0.0}
    r = client.post("/trajectory/predict", json={"history": [point] * SEQ_LEN})
    assert r.status_code == 422


def test_trajectory_predicts_inside_aoi(client):
    history = []
    lat, lon = -20.34, 57.95
    for _ in range(SEQ_LEN):
        history.append({"latitude": lat, "longitude": lon, "speed": 11.0,
                        "course": 236.0, "rot": 0.0})
        lat -= 0.0017
        lon -= 0.0025
    body = client.post("/trajectory/predict", json={"history": history}).json()
    assert -20.6 < body["predicted"]["lat"] < -20.0
    assert 57.7 < body["predicted"]["lon"] < 58.4


def test_attribution_uses_builtin_scenario(client):
    body = client.post("/attribution/rank",
                       json={"spill_lat": -20.4442, "spill_lon": 57.7433}).json()
    assert "WAKASHIO" in body["candidates"][0]["vessel_name"].upper()
    assert "ocean-current" in body["caveat"].lower()
    assert sum(body["weights"].values()) == pytest.approx(1.0)


def test_sar_segment_accepts_an_upload(client):
    rng = np.random.default_rng(0)
    img = Image.fromarray((rng.random((256, 256, 3)) * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    r = client.post("/sar/segment", files={"file": ("scene.png", buf.getvalue(), "image/png")})
    body = r.json()
    assert r.status_code == 200
    assert "oil_area_km2" in body
    assert sum(body["class_pixels"].values()) == 256 * 256   # resampled to source


def test_sar_rejects_a_non_image(client):
    r = client.post("/sar/segment", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400


def test_scenario_declares_its_provenance(client):
    body = client.get("/scenario?limit=5").json()
    assert "Real AIS" in body["provenance"]
    assert "synthetic" in body["provenance"]
    assert body["total_pings"] > 700
