"""
Tests for the Coast Guard & maritime authority forensic PDF dossier generator.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from api.main import app
from poseatsea import report
from poseatsea.scenario.wakashio import INCIDENT


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_dossier_pdf_generation_default():
    """Verify default PDF generation produces valid PDF bytes."""
    pdf_bytes = report.generate_dossier_pdf()
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 5000


def test_dossier_pdf_with_custom_attribution():
    """Verify PDF generation handles custom attribution dicts cleanly."""
    custom_results = [
        {
            "mmsi": 372711000,
            "vessel_name": "MV WAKASHIO",
            "flag": "Panama",
            "vessel_type": "Bulk carrier",
            "min_distance_km": 0.05,
            "minutes_in_radius": 180.0,
            "flagged_pings": 209,
            "max_anomaly_score": 25.08,
            "max_deviation_km": 0.19,
            "score": 0.936,
            "band": "primary suspect",
            "components": {
                "proximity": 0.95,
                "anomaly": 0.98,
                "dwell": 1.0,
                "deviation": 0.06,
            },
            "evidence": ["Closest approach 50 m", "209 anomalous pings"],
        },
        {
            "mmsi": 564796000,
            "vessel_name": "KOTA SURIA",
            "flag": "Singapore",
            "vessel_type": "Cargo ship",
            "min_distance_km": 27.7,
            "minutes_in_radius": 0.0,
            "flagged_pings": 1,
            "max_anomaly_score": 1.17,
            "max_deviation_km": None,
            "score": 0.08,
            "band": "cleared by proximity",
            "components": {
                "proximity": 0.0,
                "anomaly": 0.2,
                "dwell": 0.0,
                "deviation": 0.0,
            },
            "evidence": ["Never came closer than 27.7 km"],
        },
    ]

    pdf_bytes = report.generate_dossier_pdf(
        incident=INCIDENT,
        attribution_results=custom_results,
        case_ref="TEST-CASE-2026-001",
    )
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 5000


def test_dossier_contains_expected_evidentiary_content():
    """Verify PDF content carries key evidentiary identifiers."""
    import reportlab.rl_config

    orig_compression = reportlab.rl_config.pageCompression
    try:
        reportlab.rl_config.pageCompression = 0
        pdf_bytes = report.generate_dossier_pdf(case_ref="EVID-POINTE-DESNY-999")
        assert b"INCIDENT ASSESSMENT" in pdf_bytes
        assert b"WAKASHIO" in pdf_bytes
        assert b"372711000" in pdf_bytes
        assert b"EVID-POINTE-DESNY-999" in pdf_bytes
        assert b"PRIMARY SUSPECT" in pdf_bytes
        assert b"SHA-256" in pdf_bytes
        assert b"Pointe d'Esny" in pdf_bytes
    finally:
        reportlab.rl_config.pageCompression = orig_compression



def test_file_sha256_helper():
    """Verify cryptographic hash computation."""
    ais_csv = ROOT / "assets" / "ais" / "mauritius_aoi_202007.csv"
    h = report._file_sha256(ais_csv)
    assert isinstance(h, str)
    assert len(h) == 64  # standard sha256 hex string length

    nonexistent = ROOT / "assets" / "nonexistent_file.xyz"
    assert report._file_sha256(nonexistent) == "NOT_AVAILABLE_ON_DISK"


def test_api_dossier_endpoint(client):
    """Verify FastAPI /reports/dossier endpoint returns the PDF stream."""
    resp = client.get("/reports/dossier?scene_key=wakashio_reef")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert "attachment; filename=" in resp.headers["content-disposition"]
    assert resp.content.startswith(b"%PDF-")
    assert len(resp.content) > 5000
