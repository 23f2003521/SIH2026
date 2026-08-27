"""
Core pipeline tests.

These assert the properties that make the system trustworthy: that the weights
match the architectures we declare, that the scaler is actually applied, that
the detector separates the behaviour it claims to separate, and that the guard
rails (AOI bounds, drift caveat) genuinely fire.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from poseatsea import fusion  # noqa: E402
from poseatsea.config import (AE_THRESHOLD, FEATURE_ORDER, LAT_MIN, LON_MIN,  # noqa: E402
                              SAR_INPUT_SIZE, SEQ_LEN)
from poseatsea.inference import ais as ais_mod  # noqa: E402
from poseatsea.inference import sar as sar_mod  # noqa: E402
from poseatsea.inference import trajectory as traj_mod  # noqa: E402
from poseatsea.registry import get_registry  # noqa: E402
from poseatsea.scenario import build_scenario  # noqa: E402


@pytest.fixture(scope="module")
def scenario():
    return build_scenario()


@pytest.fixture(scope="module")
def scored(scenario):
    model, scaler = get_registry()["ais_anomaly"].get()
    return ais_mod.score_frame(model, scaler, scenario["ais"])


# --------------------------------------------------------------------------
# Weights match declared architectures
# --------------------------------------------------------------------------
def test_all_three_models_load_strict():
    """A silent architecture drift must be a load error, not a wrong answer."""
    reg = get_registry()
    reg.warmup()
    for key in ("sar", "trajectory", "ais_anomaly"):
        assert reg[key].is_loaded
        assert reg[key].error is None


def test_sar_emits_five_classes():
    model = get_registry()["sar"].get()
    rng = np.random.default_rng(0)
    img = (rng.random((320, 480, 3)) * 255).astype(np.uint8)
    result = sar_mod.segment(model, img)
    assert result.mask.shape == (SAR_INPUT_SIZE, SAR_INPUT_SIZE)
    assert result.mask.min() >= 0 and result.mask.max() <= 4
    assert sum(result.class_pixels.values()) == SAR_INPUT_SIZE * SAR_INPUT_SIZE


def test_mask_upsampling_invents_no_classes():
    """INTER_NEAREST only -- interpolation would create fractional class ids."""
    mask = np.array([[0, 1], [3, 4]], dtype=np.uint8)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    out = sar_mod.overlay_on_image(img, mask, alpha=1.0)
    colors = {tuple(c) for c in out.reshape(-1, 3)}
    allowed = {(0, 0, 0)} | {tuple(sar_mod.CLASS_COLORS_RGB[c]) for c in (1, 3, 4)}
    assert colors <= allowed


# --------------------------------------------------------------------------
# The scaler is mandatory, not decorative
# --------------------------------------------------------------------------
def test_unscaled_input_gives_a_different_answer():
    """
    Guards the single most damaging silent bug available here: forgetting the
    scaler. The model was trained on standardised features and returns
    confident nonsense on raw ones.
    """
    model, scaler = get_registry()["ais_anomaly"].get()
    row = {"speed": 11.2, "course": 225.0, "rot": 0.0, "msg_type": 1, "status": 0,
           "accuracy": 1, "course_diff": 0.4, "rot_diff": 0.0, "speed_diff": 0.1,
           "lat_diff": -0.0008, "long_diff": -0.0011}

    scaled = ais_mod.score_row(model, scaler, row).score

    import torch
    raw = np.array([[row[f] for f in FEATURE_ORDER]], dtype=np.float64)
    with torch.no_grad():
        recon = model(torch.tensor(raw, dtype=torch.float32)).numpy()
    unscaled = float(np.mean((raw - recon) ** 2))

    assert scaled < AE_THRESHOLD, "normal transit should not flag"
    assert unscaled > scaled * 100, "skipping the scaler must be obviously wrong"


# --------------------------------------------------------------------------
# The detector separates what it claims to separate
# --------------------------------------------------------------------------
def test_grounding_flags_and_normal_transit_does_not(scored):
    wakashio = scored[scored["mmsi"] == 371284000]
    aground = wakashio[wakashio["phase"] == "aground"]
    assert aground["is_anomaly"].all(), "every aground ping should flag"
    assert aground["anomaly_score"].max() > 3 * AE_THRESHOLD

    for mmsi in (419002731, 256891004, 563114900):      # clean transits
        track = scored[scored["mmsi"] == mmsi]
        assert not track["is_anomaly"].any(), f"{mmsi} transits cleanly and must not flag"


def test_anchored_vessel_is_not_treated_as_anomalous(scored):
    """
    A hull at anchor has no steerage way, so its course is held rather than
    derived from receiver dither. Without that, GPS noise reads as a vessel
    spinning on its anchor and swamps the real casualty.
    """
    anchored = scored[scored["mmsi"] == 352001899]
    assert anchored["is_anomaly"].mean() < 0.10


def test_course_diff_wraps_shortest_way():
    df = pd.DataFrame({
        "mmsi": [1, 1], "timestamp": pd.to_datetime(["2020-07-25T00:00", "2020-07-25T00:01"]),
        "latitude": [-20.4, -20.4], "longitude": [57.8, 57.8],
        "speed": [10.0, 10.0], "course": [359.0, 1.0], "rot": [0.0, 0.0],
    })
    out = ais_mod.add_derived_features(df)
    assert out["course_diff"].iloc[1] == pytest.approx(2.0)


# --------------------------------------------------------------------------
# Trajectory guard rails
# --------------------------------------------------------------------------
def test_prediction_matches_published_accuracy(scenario):
    """Reconstructed tracks must sit inside the model's own error envelope."""
    model = get_registry()["trajectory"].get()
    track = scenario["ais"]
    clean = track[track["mmsi"] == 563114900].reset_index(drop=True)
    trace = traj_mod.rolling_predictions(model, clean)
    assert len(trace) > 50
    assert trace["deviation_km"].median() < 0.37, "should beat the published mean error"


def test_out_of_aoi_history_is_refused():
    """Silently predicting outside the AOI would be the worst possible failure."""
    model = get_registry()["trajectory"].get()
    history = pd.DataFrame({
        "latitude": [19.0] * SEQ_LEN,          # Arabian Sea, far outside Mauritius
        "longitude": [70.0] * SEQ_LEN,
        "speed": [11.0] * SEQ_LEN,
        "course": [225.0] * SEQ_LEN,
        "rot": [0.0] * SEQ_LEN,
    })
    assert not traj_mod.assess_inputs(history).usable
    with pytest.raises(ValueError, match="outside the Mauritius AOI"):
        traj_mod.predict_next_position(model, history, strict=True)


def test_wrong_sequence_length_is_refused():
    history = pd.DataFrame({
        "latitude": [LAT_MIN + 0.1] * 5, "longitude": [LON_MIN + 0.1] * 5,
        "speed": [11.0] * 5, "course": [225.0] * 5, "rot": [0.0] * 5,
    })
    assessment = traj_mod.assess_inputs(history)
    assert not assessment.usable
    assert "exactly 8" in " ".join(assessment.blockers)


# --------------------------------------------------------------------------
# Attribution
# --------------------------------------------------------------------------
def test_attribution_names_the_casualty_not_the_trawler(scenario, scored):
    """
    The trawler carries the highest raw anomaly score in the whole window. If
    attribution ever ranked it first, the fusion logic would be broken.
    """
    lat, lon = scenario["spill_position"]
    results = fusion.attribute(scored, lat, lon, observed_at=scenario["grounding_utc"])

    assert results[0].mmsi == 371284000
    assert results[0].confidence_band == "primary suspect"

    trawler = next(r for r in results if r.mmsi == 645079210)
    assert trawler.max_anomaly_score == 0.0, "trawler is nowhere near the slick"
    assert results[0].total_score > trawler.total_score * 5

    # ...and it really does out-score the casualty on behaviour alone.
    peak = scored.groupby("mmsi")["anomaly_score"].max()
    assert peak[645079210] > peak[371284000]


def test_attribution_margin_is_decisive(scenario, scored):
    lat, lon = scenario["spill_position"]
    results = fusion.attribute(scored, lat, lon, observed_at=scenario["grounding_utc"])
    assert fusion.summarise(results)["margin"] > 0.2


def test_drift_caveat_is_always_present():
    assert "no ocean-current" in fusion.drift_caveat().lower()


def test_distant_spill_clears_everyone(scored):
    """A slick 200 km away must not be pinned on anybody."""
    results = fusion.attribute(scored, -20.30, 55.50,
                               radius_km=15.0, window_hours=6.0)
    assert all(r.total_score < 0.2 for r in results)
    assert "no vessel" in fusion.summarise(results)["verdict"].lower()


# --------------------------------------------------------------------------
# Scenario integrity
# --------------------------------------------------------------------------
def test_track_terminates_on_the_reef(scenario):
    wakashio = scenario["ais"]
    wakashio = wakashio[wakashio["mmsi"] == 371284000]
    lat, lon = scenario["spill_position"]
    final = wakashio.iloc[-1]
    assert traj_mod.haversine_km(final["latitude"], final["longitude"], lat, lon) < 0.15


def test_scenario_is_deterministic():
    a = build_scenario()["ais"]
    b = build_scenario()["ais"]
    pd.testing.assert_frame_equal(a, b)


def test_every_vessel_stays_inside_the_aoi(scenario):
    for row in scenario["ais"].itertuples():
        assert traj_mod.in_aoi(row.latitude, row.longitude), f"{row.vessel_name} left the AOI"
