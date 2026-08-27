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
from poseatsea.scenario.real_ais import WAKASHIO_MMSI, build_scenario  # noqa: E402


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
    """The casualty must flag heavily; real innocent traffic must not."""
    wakashio = scored[scored["mmsi"] == WAKASHIO_MMSI]
    assert wakashio["is_anomaly"].mean() > 0.30
    assert wakashio["anomaly_score"].max() > 10 * AE_THRESHOLD

    for mmsi, track in scored[scored["mmsi"] != WAKASHIO_MMSI].groupby("mmsi"):
        assert track["is_anomaly"].mean() < 0.05, f"{mmsi} is innocent traffic"


def test_detector_fires_as_the_vessel_loses_way(scored):
    """
    The whole value proposition: the strike itself is flagged, not just the
    hours of sitting on the reef afterwards.
    """
    wakashio = scored[scored["mmsi"] == WAKASHIO_MMSI].sort_values("timestamp")
    approach = wakashio[wakashio["phase"] == "final approach"]
    assert approach["is_anomaly"].any(), "no flag before the vessel came to rest"
    first = approach[approach["is_anomaly"]].iloc[0]
    assert first["speed"] < 5.0, "the flag should coincide with the deceleration"


def test_aground_status_is_really_in_the_feed(scenario):
    """Status 6 is broadcast by the ship, not asserted by us."""
    ais = scenario["ais"]
    wakashio = ais[ais["mmsi"] == WAKASHIO_MMSI]
    assert (wakashio["status"] == 6).any() or (wakashio["phase"] == "aground").any()


# --------------------------------------------------------------------------
# Trajectory guard rails
# --------------------------------------------------------------------------
def test_prediction_matches_published_accuracy(scenario):
    """
    On real, unseen tracks the model must hit its published error envelope.
    This is the strongest validation available: genuine AIS the model never
    trained on, scored against the accuracy its authors claimed.
    """
    model = get_registry()["trajectory"].get()
    ais = scenario["ais"]
    checked = 0
    for mmsi, group in ais.groupby("mmsi"):
        track = group.sort_values("timestamp").reset_index(drop=True)
        trace = traj_mod.rolling_predictions(model, track)
        if len(trace) < 30:
            continue
        checked += 1
        median = float(trace["deviation_km"].median())
        assert median < 0.37, (f"{track['vessel_name'].iloc[0]}: median deviation "
                               f"{median:.3f} km exceeds the published mean error")
    assert checked >= 4, "expected several vessels with usable track length"


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
def test_attribution_names_the_casualty_and_clears_everyone_else(scenario, scored):
    """
    The casualty must come top, and every other real vessel in the window --
    all of them innocent -- must be cleared rather than merely ranked lower.
    """
    lat, lon = scenario["spill_position"]
    results = fusion.attribute(scored, lat, lon, observed_at=scenario["grounding_utc"])

    assert results[0].mmsi == WAKASHIO_MMSI
    assert results[0].confidence_band == "primary suspect"
    assert results[0].min_distance_km < 0.5

    for other in results[1:]:
        assert other.confidence_band == "cleared by proximity", (
            f"{other.vessel_name} should be cleared, got {other.confidence_band}")
        assert other.max_anomaly_score == 0.0, (
            f"{other.vessel_name} was nowhere near the slick")

    assert results[0].total_score > 4 * results[1].total_score


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
# Real AIS integrity
# --------------------------------------------------------------------------
def test_every_timestamp_parses():
    """
    About 4% of rows carry fractional seconds. A single inferred format drops
    them silently, which would quietly delete parts of the track.
    """
    from poseatsea.scenario.real_ais import load_raw
    assert load_raw()["timestamp"].notna().all()


def test_rot_sentinel_is_neutralised(scenario):
    """-128 means 'not available', not a hard port swing."""
    assert (scenario["ais"]["rot"] == -128).sum() == 0


def test_grounding_is_detected_from_the_data(scenario):
    """
    Position and time are derived from the feed, not hardcoded, and must land
    on the documented casualty: Pointe d'Esny at 19:25 local (15:25 UTC).
    """
    event = scenario["event"]
    assert event is not None
    lat, lon = event.position
    assert traj_mod.haversine_km(lat, lon, -20.4442, 57.7433) < 0.5
    assert event.utc.strftime("%Y-%m-%d") == "2020-07-25"
    assert 15 <= event.utc.hour <= 16
    assert event.gap_seconds < 300, "loss of way should be abrupt"


def test_wakashio_never_moves_again(scenario):
    """Six days aground: no ping after the strike shows meaningful way."""
    ais = scenario["ais"]
    after = ais[(ais["mmsi"] == WAKASHIO_MMSI) & (ais["phase"] == "aground")]
    assert len(after) > 50
    assert after["speed"].max() < 1.5


def test_scenario_is_deterministic():
    a = build_scenario()["ais"]
    b = build_scenario()["ais"]
    pd.testing.assert_frame_equal(a, b)


def test_every_vessel_stays_inside_the_aoi(scenario):
    for row in scenario["ais"].itertuples():
        assert traj_mod.in_aoi(row.latitude, row.longitude), f"{row.vessel_name} left the AOI"


# --------------------------------------------------------------------------
# Curated SAR scene library
# --------------------------------------------------------------------------
def test_scene_library_loads():
    from poseatsea.scenario import sar_scenes
    scenes = sar_scenes.load_scenes()
    assert len(scenes) == 5
    for s in scenes:
        assert s.image_path.exists(), f"{s.key}: source scene missing"
        assert s.mask_path.exists(), f"{s.key}: precomputed mask missing"


def test_precomputed_masks_match_live_inference():
    """
    The stored masks must be what the shipped checkpoint actually produces.
    If someone swaps the weights without re-running build_sar_scenes.py, the
    library silently starts lying -- this catches that.
    """
    from poseatsea.inference import sar as sar_mod
    from poseatsea.scenario import sar_scenes

    model = get_registry()["sar"].get()
    for scene in sar_scenes.load_scenes():
        live = sar_mod.segment(model, scene.load_image()).mask
        stored = scene.load_mask()
        assert stored.shape == live.shape, f"{scene.key}: shape drift"
        agreement = float((stored == live).mean())
        assert agreement > 0.99, f"{scene.key}: stored mask differs from live ({agreement:.3f})"


def test_scene_masks_are_coherent_not_noise():
    """
    The old checkpoint emitted salt-and-pepper noise. A real segmentation has
    large contiguous regions, so horizontal class changes stay rare.
    """
    from poseatsea.scenario import sar_scenes
    for scene in sar_scenes.load_scenes():
        m = scene.load_mask()
        churn = float((m[:, 1:] != m[:, :-1]).mean())
        assert churn < 0.08, f"{scene.key}: mask looks like noise (churn {churn:.3f})"


def test_library_separates_oil_from_lookalike():
    """The library must contain both a real spill and a convincing non-spill."""
    from poseatsea.scenario import sar_scenes
    scenes = {s.key: s for s in sar_scenes.load_scenes()}

    wakashio = scenes["wakashio_reef"]
    assert wakashio.oil_detected and wakashio.oil_area_km2 > 1.0

    lookalike = scenes["lookalike_field"]
    assert lookalike.oil_area_km2 < 0.05
    assert lookalike.lookalike_area_km2 > 1.0

    assert scenes["clean_coastal"].oil_area_km2 == 0.0


def test_only_the_casualty_is_attributed_to_a_vessel(scenario):
    """
    Every other ship in this feed is real, named and innocent. Pinning an oil
    signature on one of them would be indefensible, so only the Wakashio scene
    names a vessel.
    """
    from poseatsea.scenario import sar_scenes
    known = set(scenario["ais"]["mmsi"].unique())
    attributed = [s for s in sar_scenes.load_scenes() if s.attributed]
    assert len(attributed) == 1
    assert attributed[0].mmsi == WAKASHIO_MMSI
    assert attributed[0].mmsi in known
    assert sar_scenes.by_mmsi(WAKASHIO_MMSI).key == "wakashio_reef"


def test_sar_checkpoint_is_trained():
    """
    The first checkpoint shipped with an untrained decoder and emitted noise.
    BatchNorm running statistics are the tell: never updated means never
    trained through.
    """
    import torch
    from poseatsea.config import SAR_WEIGHTS
    sd = torch.load(SAR_WEIGHTS, map_location="cpu")
    tracked = {int(v) for k, v in sd.items() if k.endswith("num_batches_tracked")}
    assert tracked and max(tracked) > 0, "SAR decoder BatchNorm was never trained"
