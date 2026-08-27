"""
Streamlit's bridge to the model registry.

Streamlit re-runs the whole script on every widget interaction. Without a cache
that would rebuild a 110 MB SegFormer on every click. `st.cache_resource` keeps
one registry per server process; the registry itself then handles lazy loading
and per-model locking, so this layer stays thin.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Allow `streamlit run ui/app.py` from the project root without installation.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poseatsea import fusion  # noqa: E402
from poseatsea.config import AE_THRESHOLD  # noqa: E402
from poseatsea.inference import ais as ais_mod  # noqa: E402
from poseatsea.inference import sar as sar_mod  # noqa: E402
from poseatsea.inference import trajectory as traj_mod  # noqa: E402
from poseatsea.registry import get_registry  # noqa: E402
from poseatsea.scenario import build_scenario  # noqa: E402


@st.cache_resource(show_spinner=False)
def registry():
    return get_registry()


@st.cache_resource(show_spinner="Loading SAR segmenter (110 MB, first use only)...")
def sar_model():
    return registry()["sar"].get()


@st.cache_resource(show_spinner="Loading AIS anomaly autoencoder...")
def ais_model():
    return registry()["ais_anomaly"].get()


@st.cache_resource(show_spinner="Loading trajectory LSTM...")
def trajectory_model():
    return registry()["trajectory"].get()


# --------------------------------------------------------------------------
# Scenario + derived analytics, cached on their inputs
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def scenario(interval_s: float = 60.0, seed: int = 20200725) -> Dict[str, Any]:
    return build_scenario(interval_s=interval_s, seed=seed)


@st.cache_data(show_spinner="Scoring AIS pings...")
def scored_ais(interval_s: float = 60.0, seed: int = 20200725) -> pd.DataFrame:
    model, scaler = ais_model()
    return ais_mod.score_frame(model, scaler, scenario(interval_s, seed)["ais"])


@st.cache_data(show_spinner="Running trajectory predictions...")
def deviation_traces(interval_s: float = 60.0, seed: int = 20200725) -> Dict[int, pd.DataFrame]:
    """One rolling predicted-vs-actual trace per vessel."""
    model = trajectory_model()
    out: Dict[int, pd.DataFrame] = {}
    for mmsi, group in scenario(interval_s, seed)["ais"].groupby("mmsi"):
        trace = traj_mod.rolling_predictions(model, group.reset_index(drop=True))
        if len(trace):
            out[int(mmsi)] = trace
    return out


@st.cache_data(show_spinner=False)
def max_deviations(interval_s: float = 60.0, seed: int = 20200725) -> Dict[int, float]:
    return {m: float(t["deviation_km"].max()) for m, t in deviation_traces(interval_s, seed).items()}


@st.cache_data(show_spinner="Correlating spill against AIS traffic...")
def attribution(spill_lat: float, spill_lon: float, radius_km: float,
                window_hours: float, interval_s: float = 60.0,
                seed: int = 20200725) -> List[Dict[str, Any]]:
    sc = scenario(interval_s, seed)
    results = fusion.attribute(
        scored_ais(interval_s, seed),
        spill_lat, spill_lon,
        observed_at=sc["grounding_utc"],
        radius_km=radius_km,
        window_hours=window_hours,
        threshold=AE_THRESHOLD,
        deviations=max_deviations(interval_s, seed),
    )
    return [r.as_dict() for r in results]


# --------------------------------------------------------------------------
# SAR -- not cached on the image itself (arrays are large and rarely reused)
# --------------------------------------------------------------------------
def segment_image(image_rgb, pixel_resolution_m: float = 10.0):
    """Run segmentation under the registry's per-model inference lock."""
    handle = registry()["sar"]
    handle.get()
    return handle.run(lambda m: sar_mod.segment(m, image_rgb, pixel_resolution_m))


def score_single_ping(row: Dict[str, float]):
    model, scaler = ais_model()
    return ais_mod.score_row(model, scaler, row)


def predict_next(history: pd.DataFrame, actual_next: Optional[Dict[str, float]] = None,
                 strict: bool = False):
    return traj_mod.predict_next_position(trajectory_model(), history,
                                          actual_next=actual_next, strict=strict)
