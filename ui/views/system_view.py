"""System page -- model provenance, runtime state, and declared limitations."""
from __future__ import annotations

import platform
import sys

import streamlit as st
import torch

from poseatsea import __version__
from poseatsea.config import (
    AIS_AE_WEIGHTS,
    AIS_SCALER,
    DEVICE,
    MODELS_DIR,
    SAR_WEIGHTS,
    TRAJECTORY_WEIGHTS,
)
from poseatsea.inference import ais as ais_mod
from poseatsea.inference import trajectory as traj_mod

from .. import engine, theme


def _size_mb(path) -> str:
    try:
        return f"{path.stat().st_size / (1024 * 1024):.1f} MB"
    except OSError:
        return "missing"


def render() -> None:
    st.markdown("## System")

    reg = engine.registry()
    status = reg.status()

    c1, c2, c3, c4 = st.columns(4)
    loaded = sum(1 for m in status["models"] if m["loaded"])
    with c1:
        st.markdown(theme.metric_card("Device", str(DEVICE).upper(),
                                      f"torch {torch.__version__}"), unsafe_allow_html=True)
    with c2:
        st.markdown(theme.metric_card("Models resident", f"{loaded} / 3",
                                      "loaded lazily on first use"), unsafe_allow_html=True)
    with c3:
        st.markdown(theme.metric_card("Weights on disk",
                                      _size_mb(SAR_WEIGHTS).split()[0] + " MB",
                                      "SAR segmenter dominates"), unsafe_allow_html=True)
    with c4:
        st.markdown(theme.metric_card("Console", f"v{__version__}",
                                      f"Python {sys.version_info.major}."
                                      f"{sys.version_info.minor}"), unsafe_allow_html=True)

    # ------------------------------------------------------------------ models
    st.markdown("##### Model registry")
    for m in status["models"]:
        color = theme.GOOD if m["loaded"] else theme.MUTED
        state = "RESIDENT" if m["loaded"] else "NOT LOADED"
        detail = ""
        if m["loaded"]:
            detail = (f"{m['parameters']:,} parameters &middot; "
                      f"loaded in {m['load_seconds']:.2f}s")
        elif m["error"]:
            color, state, detail = theme.BAD, "ERROR", m["error"]
        st.markdown(
            f"""
<div class="pos-card tight">
  <div style="display:flex;align-items:center;gap:.7rem">
    <b>{m['name']}</b>
    {theme.pill(state, color)}
    <span style="margin-left:auto;color:{theme.MUTED};font-size:.78rem">{detail}</span>
  </div>
  <div class="pos-sub">{m['description']}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    a, b = st.columns([1, 1])
    with a:
        if st.button("Warm up all models", use_container_width=True):
            with st.spinner("Loading every model into memory..."):
                timings = reg.warmup()
            st.success(" · ".join(f"{k}: {v:.2f}s" for k, v in timings.items()))
            st.rerun()
    with b:
        if st.button("Clear cached analytics", use_container_width=True):
            st.cache_data.clear()
            st.success("Scenario and scoring caches cleared. Models stay resident.")

    # ------------------------------------------------------------------ artefacts
    st.markdown("##### Weight files")
    st.dataframe(
        [
            {"Artefact": "best_sar_model.pth", "Size": _size_mb(SAR_WEIGHTS),
             "Architecture": "U-Net + MiT-B2 encoder, 5 classes",
             "Present": SAR_WEIGHTS.exists()},
            {"Artefact": "trajectory_lstm_baseline.pth", "Size": _size_mb(TRAJECTORY_WEIGHTS),
             "Architecture": "LSTM(6→128)×2 + Linear(128→2)",
             "Present": TRAJECTORY_WEIGHTS.exists()},
            {"Artefact": "ais_phase1_autoencoder.pth", "Size": _size_mb(AIS_AE_WEIGHTS),
             "Architecture": "Autoencoder 11→16→8→4→8→16→11",
             "Present": AIS_AE_WEIGHTS.exists()},
            {"Artefact": "ais_phase1_scaler.joblib", "Size": _size_mb(AIS_SCALER),
             "Architecture": "StandardScaler, 11 features",
             "Present": AIS_SCALER.exists()},
        ],
        use_container_width=True, hide_index=True,
    )
    st.caption(f"Resolved from `{MODELS_DIR}`. Every state dict is loaded with "
               f"`strict=True`, so a silent architecture drift becomes a startup error "
               f"rather than a wrong answer.")

    # ------------------------------------------------------------------ cards
    st.markdown("##### Reported performance")
    ais_card, traj_card = ais_mod.model_card(), traj_mod.model_card()

    left, right = st.columns(2)
    with left:
        st.markdown(
            f"""
<div class="pos-card">
  <div class="pos-label">AIS anomaly autoencoder</div>
  <div class="mono" style="line-height:1.8;color:{theme.MUTED}">
    precision &nbsp;{ais_card['precision']:.3f}<br>
    recall &nbsp;&nbsp;&nbsp;&nbsp;{ais_card['recall']:.3f}<br>
    F1 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.567<br>
    threshold {ais_card['threshold']:.6f}
  </div>
  <div class="pos-sub" style="margin-top:.55rem">{ais_card['reading']}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"""
<div class="pos-card">
  <div class="pos-label">Trajectory LSTM</div>
  <div class="mono" style="line-height:1.8;color:{theme.MUTED}">
    mean error &nbsp;&nbsp;{traj_card['mean_error_km']} km<br>
    median error {traj_card['median_error_km']} km<br>
    p90 error &nbsp;&nbsp;&nbsp;{traj_card['p90_error_km']} km<br>
    sequence &nbsp;&nbsp;&nbsp;&nbsp;{traj_card['sequence_length']} pings
  </div>
  <div class="pos-sub" style="margin-top:.55rem">{traj_card['reading']}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------ limits
    st.markdown("##### Declared limitations")
    st.markdown(
        """
These are stated here rather than discovered in a demo.

1. **No drift or hindcast modelling.** Attribution assumes the slick lies where it
   was observed. There is no ocean-current or wind model in this system.
2. **The trajectory model is regional.** It is normalised to the Mauritius AOI and
   returns nothing outside it. Use elsewhere requires retraining.
3. **The anomaly detector misses more than it catches.** Recall is 0.41. It nominates
   vessels for review; it cannot clear one.
4. **Area estimates are approximate.** The segmentation mask is a 512×512 resample of
   the source scene, and ground resolution is an operator-supplied assumption.
5. **The AIS is real.** A Mauritius AOI extract for July 2020 containing the
   Wakashio's own broadcasts. Only the documented background (owner, tonnage,
   voyage, what happened after she stopped) comes from the casualty record.
6. **The SAR scene library is precomputed.** Its five masks are genuine outputs of
   the shipped checkpoint, generated by `deploy/build_sar_scenes.py` and stored so
   the page renders without loading the 110 MB segmenter. Uploading a scene runs
   the model live, and a test asserts the stored masks still match live output.
7. **The segmenter only works at 512×512.** At any other input size it returns
   incoherent noise rather than a degraded result, so the size is fixed in config
   and not exposed as a setting.
"""
    )

    with st.expander("Runtime detail"):
        st.code(
            f"""platform   {platform.platform()}
python     {sys.version.split()[0]}
torch      {torch.__version__}
device     {DEVICE}
cuda       {torch.cuda.is_available()}
models_dir {MODELS_DIR}""",
            language="text",
        )
