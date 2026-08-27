"""Trajectory prediction page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from poseatsea.config import SEQ_LEN, TRAJ_P90_ERROR_KM
from poseatsea.scenario.real_ais import WAKASHIO_MMSI
from poseatsea.inference import trajectory as traj_mod

from streamlit_folium import st_folium

from .. import charts, engine, maps, theme


def _default_window(track: pd.DataFrame, trace, max_start: int) -> int:
    """
    Open on the most informative window for this vessel.

    For a vessel that lost way and never regained it, that is the approach into
    the stop: the eight pings ending on the last ping with way on, so the model
    predicts from a moving ship and the truth ping is the moment she stopped.
    Anchoring on peak anomaly instead lands mid-wreck, on eight identical
    zero-knot pings, where a next-position prediction says nothing at all.
    """
    moving = track.index[track["speed"] > 1.0]
    if len(moving):
        last = int(moving[-1])
        after = track["speed"].iloc[last + 1:]
        if len(after) and after.max() <= 1.5:
            return int(max(0, min(max_start, last - SEQ_LEN + 1)))

    if trace is not None and len(trace):
        clean = traj_mod.clean_trace(trace)
        if len(clean):
            worst = int(clean.loc[clean["deviation_km"].idxmax(), "index"])
            return int(max(0, min(max_start, worst - SEQ_LEN)))
    return 0


def render() -> None:
    st.markdown("## Route deviation detection")
    card = traj_mod.model_card()
    st.markdown(
        f"<span style='color:{theme.MUTED}'>A 2-layer LSTM maps a vessel's last "
        f"{SEQ_LEN} AIS pings to its next position. {card['reading']}</span>",
        unsafe_allow_html=True,
    )

    scored = engine.scored_ais()
    traces = engine.deviation_traces()

    names = scored.groupby("mmsi")["vessel_name"].first().to_dict()
    options = {v: k for k, v in names.items()}
    # Key the default off the MMSI, never a display string -- the vessel's name
    # comes from the feed and must not be hardcoded anywhere.
    casualty = names.get(WAKASHIO_MMSI)
    default_ix = list(options).index(casualty) if casualty in options else 0
    chosen = st.selectbox("Vessel", list(options), index=default_ix)
    mmsi = options[chosen]

    track = scored[scored["mmsi"] == mmsi].sort_values("timestamp").reset_index(drop=True)
    trace = traces.get(mmsi)

    # ------------------------------------------------------------------ window
    max_start = max(0, len(track) - SEQ_LEN - 1)
    start = st.slider(
        f"History window (the {SEQ_LEN} pings fed to the model)",
        0, max_start, _default_window(track, trace, max_start),
    )
    history = track.iloc[start:start + SEQ_LEN].reset_index(drop=True)
    truth_row = track.iloc[start + SEQ_LEN]
    actual = {"latitude": float(truth_row["latitude"]),
              "longitude": float(truth_row["longitude"])}

    assessment = traj_mod.assess_inputs(history)
    if not assessment.usable:
        for b in assessment.blockers:
            theme.banner(f"<b>Cannot predict.</b> {b}", "bad")
        return

    pred = engine.predict_next(history, actual_next=actual, strict=False)

    # ------------------------------------------------------------------ headline
    dev = pred.deviation_km
    tone = theme.GOOD if dev is not None and dev <= TRAJ_P90_ERROR_KM else theme.WARN

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(theme.metric_card(
            "Predicted next", f"{pred.predicted_lat:.4f}, {pred.predicted_lon:.4f}",
            f"{pred.step_km:.3f} km at {pred.predicted_bearing:.0f}°"),
            unsafe_allow_html=True)
    with c2:
        st.markdown(theme.metric_card(
            "Actual next", f"{actual['latitude']:.4f}, {actual['longitude']:.4f}",
            f"{truth_row['timestamp']:%H:%M:%S} UTC"), unsafe_allow_html=True)
    with c3:
        st.markdown(theme.metric_card(
            "Deviation", f"{dev:.3f} km" if dev is not None else "--",
            f"median error {card['median_error_km']} km", tone), unsafe_allow_html=True)
    with c4:
        conf_color = {"nominal": theme.GOOD, "degraded": theme.WARN,
                      "unreliable": theme.BAD}[assessment.confidence]
        st.markdown(theme.metric_card(
            "Input confidence", assessment.confidence.title(),
            "against the training envelope", conf_color), unsafe_allow_html=True)

    verdict = pred.deviation_verdict()
    if verdict:
        good = dev is not None and dev <= TRAJ_P90_ERROR_KM
        theme.banner(f"<b>{verdict}</b>", "good" if good else "warn")

    for w in assessment.warnings:
        theme.banner(w, "warn")

    # ------------------------------------------------------------------ plots
    left, right = st.columns([1.6, 1])
    with left:
        st.markdown("##### Predicted against actual")
        st_folium(
            maps.trajectory_map(track, history, pred.predicted_lat, pred.predicted_lon,
                                actual=actual, deviation_km=dev),
            use_container_width=True, height=520, returned_objects=[],
            key=f"traj_map_{mmsi}_{start}",
        )
        st.markdown(maps.legend([
            {"color": theme.GOOD, "label": "History the model saw (8 pings)"},
            {"color": theme.ACCENT, "label": "LSTM predicted next position"},
            {"color": theme.WARN, "label": "Deviation"},
            {"color": theme.CRITICAL, "label": "Reef hazard"},
        ]), unsafe_allow_html=True)
        st.caption("Zoom in on the red ring to see predicted versus actual separate.")
    with right:
        st.markdown("##### The window the model saw")
        cols = ["timestamp", "latitude", "longitude", "speed", "course", "rot"]
        st.dataframe(history[cols], use_container_width=True, hide_index=True, height=330)

    # ------------------------------------------------------------------ full trace
    if trace is not None and len(trace):
        st.markdown("##### Deviation across the whole transit")
        mcol, tcol = st.columns([1.15, 1])
        with mcol:
            st_folium(
                maps.deviation_map(track, trace),
                use_container_width=True, height=400, returned_objects=[],
                key=f"dev_map_{mmsi}",
            )
            st.caption("Each ping sized and coloured by how far it fell from the "
                       "model's prediction — green predictable, red not.")
        with tcol:
            st.altair_chart(charts.deviation_timeline(trace, TRAJ_P90_ERROR_KM),
                            use_container_width=True)

        clean = traj_mod.clean_trace(trace)
        gaps = int(trace["coverage_gap"].sum()) if "coverage_gap" in trace else 0
        stat = clean if len(clean) else trace

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(theme.metric_card(
                "Median deviation", f"{stat['deviation_km'].median():.3f} km",
                f"model's own median {card['median_error_km']} km"), unsafe_allow_html=True)
        with c2:
            st.markdown(theme.metric_card(
                "90th percentile", f"{stat['deviation_km'].quantile(0.9):.3f} km",
                f"model's own p90 {card['p90_error_km']} km"), unsafe_allow_html=True)
        with c3:
            st.markdown(theme.metric_card(
                "Coverage gaps", str(gaps),
                f"of {len(trace)} windows — excluded",
                theme.WARN if gaps else theme.GOOD), unsafe_allow_html=True)

        st.caption(
            "The dashed line is the model's published 90th-percentile error on vessels it "
            "never saw in training. Sustained excursions above it mean the vessel is moving "
            "in a way the model did not anticipate — a predictability signal, not a finding "
            "of wrongdoing."
        )

        if gaps:
            theme.banner(
                f"<b>{gaps} of {len(trace)} windows sit behind a satellite AIS dropout</b> and "
                f"are excluded from the figures above. The model predicts <i>the next ping</i>; "
                f"when that ping arrives many minutes late the vessel has legitimately travelled "
                f"kilometres, so the apparent error measures reception, not behaviour.",
                "warn")

    with st.expander("Operating envelope — where this model can and cannot be trusted"):
        st.markdown(
            f"""
Normalisation is anchored to the AOI and speed distribution the model was trained on,
which bounds where it can be used at all:

- **Geographic.** Latitude {card['aoi']['lat'][0]:.4f} to {card['aoi']['lat'][1]:.4f},
  longitude {card['aoi']['lon'][0]:.4f} to {card['aoi']['lon'][1]:.4f} — the Mauritius AOI.
  Outside this box the normalisation saturates and output is meaningless. The console
  refuses to predict rather than returning a confident wrong answer.
- **Cadence.** Probing the checkpoint shows step size is reproduced well at roughly
  60-second ping intervals. At 5-minute spacing the predicted step falls to under half
  the true distance travelled.
- **Heading.** Bearing error is small along the NE–SW lane (courses near 045° and 225°),
  which dominates traffic in this region. On NW–SE headings the model frequently predicts
  a bearing close to the reverse of the true one — those courses are sparse in the
  training area. Windows on that axis are marked *degraded*.

These limits were measured against the supplied weights, not taken on trust. A model
that is quietly wrong is far more dangerous than one that declares its own boundaries.
"""
        )

    theme.provenance(
        "This model is a standalone capability. It is deliberately not wired into the "
        "anomaly detector — adding predicted-versus-actual distance as a twelfth input "
        "feature was tested during development and did not improve detection accuracy."
    )
