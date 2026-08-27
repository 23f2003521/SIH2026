"""AIS anomaly inspector."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from poseatsea.config import AE_THRESHOLD, FEATURE_ORDER
from poseatsea.inference import ais as ais_mod
from poseatsea.scenario.real_ais import to_csv_bytes

from streamlit_folium import st_folium

from .. import charts, engine, maps, theme


def render() -> None:
    st.markdown("## AIS anomaly inspector")
    card = ais_mod.model_card()
    st.markdown(
        f"<span style='color:{theme.MUTED}'>An 11-feature autoencoder scores each AIS ping "
        f"by how badly it reconstructs. {card['reading']}</span>",
        unsafe_allow_html=True,
    )

    scored = engine.scored_ais()
    rollup = ais_mod.vessel_rollup(scored)

    tab_fleet, tab_vessel, tab_manual = st.tabs(
        ["Fleet view", "Vessel detail", "Score a ping"]
    )

    # ------------------------------------------------------------------ fleet
    with tab_fleet:
        st.markdown("##### Every vessel in the window, worst first")
        show = rollup.copy()
        show = show[["vessel_name", "mmsi", "pings", "flagged", "flagged_pct",
                     "max_score", "mean_score"]]
        show.columns = ["Vessel", "MMSI", "Pings", "Flagged", "Flagged %",
                        "Peak error", "Mean error"]
        st.dataframe(
            show.style.format({"Peak error": "{:.3f}", "Mean error": "{:.4f}",
                               "Flagged %": "{:.1f}"}),
            use_container_width=True, hide_index=True,
        )

        theme.banner(
            "<b>Read this table carefully.</b> The highest raw anomaly score in this "
            "window belongs to a <i>trawler working inshore</i>, not to the casualty. "
            "Tight repeated turns are kinematically anomalous and entirely lawful. "
            "A behavioural flag is a reason to look, never a finding on its own &mdash; "
            "which is exactly why attribution also requires proximity to the slick.",
            "warn",
        )

        st_folium(maps.traffic_map(scored), use_container_width=True,
                  height=460, returned_objects=[], key="fleet_map")

    # ------------------------------------------------------------------ vessel
    with tab_vessel:
        names = rollup["vessel_name"].tolist()
        chosen = st.selectbox("Vessel", names, index=names.index("MV WAKASHIO")
                              if "MV WAKASHIO" in names else 0)
        mmsi = int(rollup.loc[rollup["vessel_name"] == chosen, "mmsi"].iloc[0])
        track = scored[scored["mmsi"] == mmsi].reset_index(drop=True)

        flagged = int(track["is_anomaly"].sum())
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(theme.metric_card(
                "Verdict",
                "Flagged for review" if flagged else "Normal behaviour",
                f"{flagged} of {len(track)} pings",
                theme.CRITICAL if flagged else theme.GOOD), unsafe_allow_html=True)
        with c2:
            st.markdown(theme.metric_card(
                "Peak error", f"{track['anomaly_score'].max():.2f}",
                f"threshold {AE_THRESHOLD:.3f}"), unsafe_allow_html=True)
        with c3:
            st.markdown(theme.metric_card(
                "Flag", str(track["flag"].iloc[0]),
                str(track["vessel_type"].iloc[0])), unsafe_allow_html=True)
        with c4:
            st.markdown(theme.metric_card(
                "Top driver",
                track.loc[track["anomaly_score"].idxmax(), "top_feature"],
                "largest reconstruction error"), unsafe_allow_html=True)

        st.markdown("##### Reconstruction error over time")
        st.altair_chart(charts.anomaly_timeline(track, AE_THRESHOLD),
                        use_container_width=True)
        st.caption("Log-scaled. The dashed line is the training-time decision threshold; "
                   "points above it are flagged.")

        # Ping-level inspection
        st.markdown("##### Inspect a single ping")
        idx = st.slider("Ping", 0, len(track) - 1,
                        int(track["anomaly_score"].idxmax()), key="ais_ping")
        row = track.iloc[idx]
        result = engine.score_single_ping({f: float(row[f]) for f in FEATURE_ORDER})

        a, b = st.columns([1, 1.25])
        with a:
            color = theme.SEVERITY_COLORS[result.severity]
            st.markdown(
                f"""
<div class="pos-card">
  <div class="pos-label">Ping {idx} &middot; {row['timestamp']:%Y-%m-%d %H:%M} UTC</div>
  <div style="margin:.4rem 0 .7rem">
    {theme.pill('FLAGGED FOR REVIEW' if result.is_anomaly else 'NORMAL BEHAVIOUR', color)}
    {theme.pill(result.severity.upper(), color) if result.is_anomaly else ''}
  </div>
  <div class="mono" style="color:{theme.MUTED};line-height:1.75">
    position &nbsp;{row['latitude']:.5f}, {row['longitude']:.5f}<br>
    speed &nbsp;&nbsp;&nbsp;&nbsp;{row['speed']:.1f} kn &nbsp;&middot;&nbsp; course {row['course']:.0f}°<br>
    rot &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{row['rot']:.0f} °/min<br>
    status &nbsp;&nbsp;&nbsp;{ais_mod.status_label(row['status'])}<br>
    phase &nbsp;&nbsp;&nbsp;&nbsp;{row.get('phase', '-')}
  </div>
  <div class="pos-sub" style="margin-top:.6rem">
    Reconstruction error <b style="color:{color}">{result.score:.4f}</b>
    against a threshold of {AE_THRESHOLD:.4f}
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
        with b:
            st.markdown("**Which features drove the score**")
            contrib = pd.DataFrame(
                [{"Feature": f, "Squared error": e} for f, e in
                 sorted(result.per_feature_error.items(), key=lambda kv: -kv[1])[:6]]
            )
            st.dataframe(contrib.style.format({"Squared error": "{:.4f}"}),
                         use_container_width=True, hide_index=True)
            st.caption("Per-feature reconstruction error, largest first — the model's "
                       "own account of what looked wrong.")

        with st.expander("Full ping history"):
            cols = ["timestamp", "latitude", "longitude", "speed", "course", "rot",
                    "status", "phase", "anomaly_score", "is_anomaly", "severity",
                    "top_feature"]
            st.dataframe(track[[c for c in cols if c in track.columns]],
                         use_container_width=True, hide_index=True, height=320)

        st.download_button(
            "Download this day's AIS as CSV",
            data=to_csv_bytes(engine.scenario()["ais"]),
            file_name="mauritius_aoi_20200725_ais.csv",
            mime="text/csv",
        )

    # ------------------------------------------------------------------ manual
    with tab_manual:
        st.markdown("##### Score an arbitrary ping")
        st.caption("The model consumes exactly these eleven features, in this order. "
                   "The five `_diff` features are the change since the vessel's previous ping.")

        c1, c2, c3 = st.columns(3)
        with c1:
            speed = st.number_input("speed (kn)", 0.0, 40.0, 0.3, 0.1)
            course = st.number_input("course (°)", 0.0, 359.9, 246.0, 1.0)
            rot = st.number_input("rot (°/min)", -127.0, 127.0, 127.0, 1.0)
            msg_type = st.number_input("msg_type", 1, 27, 1, 1)
        with c2:
            status = st.number_input("status", 0, 15, 6, 1)
            accuracy = st.number_input("accuracy", 0, 1, 1, 1)
            course_diff = st.number_input("course_diff", -180.0, 180.0, 41.0, 1.0)
            rot_diff = st.number_input("rot_diff", -254.0, 254.0, 127.0, 1.0)
        with c3:
            speed_diff = st.number_input("speed_diff", -40.0, 40.0, -10.6, 0.1)
            lat_diff = st.number_input("lat_diff", -1.0, 1.0, -0.0001, 0.0001, format="%.5f")
            long_diff = st.number_input("long_diff", -1.0, 1.0, -0.0001, 0.0001, format="%.5f")

        st.caption(f"Navigational status {int(status)}: "
                   f"**{ais_mod.status_label(status)}**")

        row = dict(speed=speed, course=course, rot=rot, msg_type=msg_type,
                   status=status, accuracy=accuracy, course_diff=course_diff,
                   rot_diff=rot_diff, speed_diff=speed_diff,
                   lat_diff=lat_diff, long_diff=long_diff)
        result = engine.score_single_ping(row)
        color = theme.SEVERITY_COLORS[result.severity]

        st.markdown(
            f"""
<div class="pos-card">
  <div style="display:flex;align-items:center;gap:.7rem">
    {theme.pill('FLAGGED FOR REVIEW' if result.is_anomaly else 'NORMAL BEHAVIOUR', color)}
    <span style="font-size:1.35rem;font-weight:660;color:{color}">{result.score:.4f}</span>
    <span style="color:{theme.MUTED};font-size:.82rem">threshold {AE_THRESHOLD:.4f}</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        top = result.top_contributors(4)
        if top:
            st.markdown("**Largest error contributors:** " +
                        ", ".join(f"`{f}` ({e:.3f})" for f, e in top))

    theme.provenance(
        f"Precision {card['precision']:.0%} · recall {card['recall']:.0%} on held-out data. "
        f"The wording throughout is \"flagged for review\", never \"confirmed violation\", "
        f"because a screening model with 41% recall cannot clear a vessel — only nominate one."
    )
