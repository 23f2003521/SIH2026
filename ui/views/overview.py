"""Incident console -- the one screen that tells the whole story."""
from __future__ import annotations

import streamlit as st

from poseatsea.scenario.wakashio import INCIDENT

from streamlit_folium import st_folium

from .. import charts, engine, maps, theme


def render() -> None:
    sc = engine.scenario()
    incident = sc["incident"]
    vessel = incident["vessel"]

    st.markdown("## Maritime surveillance overview")
    st.markdown(
        f"#### {incident['name']} &nbsp;<span style='color:{theme.MUTED};font-weight:400'>"
        f"{incident['location']}</span>",
        unsafe_allow_html=True,
    )

    scored = engine.scored_ais()
    spill_lat, spill_lon = sc["spill_position"]
    results = engine.attribution(spill_lat, spill_lon, 15.0, 6.0)
    top = results[0] if results else None

    # ---------------------------------------------------------------- headline
    c1, c2, c3, c4 = st.columns(4)
    flagged_vessels = int(scored.groupby("mmsi")["is_anomaly"].any().sum())
    with c1:
        st.markdown(theme.metric_card(
            "Vessels tracked", str(scored["mmsi"].nunique()),
            f"{len(scored):,} AIS pings"), unsafe_allow_html=True)
    with c2:
        st.markdown(theme.metric_card(
            "Flagged for review", str(flagged_vessels),
            f"{int(scored['is_anomaly'].sum())} anomalous pings",
            theme.WARN if flagged_vessels else theme.GOOD), unsafe_allow_html=True)
    with c3:
        st.markdown(theme.metric_card(
            "Primary suspect", top["vessel_name"] if top else "--",
            f"attribution score {top['score']:.2f}" if top else "",
            theme.CRITICAL if top else theme.MUTED), unsafe_allow_html=True)
    with c4:
        st.markdown(theme.metric_card(
            "Oil released", f"{incident['oil_released_t']:,} t",
            "of 3,894 t bunkers aboard", theme.BAD), unsafe_allow_html=True)

    st.markdown("")

    # ---------------------------------------------------------------- the story
    left, right = st.columns([1.55, 1])

    with left:
        st.markdown("##### Traffic in the area of interest")
        st_folium(
            maps.traffic_map(scored,
                             spill={"latitude": spill_lat, "longitude": spill_lon},
                             highlight_mmsi=vessel["mmsi"]),
            use_container_width=True, height=520, returned_objects=[],
            key="overview_map",
        )
        st.markdown(maps.legend([
            {"color": theme.CRITICAL, "label": "MV Wakashio (suspect)"},
            {"color": theme.WARN, "label": "Flagged AIS ping"},
            {"color": theme.ACCENT, "label": "Other traffic"},
        ]), unsafe_allow_html=True)
        theme.provenance(
            "Esri World Imagery. The red ring is the 5 km hazard radius around "
            "Pointe d'Esny; amber dots are pings the autoencoder flagged."
        )

    with right:
        st.markdown("##### What happened")
        st.markdown(
            f"""
<div class="pos-card">
<div class="pos-label">Vessel</div>
<div style="font-size:1.05rem;font-weight:640">{vessel['name']}</div>
<div class="mono" style="color:{theme.MUTED};margin-top:.35rem">
MMSI {vessel['mmsi']} &middot; IMO {vessel['imo']}<br>
{vessel['flag']} flag &middot; {vessel['type']}<br>
{vessel['length_m']} m &middot; {vessel['dwt']:,} DWT
</div>
<div class="pos-sub" style="margin-top:.55rem">{incident['voyage']}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
<div class="pos-card">
<div class="pos-label">Sequence</div>
<div style="font-size:.87rem;line-height:1.65">
<b>{incident['grounding_local']}</b> &mdash; struck the reef at
{incident['position'][0]:.4f}, {incident['position'][1]:.4f}<br>
<b>{incident['leak_began']}</b> &mdash; hull breached, oil began escaping<br>
<b>15 August 2020</b> &mdash; vessel broke in two
</div>
<div class="pos-sub" style="margin-top:.6rem">{incident['cause_summary']}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    # ---------------------------------------------------------------- verdict
    if top:
        st.markdown("##### Attribution")
        band_color = theme.BAND_COLORS.get(top["band"], theme.ACCENT)
        evidence = "".join(f"<li>{e}</li>" for e in top["evidence"])
        st.markdown(
            f"""
<div class="pos-card">
  <div style="display:flex;align-items:center;gap:.65rem;margin-bottom:.5rem">
    <span style="font-size:1.15rem;font-weight:660">{top['vessel_name']}</span>
    {theme.pill(top['band'].upper(), band_color)}
    <span style="color:{theme.MUTED};font-size:.8rem">score {top['score']:.3f}</span>
  </div>
  <ul class="evidence" style="margin:0 0 0 1.05rem;padding:0">{evidence}</ul>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("")
        dcol1, dcol2 = st.columns([1.75, 1])
        with dcol1:
            st.markdown(
                f"""
<div class="pos-card" style="border-left:3px solid {theme.ACCENT}">
  <div class="pos-label">Maritime Law Enforcement &amp; Evidentiary Dossier</div>
  <div style="font-size:.95rem;font-weight:600;margin:.2rem 0 .2rem">Official IMO / MARPOL Annex I Forensic Report</div>
  <div class="pos-sub" style="font-size:.78rem;line-height:1.5">
    Algorithmically compiles incident casualty telemetry, Sentinel-1 SAR slick classification metrics,
    multi-criteria fleet attribution rankings, and SHA-256 chain-of-custody hashes
    into an evidentiary PDF for Coast Guard, flag state, and port authority inquiries.
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
        with dcol2:
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            from poseatsea import report as report_mod
            pdf_data = report_mod.generate_dossier_pdf(
                incident=incident,
                attribution_results=results,
            )
            st.download_button(
                label="📥 Export Forensic Dossier (PDF)",
                data=pdf_data,
                file_name=f"POSEATSEA_Forensic_Dossier_{vessel['mmsi']}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


