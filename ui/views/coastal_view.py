"""Dedicated Coastal Vulnerability page -- InVEST biophysical exposure model."""
from __future__ import annotations

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from poseatsea import coastal
from .. import maps, theme


def render() -> None:
    st.markdown("## Coastal Vulnerability")
    st.markdown(
        f"#### InVEST-based coastal exposure assessment &nbsp;<span style='color:{theme.MUTED};font-weight:400'>"
        f"Natural Capital Project biophysical model &middot; Mauritius AOI</span>",
        unsafe_allow_html=True,
    )

    summary = coastal.coastal_summary()
    df = coastal.load_exposure_points()

    if not summary.get("available") or df.empty:
        theme.banner(
            "<b>InVEST Coastal Vulnerability output layer not found.</b> "
            "Run <code>python scripts/run_coastal_vulnerability.py</code> to generate exposure points.",
            "warn",
        )
        return

    # ---------------------------------------------------------------- headline
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            theme.metric_card(
                "Shoreline evaluated",
                f"{summary['segments_count']} points",
                "250 m resolution along AOI",
            ),
            unsafe_allow_html=True,
        )
    with c2:
        high_cnt = summary.get("high_risk_segments", 0)
        high_pct = summary.get("high_risk_pct", 0.0)
        tone = theme.CRITICAL if high_cnt > 0 else theme.GOOD
        st.markdown(
            theme.metric_card(
                "High-risk segments",
                f"{high_cnt} ({high_pct:.1f}%)",
                "High or Very High exposure",
                tone,
            ),
            unsafe_allow_html=True,
        )
    with c3:
        max_ei = summary.get("max_exposure", 0.0)
        mean_ei = summary.get("mean_exposure", 0.0)
        st.markdown(
            theme.metric_card(
                "Max exposure index",
                f"{max_ei:.2f} / 5.00",
                f"Mean exposure {mean_ei:.2f}",
                theme.WARN,
            ),
            unsafe_allow_html=True,
        )
    with c4:
        dist_km = summary.get("dist_wakashio_to_high_risk_km", 0.0)
        dist_tone = theme.BAD if dist_km < 5.0 else theme.GOOD
        st.markdown(
            theme.metric_card(
                "Proximity to grounding",
                f"{dist_km:.2f} km",
                "from MV Wakashio to high-risk shore",
                dist_tone,
            ),
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ---------------------------------------------------------------- map & analytics
    left, right = st.columns([1.65, 1])

    with left:
        st.markdown("##### Interactive coastal exposure map")
        st_folium(
            maps.coastal_vulnerability_map(height=540),
            use_container_width=True,
            height=540,
            returned_objects=[],
            key="coastal_full_map",
        )
        st.caption(
            "Layer control in top-right corner allows switching between Overall Exposure "
            "and individual biophysical components (Habitats, Waves, Winds, Relief, Surge)."
        )

        st.markdown(
            f"<div style='font-size:.78rem;font-weight:600;color:{theme.MUTED};margin-top:8px'>"
            f"OVERALL EXPOSURE TIERS</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            maps.legend([
                {"color": "#3fb950", "label": "Low (≤ 2.0)"},
                {"color": "#d29922", "label": "Moderate (2.0 - 3.0)"},
                {"color": "#f85149", "label": "High (3.0 - 4.0)"},
                {"color": "#bd561d", "label": "Very High (> 4.0)"},
            ]),
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div style='font-size:.78rem;font-weight:600;color:{theme.MUTED};margin-top:6px'>"
            f"BIOPHYSICAL COMPONENT RANKS (1-5)</div>",
            unsafe_allow_html=True,
        )
        st.markdown(maps.component_legend(), unsafe_allow_html=True)

        theme.provenance(
            "InVEST Coastal Vulnerability model (Sharp et al. / Arkema et al. 2013). "
            "Inputs: GEBCO 2026 bathymetry, Copernicus DSM 30m, ERA5 WWIII 16-sector wave/wind, "
            "and UNEP-WCMC coral/mangrove habitats."
        )

    with right:
        st.markdown("##### Exposure components")
        st.markdown(
            f"""
<div class="pos-card">
<div class="pos-label">Biophysical Drivers Breakdown</div>
<div style="font-size:.87rem;line-height:1.75;margin-top:.3rem">
<b>Storm Surge Potential (R_surge):</b> {summary['avg_r_surge']:.2f} / 5.0<br>
<b>Wind Fetch Exposure (R_wind):</b> {summary['avg_r_wind']:.2f} / 5.0<br>
<b>Natural Habitats Buffer (R_hab):</b> {summary['avg_r_hab']:.2f} / 5.0<br>
<b>Wave Energy Exposure (R_wave):</b> {summary['avg_r_wave']:.2f} / 5.0<br>
<b>Coastal Elevation/Relief (R_relief):</b> {summary['avg_r_relief']:.2f} / 5.0
</div>
<div class="pos-sub" style="margin-top:.5rem">
Ranks scale 1 to 5. Average surge ({summary['avg_r_surge']:.1f}) and wind fetch ({summary['avg_r_wind']:.1f}) represent the primary environmental exposure drivers across the southeastern Mauritian shore.
</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
<div class="pos-card">
<div class="pos-label">Ecosystem Defense Assessment</div>
<div style="font-size:.85rem;line-height:1.6">
<b>Coral Reef Buffer:</b> 2,000 m protective radius<br>
<b>Mangrove Buffer:</b> 1,000 m attenuation zone<br>
<b>Protection Role:</b> Fringed barrier reefs around Pointe d'Esny and Blue Bay Marine Park dissipate up to 97% of open-ocean swell energy, significantly buffering inshore coastal relief.
</div>
</div>
""",
            unsafe_allow_html=True,
        )

        # Segment inspector
        st.markdown("##### Inspect shore segment")
        segment_id = st.slider("Segment ID", 0, len(df) - 1, 0, key="coastal_seg_slider")
        pt = df.iloc[segment_id]
        shore_id_val = int(pt.get("shore_id", segment_id))
        tier_val = pt.get("exposure_tier", "Unknown")
        color_val = pt.get("color", theme.ACCENT)
        lat_val = float(pt.get("latitude", 0.0))
        lon_val = float(pt.get("longitude", 0.0))
        dist_val = pt.get("dist_to_grounding_km")
        if dist_val is not None and not pd.isna(dist_val):
            dist_str = f"{float(dist_val):.2f} km to reef"
        else:
            dist_str = "-- km to reef"
        ei_val = float(pt.get("exposure_index", 0.0))
        r_hab_val = float(pt.get("r_hab", 1.0))
        r_wave_val = float(pt.get("r_wave", 1.0))
        r_wind_val = float(pt.get("r_wind", 1.0))
        r_relief_val = float(pt.get("r_relief", 1.0))
        r_surge_val = float(pt.get("r_surge", 1.0))

        st.markdown(
            f"""
<div class="pos-card">
<div style="display:flex;justify-content:space-between;align-items:center">
  <span style="font-weight:700;font-size:1rem">Segment #{shore_id_val}</span>
  <span style="padding:2px 8px;border-radius:4px;background:{color_val}22;color:{color_val};font-weight:600;font-size:.78rem">{tier_val}</span>
</div>
<div class="mono" style="font-size:.76rem;color:{theme.MUTED};margin-top:4px">
  {lat_val:.4f}&deg; N, {lon_val:.4f}&deg; E &middot; {dist_str}
</div>
<div style="font-size:.84rem;margin-top:8px;line-height:1.6">
  <b>Cumulative Exposure Index:</b> {ei_val:.2f}<br>
  &bull; Habitats Protection: <b>{r_hab_val:.0f}</b> / 5<br>
  &bull; Wave Exposure: <b>{r_wave_val:.0f}</b> / 5<br>
  &bull; Wind Fetch: <b>{r_wind_val:.0f}</b> / 5<br>
  &bull; Coastal Relief: <b>{r_relief_val:.0f}</b> / 5<br>
  &bull; Storm Surge: <b>{r_surge_val:.0f}</b> / 5
</div>
</div>
""",
            unsafe_allow_html=True,
        )
