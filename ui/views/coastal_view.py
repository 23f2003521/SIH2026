"""Dedicated Coastal Vulnerability page -- InVEST biophysical exposure & habitat impact model."""
from __future__ import annotations

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from poseatsea import coastal
from .. import maps, theme


def render() -> None:
    st.markdown("## Coastal Vulnerability")

    summary = coastal.coastal_summary()
    df = coastal.load_exposure_points()

    mode = summary.get("execution_mode", "cached")
    if mode == "live":
        mode_pill = f'<span class="pill" style="background:#3ddc9722;color:#3ddc97;border:1px solid #3ddc9755">● Live InVEST Model</span>'
    else:
        mode_pill = f'<span class="pill" style="background:#8ea3bf22;color:#8ea3bf;border:1px solid #8ea3bf44">Using cached analysis</span>'

    st.markdown(
        f"#### InVEST-based coastal exposure assessment &nbsp;{mode_pill}&nbsp;"
        f"<span style='color:{theme.MUTED};font-weight:400'>"
        f"Natural Capital Project biophysical model &middot; Mauritius AOI</span>",
        unsafe_allow_html=True,
    )

    if not summary.get("available") or df.empty:
        theme.banner(
            "<b>InVEST Coastal Vulnerability layer unavailable.</b> "
            "Neither live InVEST execution nor cached analysis could be loaded.",
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
    selected_seg_id = st.session_state.get("coastal_seg_slider", 77)

    left, right = st.columns([1.55, 1.1])

    with left:
        c_title, c_sel = st.columns([1, 1.45])
        with c_title:
            st.markdown("##### Coastal exposure map")
        with c_sel:
            THEMATIC_LAYERS = [
                "Overall Exposure",
                "Natural Habitat Protection (R_hab)",
                "Wave Exposure (R_wave)",
                "Wind Fetch Exposure (R_wind)",
                "Coastal Relief (R_relief)",
                "Storm Surge Potential (R_surge)",
            ]
            active_layer = st.selectbox(
                "Map Layer",
                THEMATIC_LAYERS,
                index=0,
                label_visibility="collapsed",
                key="coastal_active_layer_sel",
            )

        c_t1, c_t2 = st.columns(2)
        show_habitats = c_t1.checkbox(
            "Show Habitat Polygons (Coral & Mangrove)",
            value=True,
            key="cv_show_habitats",
        )
        show_buffers = c_t2.checkbox(
            "Highlight Active Segment & Buffers",
            value=True,
            key="cv_show_buffers",
        )

        st_folium(
            maps.coastal_vulnerability_map(
                height=520,
                active_layer=active_layer,
                selected_segment=selected_seg_id,
                show_habitats=show_habitats,
                show_buffers=show_buffers,
            ),
            use_container_width=True,
            height=520,
            returned_objects=[],
            key=f"coastal_map_{active_layer}_{selected_seg_id}_{show_habitats}_{show_buffers}",
        )

        if active_layer == "Overall Exposure":
            st.markdown(
                f"<div style='font-size:.78rem;font-weight:600;color:{theme.MUTED};margin-top:6px'>"
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
        else:
            st.markdown(
                f"<div style='font-size:.78rem;font-weight:600;color:{theme.MUTED};margin-top:6px'>"
                f"{active_layer.upper()} (RANKS 1-5)</div>",
                unsafe_allow_html=True,
            )
            st.markdown(maps.component_legend(), unsafe_allow_html=True)

        if show_habitats or show_buffers:
            st.markdown(
                f"<div style='font-size:.78rem;font-weight:600;color:{theme.MUTED};margin-top:4px'>"
                f"HABITAT & OPERATIONAL FEATURES</div>",
                unsafe_allow_html=True,
            )
            st.markdown(maps.habitat_legend(), unsafe_allow_html=True)

        theme.provenance(
            "InVEST Coastal Vulnerability model (Sharp et al. / Arkema et al. 2013). "
            "Inputs: GEBCO 2026 bathymetry, Copernicus DSM 30m, ERA5 WWIII 16-sector wave/wind, "
            "and UNEP-WCMC coral/mangrove habitats."
        )

    with right:
        # Segment inspector slider
        st.markdown("##### Shore segment inspector")
        segment_id = st.slider("Select Shore Segment", 0, len(df) - 1, selected_seg_id, key="coastal_seg_slider")
        prof = coastal.get_segment_habitat_profile(segment_id)

        # Segment Header Card
        st.markdown(
            f"""
<div class="pos-card" style="margin-bottom:0.75rem">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <span style="font-weight:700;font-size:1.02rem">Shore Segment #{prof['shore_id']}</span>
    <span style="padding:2px 8px;border-radius:4px;background:{prof['tier_color']}22;color:{prof['tier_color']};font-weight:700;font-size:.78rem;border:1px solid {prof['tier_color']}44">{prof['exposure_tier']} Exposure</span>
  </div>
  <div class="mono" style="font-size:.76rem;color:{theme.MUTED};margin-top:4px">
    {prof['latitude']:.4f}&deg; N, {prof['longitude']:.4f}&deg; E &middot; {prof['dist_to_grounding_km']:.2f} km to MV Wakashio
  </div>
  <div style="display:flex;gap:6px;margin-top:8px">
    <div style="flex:1;background:{theme.PANEL_2};padding:6px 8px;border-radius:6px;border:1px solid {theme.LINE}">
      <div style="font-size:.67rem;color:{theme.MUTED};text-transform:uppercase">Exposure</div>
      <div style="font-size:1.05rem;font-weight:700;color:{prof['tier_color']}">{prof['exposure_index']:.2f}</div>
    </div>
    <div style="flex:1.1;background:{theme.PANEL_2};padding:6px 8px;border-radius:6px;border:1px solid {theme.LINE}">
      <div style="font-size:.67rem;color:{theme.MUTED};text-transform:uppercase">Habitat Role</div>
      <div style="font-size:1.05rem;font-weight:700;color:{theme.GOOD}">▼ {prof['habitat_role']:.2f} <span style="font-size:.72rem;font-weight:500">(-{prof['habitat_role_pct']:.0f}%)</span></div>
    </div>
    <div style="flex:1.1;background:{theme.PANEL_2};padding:6px 8px;border-radius:6px;border:1px solid {theme.LINE}">
      <div style="font-size:.67rem;color:{theme.MUTED};text-transform:uppercase">Active Habitats</div>
      <div style="font-size:1.05rem;font-weight:700;color:{theme.ACCENT}">{prof['habitats_count']} <span style="font-size:.72rem;font-weight:500">({prof['species_count']} taxa)</span></div>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        # Habitat & Species Impact Section
        st.markdown("##### Habitat & species impact")

        # 1. Coral Reef Card
        coral_reg = prof["coral_registry"]
        if prof["has_coral"]:
            coral_badge = f'<span style="padding:2px 7px;border-radius:4px;background:#00d2d222;color:#00d2d2;font-weight:600;font-size:.72rem;border:1px solid #00d2d255">● 2,000 m Buffer Active ({prof["coral_area_km2"]:.1f} km²)</span>'
            coral_border = "#00d2d255"
            coral_taxa_html = "".join([
                f'<div style="font-size:.78rem;line-height:1.45;margin-top:3px">'
                f'&bull; <i>{t["scientific"]}</i> ({t["common"]}) &middot; <span style="color:#00d2d2;font-weight:600">{t["status"]}</span><br>'
                f'<span style="color:{theme.MUTED};font-size:.73rem;padding-left:8px">{t["role"]}</span></div>'
                for t in coral_reg["taxa"]
            ])
        else:
            coral_badge = f'<span style="padding:2px 7px;border-radius:4px;background:#8b949e22;color:#8b949e;font-weight:600;font-size:.72rem;border:1px solid #8b949e44">○ No Buffer Within 2 km</span>'
            coral_border = theme.LINE
            coral_taxa_html = f'<div style="font-size:.75rem;color:{theme.MUTED};margin-top:2px">Segment is outside active coral reef buffer (&gt; 2,000 m).</div>'

        st.markdown(
            f"""
<div class="pos-card" style="border-left:3px solid #00d2d2;padding:0.75rem 0.9rem;margin-bottom:0.6rem">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <span style="font-weight:650;font-size:.88rem;color:#00d2d2">Coral Reef Barrier</span>
    {coral_badge}
  </div>
  <div style="font-size:.79rem;line-height:1.45;margin-top:4px">
    <b>Coastal Defense Role:</b> {coral_reg['protection_service']}.
  </div>
  <div style="font-size:.76rem;color:{theme.MUTED};margin-top:5px;font-weight:600">DOCUMENTED ASSOCIATED SPECIES / TAXA:</div>
  {coral_taxa_html}
</div>
""",
            unsafe_allow_html=True,
        )

        # 2. Mangrove Card
        mangrove_reg = prof["mangrove_registry"]
        if prof["has_mangrove"]:
            mangrove_badge = f'<span style="padding:2px 7px;border-radius:4px;background:#2ea04322;color:#2ea043;font-weight:600;font-size:.72rem;border:1px solid #2ea04355">● 1,000 m Buffer Active ({prof["mangrove_area_km2"]:.1f} km²)</span>'
            mangrove_border = "#2ea04355"
            mangrove_taxa_html = "".join([
                f'<div style="font-size:.78rem;line-height:1.45;margin-top:3px">'
                f'&bull; <i>{t["scientific"]}</i> ({t["common"]}) &middot; <span style="color:#2ea043;font-weight:600">{t["status"]}</span><br>'
                f'<span style="color:{theme.MUTED};font-size:.73rem;padding-left:8px">{t["role"]}</span></div>'
                for t in mangrove_reg["taxa"]
            ])
        else:
            mangrove_badge = f'<span style="padding:2px 7px;border-radius:4px;background:#8b949e22;color:#8b949e;font-weight:600;font-size:.72rem;border:1px solid #8b949e44">○ No Buffer Within 1 km</span>'
            mangrove_border = theme.LINE
            mangrove_taxa_html = f'<div style="font-size:.75rem;color:{theme.MUTED};margin-top:2px">Segment is outside active estuarine mangrove buffer (&gt; 1,000 m).</div>'

        st.markdown(
            f"""
<div class="pos-card" style="border-left:3px solid #2ea043;padding:0.75rem 0.9rem;margin-bottom:0.6rem">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <span style="font-weight:650;font-size:.88rem;color:#2ea043">Estuarine Mangrove Forest</span>
    {mangrove_badge}
  </div>
  <div style="font-size:.79rem;line-height:1.45;margin-top:4px">
    <b>Coastal Defense Role:</b> {mangrove_reg['protection_service']}.
  </div>
  <div style="font-size:.76rem;color:{theme.MUTED};margin-top:5px;font-weight:600">DOCUMENTED ASSOCIATED SPECIES / TAXA:</div>
  {mangrove_taxa_html}
</div>
""",
            unsafe_allow_html=True,
        )

        # 3. Seagrass Meadow Card
        seagrass_reg = prof["seagrass_registry"]
        seagrass_taxa_html = "".join([
            f'<div style="font-size:.78rem;line-height:1.45;margin-top:3px">'
            f'&bull; <i>{t["scientific"]}</i> ({t["common"]}) &middot; <span style="color:{theme.MUTED};font-weight:600">{t["status"]}</span><br>'
            f'<span style="color:{theme.MUTED};font-size:.73rem;padding-left:8px">{t["role"]}</span></div>'
            for t in seagrass_reg["taxa"]
        ])
        st.markdown(
            f"""
<div class="pos-card" style="border-left:3px solid #8b949e;padding:0.75rem 0.9rem;margin-bottom:0.6rem">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <span style="font-weight:650;font-size:.88rem;color:#c9d1d9">Lagoon Seagrass Meadows</span>
    <span style="padding:2px 7px;border-radius:4px;background:#8b949e22;color:#8b949e;font-weight:600;font-size:.72rem;border:1px solid #8b949e44">Unmapped in InVEST Model</span>
  </div>
  <div style="font-size:.79rem;line-height:1.45;margin-top:4px">
    <b>Ecological &amp; Defense Role:</b> {seagrass_reg['protection_service']}.
  </div>
  <div style="font-size:.76rem;color:{theme.MUTED};margin-top:5px;font-weight:600">CHARACTERISTIC LAGOON TAXA:</div>
  {seagrass_taxa_html}
  <div style="font-size:.72rem;color:{theme.MUTED};margin-top:4px;font-style:italic">
    Note: Seagrass beds are documented in Blue Bay lagoon but are unmapped in the current GIS vector inputs.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        # 4. Tailored Ecological Narrative Callout
        st.markdown(
            f"""
<div class="pos-card" style="background:{theme.PANEL_2};border-left:3px solid {theme.ACCENT};padding:0.75rem 0.9rem;margin-bottom:0.6rem">
  <div class="pos-label" style="color:{theme.ACCENT}">Ecological &amp; Operational Assessment</div>
  <div style="font-size:.82rem;line-height:1.55;color:{theme.TEXT}">
    {prof['ecological_summary']}
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        # 5. Biophysical Drivers Component Breakdown
        pt = df.iloc[segment_id]
        r_hab_val = float(pt.get("r_hab", 1.0))
        r_wave_val = float(pt.get("r_wave", 1.0))
        r_wind_val = float(pt.get("r_wind", 1.0))
        r_relief_val = float(pt.get("r_relief", 1.0))
        r_surge_val = float(pt.get("r_surge", 1.0))

        with st.expander("Biophysical components breakdown (Ranks 1–5)", expanded=False):
            st.markdown(
                f"""
<div style="font-size:.83rem;line-height:1.7">
<b>Storm Surge Potential (R_surge):</b> {r_surge_val:.0f} / 5<br>
<b>Wind Fetch Exposure (R_wind):</b> {r_wind_val:.0f} / 5<br>
<b>Natural Habitats Buffer (R_hab):</b> {r_hab_val:.0f} / 5<br>
<b>Wave Energy Exposure (R_wave):</b> {r_wave_val:.0f} / 5<br>
<b>Coastal Elevation / Relief (R_relief):</b> {r_relief_val:.0f} / 5
</div>
<div style="font-size:.74rem;color:{theme.MUTED};margin-top:6px">
InVEST ranks range 1 (maximum natural protection / lowest exposure) to 5 (least protected / highest exposure).
</div>
""",
                unsafe_allow_html=True,
            )
