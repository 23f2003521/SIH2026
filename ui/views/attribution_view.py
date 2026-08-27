"""Attribution -- correlating a spill against AIS traffic."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from poseatsea import fusion

from streamlit_folium import st_folium

from .. import charts, engine, maps, theme


def render() -> None:
    st.markdown("## Attribution pipeline")
    st.markdown(
        f"<span style='color:{theme.MUTED}'>Correlates a detected slick against every "
        f"vessel in the area on four measurable axes. The ranking is a transparent "
        f"weighted sum, not a learned model &mdash; every component is shown.</span>",
        unsafe_allow_html=True,
    )

    sc = engine.scenario()
    default_lat, default_lon = sc["spill_position"]

    # ------------------------------------------------------------------ controls
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        lat = st.number_input("Slick latitude", -90.0, 90.0, float(default_lat),
                              0.001, format="%.4f")
    with c2:
        lon = st.number_input("Slick longitude", -180.0, 180.0, float(default_lon),
                              0.001, format="%.4f")
    with c3:
        radius = st.slider("Search radius (km)", 1.0, 50.0, 15.0, 1.0)
    with c4:
        window = st.slider("Time window (± hours)", 0.5, 24.0, 6.0, 0.5)

    results = engine.attribution(lat, lon, radius, window)
    if not results:
        theme.banner("No AIS traffic inside this search window.", "warn")
        return

    summary = fusion.summarise([_Rehydrate(r) for r in results])

    # ------------------------------------------------------------------ verdict
    top = results[0]
    band_color = theme.BAND_COLORS.get(top["band"], theme.ACCENT)
    theme.banner(f"<b>{summary['verdict']}</b>", "bad" if top["score"] >= 0.7 else "warn")

    st.markdown("##### Ranked candidates")
    for r in results:
        color = theme.BAND_COLORS.get(r["band"], theme.ACCENT)
        evidence = "".join(f"<li>{e}</li>" for e in r["evidence"])
        bar = int(round(r["score"] * 100))
        st.markdown(
            f"""
<div class="pos-card">
  <div style="display:flex;align-items:center;gap:.7rem;flex-wrap:wrap">
    <span style="font-size:1.02rem;font-weight:650">{r['vessel_name']}</span>
    {theme.pill(r['band'].upper(), color)}
    <span class="mono" style="color:{theme.MUTED}">MMSI {r['mmsi']} &middot;
      {r['flag']} &middot; {r['vessel_type']}</span>
    <span style="margin-left:auto;font-size:1.1rem;font-weight:660;color:{color}">
      {r['score']:.3f}</span>
  </div>
  <div style="background:{theme.PANEL_2};border-radius:999px;height:6px;margin:.6rem 0 .7rem">
    <div style="background:{color};width:{bar}%;height:6px;border-radius:999px"></div>
  </div>
  <ul class="evidence" style="margin:0 0 0 1.05rem;padding:0;color:{theme.MUTED}">
    {evidence}
  </ul>
</div>
""",
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------ evidence
    left, right = st.columns([1, 1])
    with left:
        st.markdown("##### What drives each score")
        st.altair_chart(charts.attribution_bars(results), use_container_width=True)
        weights = " · ".join(f"{k} {v:.0%}" for k, v in fusion.WEIGHTS.items())
        st.caption(f"Weights: {weights}. These are a policy choice, not a measurement — "
                   f"an investigator who weighs dwell time above proximity should say so "
                   f"and change them.")
    with right:
        st.markdown("##### Traffic against the slick")
        st_folium(
            maps.traffic_map(engine.scored_ais(),
                             spill={"latitude": lat, "longitude": lon},
                             highlight_mmsi=top["mmsi"]),
            use_container_width=True, height=420, returned_objects=[],
            key="attribution_map",
        )

    with st.expander("Comparison table"):
        table = pd.DataFrame([{
            "Vessel": r["vessel_name"],
            "Score": r["score"],
            "Band": r["band"],
            "Closest (km)": r["min_distance_km"],
            "Dwell (min)": r["minutes_in_radius"],
            "Flagged pings": r["flagged_pings"],
            "Peak anomaly": r["max_anomaly_score"],
            "Max deviation (km)": r["max_deviation_km"],
        } for r in results])
        st.dataframe(table, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------ caveat
    theme.banner(
        f"<b>Standing limitation.</b> {fusion.drift_caveat()}",
        "warn",
    )

    with st.expander("Why drift modelling is absent, and what it would take"):
        st.markdown(
            """
Tracing a slick backwards to its release point and time — *hindcasting* — needs
inputs this system does not have: gridded ocean surface currents, wind fields at
the time of release, and an advection-diffusion model for weathering oil.

That is a genuine and well-understood discipline, and faking it would be worse
than omitting it: a plausible-looking backtrack with no physics behind it would
send an investigation to the wrong vessel with false confidence.

What the console does instead is state its assumption plainly — the oil lies
where it was seen — and let the operator judge whether elapsed time makes that
assumption safe. For the Wakashio case it is unusually safe: the
source was a stationary hull pinned on a reef, continuously transmitting AIS
from the position the oil was escaping.

Wiring in a drift model (for example from INCOIS ocean-current products) would
replace the fixed search radius with a time-reversed probability field. That is
the natural next component, and it is deliberately not stubbed out here.
"""
        )


class _Rehydrate:
    """Adapter so `fusion.summarise` can consume cached dictionaries."""

    def __init__(self, d: dict):
        self._d = d
        self.vessel_name = d["vessel_name"]
        self.total_score = d["score"]

    def as_dict(self) -> dict:
        return self._d
