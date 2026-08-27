"""
POSEatSea -- maritime oil-spill detection and vessel attribution console.

Run with:  streamlit run ui/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="POSEatSea — Oil Spill Detection & Attribution",
    page_icon="🛰",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui import engine, theme  # noqa: E402
from ui.views import (  # noqa: E402
    ais_view,
    attribution_view,
    overview,
    sar_view,
    system_view,
    trajectory_view,
)

theme.inject()

# Names deliberately match the pipeline vocabulary the team already uses, so
# the navigation reads the same way the problem statement does.
PAGES = {
    "Overview": ("🎯", overview.render),
    "Route Deviation (LSTM)": ("🧭", trajectory_view.render),
    "AIS Anomaly Detection": ("📡", ais_view.render),
    "SAR Oil Spill Segmenter": ("🛰", sar_view.render),
    "Attribution Pipeline": ("⚖", attribution_view.render),
    "System": ("⚙", system_view.render),
}


def sidebar() -> str:
    with st.sidebar:
        st.markdown(
            f"""
<div style="padding:.2rem 0 .8rem">
  <div style="font-size:1.15rem;font-weight:700;letter-spacing:-.02em">POSEatSea</div>
  <div style="font-size:.74rem;color:{theme.MUTED};line-height:1.5;margin-top:.15rem">
    Prediction of Oil Spill Events at Sea<br>
    Satellite &amp; AIS fusion for spill attribution
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        choice = st.radio("Navigation", list(PAGES),
                          format_func=lambda k: f"{PAGES[k][0]}  {k}",
                          label_visibility="collapsed")

        st.divider()
        _fleet_monitor()
        st.divider()

        # Live model residency, so the cost of the heavy model is never hidden.
        st.markdown("<div class='pos-label'>Model registry</div>", unsafe_allow_html=True)
        for m in engine.registry().status()["models"]:
            dot = theme.GOOD if m["loaded"] else theme.LINE
            detail = f"{m['load_seconds']:.1f}s" if m["loaded"] else "idle"
            st.markdown(
                f"<div style='font-size:.75rem;color:{theme.MUTED};margin:.2rem 0'>"
                f"<span style='display:inline-block;width:7px;height:7px;border-radius:50%;"
                f"background:{dot};margin-right:7px'></span>{m['name']}"
                f"<span style='float:right'>{detail}</span></div>",
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown(
            f"<div style='font-size:.7rem;color:{theme.MUTED};line-height:1.6'>"
            f"Real AIS &mdash; Mauritius AOI, July 2020. The <b>MV Wakashio</b> "
            f"grounding of 25 July 2020 is recorded in this feed by the "
            f"vessel itself.</div>",
            unsafe_allow_html=True,
        )
    return choice


def _fleet_monitor() -> None:
    """
    Per-vessel status cards.

    The sidebar is the one surface visible on every page, so it carries the
    fleet at a glance: who is being tracked, and who the detector has flagged.
    """
    scored = engine.scored_ais()
    st.markdown("<div class='pos-label'>Fleet monitor</div>", unsafe_allow_html=True)

    rollup = scored.groupby("mmsi").agg(
        name=("vessel_name", "first"),
        flag=("flag", "first"),
        vtype=("vessel_type", "first"),
        flagged=("is_anomaly", "sum"),
        peak=("anomaly_score", "max"),
    ).reset_index().sort_values("peak", ascending=False)

    for r in rollup.itertuples():
        last = scored[scored["mmsi"] == r.mmsi].iloc[-1]
        if r.flagged:
            status, color = "FLAGGED", theme.CRITICAL
        else:
            status, color = "Normal", theme.GOOD
        st.markdown(
            f"""<div style="background:{theme.PANEL_2};border:1px solid {theme.LINE};
 border-left:3px solid {color};border-radius:8px;padding:8px 11px;margin-bottom:7px">
  <div style="font-size:.8rem;font-weight:660;color:{theme.TEXT}">{r.name}</div>
  <div style="font-size:.68rem;color:{theme.MUTED};margin-top:1px">
    MMSI {r.mmsi} &middot; {r.flag}<br>{r.vtype}
  </div>
  <div style="margin-top:5px;font-size:.71rem">
    <span style="color:{color};font-weight:640">{status}</span>
    <span style="color:{theme.MUTED}"> &nbsp;{last['speed']:.1f} kn
      &nbsp;{last['course']:.0f}&deg;</span>
  </div>
</div>""",
            unsafe_allow_html=True,
        )


def main() -> None:
    choice = sidebar()
    PAGES[choice][1]()


if __name__ == "__main__":
    main()
