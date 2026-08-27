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

PAGES = {
    "Incident console": ("🎯", overview.render),
    "SAR segmentation": ("🛰", sar_view.render),
    "AIS anomalies": ("📡", ais_view.render),
    "Trajectory": ("🧭", trajectory_view.render),
    "Attribution": ("⚖", attribution_view.render),
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

        # Live model residency, so the cost of the heavy model is never hidden.
        st.markdown(f"<div class='pos-label'>Model registry</div>", unsafe_allow_html=True)
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
            f"Demonstration scenario reconstructs the <b>MV Wakashio</b> grounding "
            f"(Mauritius, 25 July 2020) from the public casualty record. "
            f"AIS pings are synthesis, not recovered signal.</div>",
            unsafe_allow_html=True,
        )
    return choice


def main() -> None:
    choice = sidebar()
    PAGES[choice][1]()


if __name__ == "__main__":
    main()
