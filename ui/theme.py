"""Shared visual language for the console: CSS, colour tokens, small components."""
from __future__ import annotations

from typing import Optional

import streamlit as st

# Palette -- a dark operations console, with the SAR class colours reserved so
# nothing in the chrome collides with the meaning of a segmentation overlay.
INK = "#070d15"
PANEL = "#0e1725"
PANEL_2 = "#141f31"
LINE = "#23324a"
TEXT = "#e6edf6"
MUTED = "#8ea3bf"
ACCENT = "#31c8e8"
GOOD = "#3ddc97"
WARN = "#f0b429"
BAD = "#ff5a5f"
CRITICAL = "#ff2e63"

SEVERITY_COLORS = {
    "normal": GOOD,
    "elevated": WARN,
    "high": "#ff8c42",
    "critical": CRITICAL,
}

BAND_COLORS = {
    "primary suspect": CRITICAL,
    "person of interest": WARN,
    "in the area": ACCENT,
    "cleared by proximity": MUTED,
}

# Keyed by the real MMSIs in the Mauritius AOI feed. The casualty is the only
# one in the alert colour; anything not listed falls back to ACCENT.
VESSEL_COLORS = {
    372711000: CRITICAL,     # WAKASHIO -- the casualty
    564796000: "#6c8cff",    # KOTA SURIA
    538006057: "#38d9a9",    # VERY MARIA
    477007600: "#b197fc",    # DHT EDELWEISS
    477848500: "#ffa94d",    # PALONA
    371282000: "#f783ac",    # AQUAVITA SOL
}

CSS = f"""
<style>
  .stApp {{ background: {INK}; }}
  section.main > div {{ padding-top: 1.2rem; }}

  /* Typography */
  html, body, [class*="css"] {{ color: {TEXT}; }}
  h1, h2, h3, h4 {{ letter-spacing: -0.015em; font-weight: 650; }}

  /* Cards */
  .pos-card {{
    background: {PANEL};
    border: 1px solid {LINE};
    border-radius: 12px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.85rem;
  }}
  .pos-card.tight {{ padding: 0.7rem 0.9rem; }}

  .pos-label {{
    font-size: 0.70rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: {MUTED};
    margin-bottom: 0.3rem;
  }}
  .pos-value {{ font-size: 1.7rem; font-weight: 680; line-height: 1.15; }}
  .pos-sub {{ font-size: 0.78rem; color: {MUTED}; margin-top: 0.25rem; }}

  /* Status pills */
  .pill {{
    display: inline-block;
    padding: 0.16rem 0.6rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 620;
    letter-spacing: 0.03em;
    border: 1px solid transparent;
  }}

  /* Banners */
  .banner {{
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-size: 0.86rem;
    line-height: 1.5;
    margin-bottom: 0.85rem;
    border-left: 3px solid {ACCENT};
    background: rgba(49, 200, 232, 0.08);
  }}
  .banner.warn {{ border-left-color: {WARN}; background: rgba(240, 180, 41, 0.09); }}
  .banner.bad  {{ border-left-color: {BAD};  background: rgba(255, 90, 95, 0.09); }}
  .banner.good {{ border-left-color: {GOOD}; background: rgba(61, 220, 151, 0.09); }}
  .banner b {{ color: {TEXT}; }}

  /* Evidence list */
  .evidence li {{ margin-bottom: 0.32rem; font-size: 0.87rem; }}

  /* Legend swatches */
  .swatch {{
    display: inline-block; width: 13px; height: 13px;
    border-radius: 3px; margin-right: 8px;
    vertical-align: -2px; border: 1px solid rgba(255,255,255,0.22);
  }}

  /* Tables */
  .stDataFrame {{ border: 1px solid {LINE}; border-radius: 10px; }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{
    background: {PANEL};
    border-right: 1px solid {LINE};
  }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{ gap: 2px; border-bottom: 1px solid {LINE}; }}
  .stTabs [data-baseweb="tab"] {{
    background: transparent; border-radius: 8px 8px 0 0;
    padding: 0.5rem 1rem; color: {MUTED};
  }}
  .stTabs [aria-selected="true"] {{ background: {PANEL_2}; color: {TEXT}; }}

  /* Mono details */
  .mono {{ font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 0.8rem; }}

  /* Provenance strip */
  .provenance {{
    font-size: 0.74rem; color: {MUTED};
    border-top: 1px dashed {LINE};
    padding-top: 0.55rem; margin-top: 0.9rem;
  }}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def pill(text: str, color: str) -> str:
    return (f'<span class="pill" style="background:{color}22;color:{color};'
            f'border-color:{color}55">{text}</span>')


def metric_card(label: str, value: str, sub: str = "", color: Optional[str] = None) -> str:
    tone = f"color:{color}" if color else ""
    sub_html = f'<div class="pos-sub">{sub}</div>' if sub else ""
    return (f'<div class="pos-card tight"><div class="pos-label">{label}</div>'
            f'<div class="pos-value" style="{tone}">{value}</div>{sub_html}</div>')


def banner(text: str, tone: str = "info") -> None:
    cls = {"info": "", "warn": "warn", "bad": "bad", "good": "good"}.get(tone, "")
    st.markdown(f'<div class="banner {cls}">{text}</div>', unsafe_allow_html=True)


def provenance(text: str) -> None:
    st.markdown(f'<div class="provenance">{text}</div>', unsafe_allow_html=True)


def altair_theme(chart):
    """Apply the console palette to an Altair chart."""
    return (
        chart.configure_view(strokeWidth=0, fill=PANEL)
        .configure_axis(
            labelColor=MUTED, titleColor=MUTED, gridColor=LINE,
            domainColor=LINE, tickColor=LINE, labelFontSize=11, titleFontSize=11,
        )
        .configure_legend(labelColor=TEXT, titleColor=MUTED, labelFontSize=11, titleFontSize=11)
        .configure_title(color=TEXT, fontSize=13, anchor="start")
    )
