"""
Charts for the console.

Everything renders through Altair, which Streamlit bundles locally -- the app
draws its geography without contacting a tile server, so it works in an
air-gapped demo room. That means no basemap: these are plan views in plain
lat/lon, with the AOI box and the reef marked for orientation.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import altair as alt
import pandas as pd

from poseatsea.config import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN

from . import theme


def _aoi_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "longitude": [LON_MIN, LON_MAX, LON_MAX, LON_MIN, LON_MIN],
        "latitude": [LAT_MIN, LAT_MIN, LAT_MAX, LAT_MAX, LAT_MIN],
    })


def _latlon_scales(df: pd.DataFrame, pad: float = 0.02):
    lat_lo, lat_hi = df["latitude"].min() - pad, df["latitude"].max() + pad
    lon_lo, lon_hi = df["longitude"].min() - pad, df["longitude"].max() + pad
    return (
        alt.Scale(domain=[lon_lo, lon_hi], nice=False),
        alt.Scale(domain=[lat_lo, lat_hi], nice=False),
    )


def traffic_plan_view(
    ais: pd.DataFrame,
    spill: Optional[Dict[str, float]] = None,
    highlight_mmsi: Optional[int] = None,
    show_flags: bool = True,
    height: int = 460,
):
    """
    All vessel tracks in the AOI, with anomalous pings called out.

    Longitude is not rescaled by cos(latitude): over a 0.5-degree AOI the
    distortion is a few percent, and honest axis labels beat a projection the
    viewer cannot interrogate.
    """
    df = ais.copy()
    if "vessel_name" not in df.columns:
        df["vessel_name"] = df["mmsi"].astype(str)

    lon_scale, lat_scale = _latlon_scales(df)
    domain = sorted(df["vessel_name"].unique().tolist())
    colors = [theme.VESSEL_COLORS.get(
        int(df.loc[df["vessel_name"] == n, "mmsi"].iloc[0]), theme.ACCENT) for n in domain]

    aoi = alt.Chart(_aoi_frame()).mark_line(
        color=theme.LINE, strokeDash=[5, 4], strokeWidth=1,
    ).encode(x=alt.X("longitude:Q", scale=lon_scale), y=alt.Y("latitude:Q", scale=lat_scale))

    tracks = alt.Chart(df).mark_line(strokeWidth=1.9, opacity=0.85).encode(
        x=alt.X("longitude:Q", scale=lon_scale, title="Longitude (deg E)"),
        y=alt.Y("latitude:Q", scale=lat_scale, title="Latitude (deg N)"),
        color=alt.Color("vessel_name:N", title="Vessel",
                        scale=alt.Scale(domain=domain, range=colors)),
        detail="mmsi:N",
        opacity=alt.value(0.45) if highlight_mmsi else alt.value(0.85),
        tooltip=["vessel_name:N", "mmsi:N"],
    )

    layers = [aoi, tracks]

    if highlight_mmsi is not None:
        hi = df[df["mmsi"] == highlight_mmsi]
        if len(hi):
            layers.append(
                alt.Chart(hi).mark_line(strokeWidth=3.2, color=theme.CRITICAL).encode(
                    x=alt.X("longitude:Q", scale=lon_scale),
                    y=alt.Y("latitude:Q", scale=lat_scale),
                )
            )

    if show_flags and "is_anomaly" in df.columns:
        flags = df[df["is_anomaly"]]
        if len(flags):
            layers.append(
                alt.Chart(flags).mark_point(
                    shape="diamond", size=44, filled=True,
                    color=theme.WARN, opacity=0.9,
                ).encode(
                    x=alt.X("longitude:Q", scale=lon_scale),
                    y=alt.Y("latitude:Q", scale=lat_scale),
                    tooltip=["vessel_name:N", "timestamp:T",
                             alt.Tooltip("anomaly_score:Q", format=".2f"), "severity:N"],
                )
            )

    if spill:
        sp = pd.DataFrame([spill])
        layers.append(
            alt.Chart(sp).mark_point(
                shape="cross", size=280, color=theme.CRITICAL,
                strokeWidth=3, filled=False,
            ).encode(
                x=alt.X("longitude:Q", scale=lon_scale),
                y=alt.Y("latitude:Q", scale=lat_scale),
                tooltip=alt.value("Observed slick / grounding position"),
            )
        )

    chart = alt.layer(*layers).properties(height=height)
    return theme.altair_theme(chart)


def anomaly_timeline(track: pd.DataFrame, threshold: float, height: int = 230):
    """Reconstruction error against time for one vessel, with the decision line."""
    df = track.copy()
    df["flagged"] = df["is_anomaly"].map({True: "Flagged", False: "Normal"})

    base = alt.Chart(df)
    area = base.mark_area(opacity=0.20, color=theme.ACCENT).encode(
        x=alt.X("timestamp:T", title=None),
        y=alt.Y("anomaly_score:Q", title="Reconstruction error",
                scale=alt.Scale(type="symlog")),
    )
    line = base.mark_line(strokeWidth=1.6, color=theme.ACCENT).encode(
        x="timestamp:T", y=alt.Y("anomaly_score:Q", scale=alt.Scale(type="symlog")),
    )
    pts = base.mark_point(size=42, filled=True).encode(
        x="timestamp:T",
        y=alt.Y("anomaly_score:Q", scale=alt.Scale(type="symlog")),
        color=alt.Color("flagged:N", title=None,
                        scale=alt.Scale(domain=["Normal", "Flagged"],
                                        range=[theme.MUTED, theme.CRITICAL])),
        tooltip=["timestamp:T", alt.Tooltip("anomaly_score:Q", format=".3f"),
                 "severity:N", "top_feature:N", "phase:N"],
    )
    rule = alt.Chart(pd.DataFrame({"y": [threshold]})).mark_rule(
        color=theme.WARN, strokeDash=[6, 4], strokeWidth=1.4,
    ).encode(y=alt.Y("y:Q", scale=alt.Scale(type="symlog")))

    return theme.altair_theme(
        alt.layer(area, line, pts, rule).properties(height=height)
    )


def deviation_timeline(trace: pd.DataFrame, p90: float, height: int = 230):
    """Predicted-vs-actual deviation over a transit."""
    base = alt.Chart(trace)
    line = base.mark_line(strokeWidth=1.8, color=theme.ACCENT).encode(
        x=alt.X("index:Q", title="Ping number"),
        y=alt.Y("deviation_km:Q", title="Deviation from prediction (km)"),
        tooltip=["index:Q", alt.Tooltip("deviation_km:Q", format=".3f"), "confidence:N"],
    )
    band = alt.Chart(pd.DataFrame({"y": [p90]})).mark_rule(
        color=theme.WARN, strokeDash=[6, 4], strokeWidth=1.3,
    ).encode(y="y:Q")
    return theme.altair_theme(alt.layer(line, band).properties(height=height))


def prediction_detail(history: pd.DataFrame, pred_lat: float, pred_lon: float,
                      actual: Optional[Dict[str, float]] = None, height: int = 380):
    """Zoomed view of one 8-ping window, its prediction, and the truth."""
    hist = history.copy()
    hist["kind"] = "History (8 pings)"

    pts = [{"latitude": pred_lat, "longitude": pred_lon, "kind": "Predicted next"}]
    if actual:
        pts.append({"latitude": actual["latitude"], "longitude": actual["longitude"],
                    "kind": "Actual next"})
    marks = pd.DataFrame(pts)

    combined = pd.concat([hist[["latitude", "longitude"]], marks[["latitude", "longitude"]]])
    lon_scale, lat_scale = _latlon_scales(combined, pad=0.004)

    track = alt.Chart(hist).mark_line(
        strokeWidth=2.2, color=theme.MUTED, point=alt.OverlayMarkDef(color=theme.MUTED, size=48),
    ).encode(
        x=alt.X("longitude:Q", scale=lon_scale, title="Longitude (deg E)"),
        y=alt.Y("latitude:Q", scale=lat_scale, title="Latitude (deg N)"),
        tooltip=["timestamp:T", alt.Tooltip("speed:Q", format=".1f"),
                 alt.Tooltip("course:Q", format=".0f")],
    )

    domain = ["Predicted next", "Actual next"]
    marker = alt.Chart(marks).mark_point(size=260, filled=True, opacity=0.95).encode(
        x=alt.X("longitude:Q", scale=lon_scale),
        y=alt.Y("latitude:Q", scale=lat_scale),
        color=alt.Color("kind:N", title=None,
                        scale=alt.Scale(domain=domain, range=[theme.ACCENT, theme.GOOD])),
        shape=alt.Shape("kind:N", title=None,
                        scale=alt.Scale(domain=domain, range=["triangle", "circle"])),
        tooltip=["kind:N", alt.Tooltip("latitude:Q", format=".5f"),
                 alt.Tooltip("longitude:Q", format=".5f")],
    )

    layers = [track, marker]
    if actual:
        link = pd.DataFrame([
            {"latitude": pred_lat, "longitude": pred_lon, "g": 1},
            {"latitude": actual["latitude"], "longitude": actual["longitude"], "g": 1},
        ])
        layers.insert(1, alt.Chart(link).mark_line(
            strokeWidth=1.4, strokeDash=[4, 4], color=theme.WARN,
        ).encode(
            x=alt.X("longitude:Q", scale=lon_scale),
            y=alt.Y("latitude:Q", scale=lat_scale), detail="g:N",
        ))

    return theme.altair_theme(alt.layer(*layers).properties(height=height))


def class_distribution(class_pixels: Dict[str, int], height: int = 200):
    """Segmented class breakdown, in the dataset's own palette."""
    from poseatsea.config import CLASS_COLORS_RGB, CLASS_NAMES

    rows = [{"cls": k, "pixels": v} for k, v in class_pixels.items() if v > 0]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    order = [c for c in CLASS_NAMES if c in df["cls"].values]
    palette = ["#%02x%02x%02x" % CLASS_COLORS_RGB[CLASS_NAMES.index(c)] for c in order]
    # Sea surface is pure black in the dataset palette; nudge it so the bar is visible.
    palette = [p if p != "#000000" else "#2a3548" for p in palette]

    chart = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=18).encode(
        x=alt.X("pixels:Q", title="Pixels (256 x 256 mask)"),
        y=alt.Y("cls:N", title=None, sort=order),
        color=alt.Color("cls:N", scale=alt.Scale(domain=order, range=palette), legend=None),
        tooltip=["cls:N", "pixels:Q"],
    ).properties(height=height)
    return theme.altair_theme(chart)


def attribution_bars(results: List[Dict], height: int = 260):
    """Stacked evidence contributions behind each vessel's attribution score."""
    from poseatsea.fusion import WEIGHTS

    rows = []
    for r in results:
        for comp, val in r["components"].items():
            rows.append({
                "vessel": r["vessel_name"],
                "component": comp.capitalize(),
                "contribution": val * WEIGHTS[comp],
                "total": r["score"],
            })
    df = pd.DataFrame(rows)
    order = [r["vessel_name"] for r in results]
    comp_order = ["Proximity", "Anomaly", "Dwell", "Deviation"]
    comp_colors = [theme.CRITICAL, theme.WARN, theme.ACCENT, "#b197fc"]

    chart = alt.Chart(df).mark_bar(height=22).encode(
        x=alt.X("contribution:Q", title="Weighted contribution to attribution score",
                stack="zero"),
        y=alt.Y("vessel:N", title=None, sort=order),
        color=alt.Color("component:N", title="Evidence",
                        scale=alt.Scale(domain=comp_order, range=comp_colors)),
        tooltip=["vessel:N", "component:N", alt.Tooltip("contribution:Q", format=".3f"),
                 alt.Tooltip("total:Q", format=".3f")],
    ).properties(height=height)
    return theme.altair_theme(chart)
