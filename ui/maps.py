"""
Satellite maps.

Everything geographic renders on Esri World Imagery, so a track reads as a
vessel closing on the Mauritius reef rather than as an abstract lat/lon
scatter. Esri's tile service needs no API key or token.

This does require network access at view time. That is a deliberate trade:
the console is deployed on Streamlit Cloud where connectivity is a given, and
a spill-attribution demo without a coastline underneath it is much harder to
read. `OFFLINE_FALLBACK` swaps every basemap for a plain dark canvas if the
app ever has to run air-gapped.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import folium
import pandas as pd

from poseatsea.config import CLASS_COLORS_RGB
from poseatsea.scenario.wakashio import GROUNDING_LAT, GROUNDING_LON

from . import theme

OFFLINE_FALLBACK = os.getenv("POSEATSEA_OFFLINE_MAPS", "").lower() in {"1", "true", "yes"}

ESRI_IMAGERY = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Imagery/MapServer/tile/{z}/{y}/{x}")
ESRI_LABELS = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
               "Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}")

REEF = (GROUNDING_LAT, GROUNDING_LON)


def base_map(center, zoom: int = 11, height: int = 520) -> folium.Map:
    """A dark satellite canvas centred on `center`."""
    if OFFLINE_FALLBACK:
        fmap = folium.Map(location=center, zoom_start=zoom, tiles=None,
                          prefer_canvas=True, control_scale=True)
        folium.TileLayer(tiles="", attr="offline", name="offline").add_to(fmap)
    else:
        fmap = folium.Map(location=center, zoom_start=zoom, tiles=ESRI_IMAGERY,
                          attr="Esri World Imagery", prefer_canvas=True,
                          control_scale=True)
        # Place names on a separate translucent layer so the imagery stays clean.
        folium.TileLayer(tiles=ESRI_LABELS, attr="Esri", name="Place names",
                         overlay=True, opacity=0.55).add_to(fmap)
    fmap.get_root().html.add_child(folium.Element(
        f"<style>.folium-map{{background:{theme.INK};}}</style>"))
    return fmap


def _fit(fmap: folium.Map, lats, lons, pad: float = 0.01) -> None:
    if not len(lats):
        return
    fmap.fit_bounds([[min(lats) - pad, min(lons) - pad],
                     [max(lats) + pad, max(lons) + pad]])


def add_reef(fmap: folium.Map, label: str = "Pointe d'Esny reef") -> None:
    """The grounding site, with a 5 km hazard ring."""
    folium.Circle(REEF, radius=5000, color=theme.CRITICAL, weight=1.4,
                  fill=True, fill_color=theme.CRITICAL, fill_opacity=0.07,
                  dash_array="6 6", tooltip=f"{label} — 5 km hazard radius").add_to(fmap)
    folium.Marker(
        REEF,
        icon=folium.DivIcon(
            html=f'<div style="width:20px;height:20px;border-radius:50%;'
                 f'border:2.5px solid {theme.CRITICAL};box-shadow:0 0 10px {theme.CRITICAL};'
                 f'background:{theme.CRITICAL}33"></div>',
            icon_size=(20, 20), icon_anchor=(10, 10)),
        tooltip=label,
    ).add_to(fmap)


def _dot(color: str, size: int, glow: bool = False) -> folium.DivIcon:
    shadow = f"box-shadow:0 0 10px {color};" if glow else ""
    return folium.DivIcon(
        html=f'<div style="width:{size}px;height:{size}px;border-radius:50%;'
             f'background:{color};border:2px solid #fff;{shadow}"></div>',
        icon_size=(size, size), icon_anchor=(size // 2, size // 2))


def traffic_map(scored: pd.DataFrame, spill: Optional[Dict[str, float]] = None,
                highlight_mmsi: Optional[int] = None, height: int = 520) -> folium.Map:
    """
    Every vessel track in the AOI, with flagged pings called out.

    The suspect track is drawn heavier and last so it sits above the rest.
    """
    fmap = base_map([scored["latitude"].mean(), scored["longitude"].mean()],
                    zoom=11, height=height)

    ordered = sorted(scored["mmsi"].unique(),
                     key=lambda m: (m == highlight_mmsi))   # highlight drawn last

    for mmsi in ordered:
        track = scored[scored["mmsi"] == mmsi]
        name = track["vessel_name"].iloc[0] if "vessel_name" in track else str(mmsi)
        is_hi = mmsi == highlight_mmsi
        color = theme.VESSEL_COLORS.get(int(mmsi), theme.ACCENT)

        folium.PolyLine(
            list(zip(track["latitude"], track["longitude"])),
            color=color, weight=4 if is_hi else 2,
            opacity=0.95 if is_hi else 0.55,
            tooltip=f"{name} · MMSI {mmsi}",
        ).add_to(fmap)

        last = track.iloc[-1]
        folium.Marker(
            [last["latitude"], last["longitude"]],
            icon=_dot(color, 15 if is_hi else 11, glow=is_hi),
            tooltip=(f"{name}<br>{last['speed']:.1f} kn · {last['course']:.0f}°"),
        ).add_to(fmap)

    # Flagged pings on top of everything.
    if "is_anomaly" in scored.columns:
        for row in scored[scored["is_anomaly"]].itertuples():
            folium.CircleMarker(
                [row.latitude, row.longitude], radius=4,
                color=theme.WARN, fill=True, fill_color=theme.WARN,
                fill_opacity=0.9, weight=1,
                tooltip=(f"{row.vessel_name} — flagged<br>"
                         f"score {row.anomaly_score:.2f} ({row.severity})<br>"
                         f"{row.timestamp:%H:%M} UTC · driver: {row.top_feature}"),
            ).add_to(fmap)

    add_reef(fmap)
    if spill:
        folium.Marker(
            [spill["latitude"], spill["longitude"]],
            icon=folium.DivIcon(
                html=f'<div style="font-size:22px;line-height:22px;color:{theme.CRITICAL};'
                     f'text-shadow:0 0 6px #000">✕</div>',
                icon_size=(22, 22), icon_anchor=(11, 11)),
            tooltip="Observed slick",
        ).add_to(fmap)

    _fit(fmap, list(scored["latitude"]), list(scored["longitude"]))
    return fmap


def trajectory_map(track: pd.DataFrame, history: pd.DataFrame,
                   pred_lat: float, pred_lon: float,
                   actual: Optional[Dict[str, float]] = None,
                   deviation_km: Optional[float] = None,
                   height: int = 520) -> folium.Map:
    """
    Predicted next position against the vessel's real track.

    The full passage is drawn faintly for context, the 8 pings the model
    actually saw are drawn solid, and the prediction/actual pair is circled so
    the deviation is visible at a glance.
    """
    fmap = base_map([history["latitude"].mean(), history["longitude"].mean()], zoom=13)

    folium.PolyLine(list(zip(track["latitude"], track["longitude"])),
                    color="#ffffff", weight=1.2, opacity=0.30,
                    tooltip="Full passage").add_to(fmap)

    folium.PolyLine(list(zip(history["latitude"], history["longitude"])),
                    color=theme.GOOD, weight=3.5, opacity=0.95,
                    tooltip="History the model saw (8 pings)").add_to(fmap)

    for i, row in enumerate(history.itertuples()):
        folium.CircleMarker(
            [row.latitude, row.longitude], radius=4.5,
            color=theme.GOOD, fill=True, fill_color=theme.GOOD,
            fill_opacity=0.9, weight=1,
            tooltip=(f"ping {i + 1}/8 · {row.speed:.1f} kn · {row.course:.0f}°"),
        ).add_to(fmap)

    lats = list(history["latitude"]) + [pred_lat]
    lons = list(history["longitude"]) + [pred_lon]

    if actual:
        folium.PolyLine([[pred_lat, pred_lon],
                         [actual["latitude"], actual["longitude"]]],
                        color=theme.WARN, weight=2, dash_array="5 5",
                        tooltip=(f"Deviation {deviation_km:.3f} km"
                                 if deviation_km is not None else "Deviation")).add_to(fmap)
        folium.Marker([actual["latitude"], actual["longitude"]],
                      icon=_dot(theme.GOOD, 14, glow=True),
                      tooltip="Actual next position").add_to(fmap)
        lats.append(actual["latitude"])
        lons.append(actual["longitude"])

    folium.Marker([pred_lat, pred_lon], icon=_dot(theme.ACCENT, 14, glow=True),
                  tooltip=f"LSTM predicted next position").add_to(fmap)

    # Ring the prediction so it reads immediately at demo zoom.
    folium.Circle([pred_lat, pred_lon],
                  radius=max((deviation_km or 0.3) * 1000 * 1.4, 120),
                  color=theme.CRITICAL, weight=2, fill=False,
                  tooltip="Prediction vs actual").add_to(fmap)

    add_reef(fmap)
    _fit(fmap, lats, lons, pad=0.004)
    return fmap


def deviation_map(track: pd.DataFrame, trace: pd.DataFrame, height: int = 520) -> folium.Map:
    """
    A whole passage coloured by how far each ping fell from its prediction.

    This is the view that shows an inshore deviation developing: the track runs
    green while the vessel behaves predictably and warms as it stops doing so.
    """
    fmap = base_map([track["latitude"].mean(), track["longitude"].mean()], zoom=12)

    folium.PolyLine(list(zip(track["latitude"], track["longitude"])),
                    color="#ffffff", weight=1.2, opacity=0.28).add_to(fmap)

    hi = float(trace["deviation_km"].max()) or 1.0
    for row in trace.itertuples():
        frac = min(row.deviation_km / hi, 1.0)
        color = theme.GOOD if frac < 0.33 else (theme.WARN if frac < 0.66 else theme.CRITICAL)
        folium.CircleMarker(
            [row.actual_lat, row.actual_lon], radius=3 + 4 * frac,
            color=color, fill=True, fill_color=color, fill_opacity=0.85, weight=0.8,
            tooltip=f"ping {row.index} · deviation {row.deviation_km:.3f} km",
        ).add_to(fmap)

    worst = trace.loc[trace["deviation_km"].idxmax()]
    folium.Circle([worst["actual_lat"], worst["actual_lon"]], radius=400,
                  color=theme.CRITICAL, weight=2.5, fill=False,
                  tooltip=f"Worst deviation {worst['deviation_km']:.3f} km").add_to(fmap)

    add_reef(fmap)
    _fit(fmap, list(track["latitude"]), list(track["longitude"]))
    return fmap


def legend(items: List[Dict[str, str]]) -> str:
    """Small HTML legend rendered under a map."""
    cells = "".join(
        f'<span style="margin-right:14px;white-space:nowrap">'
        f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
        f'background:{i["color"]};margin-right:5px;vertical-align:-1px"></span>'
        f'<span style="font-size:.76rem;color:{theme.MUTED}">{i["label"]}</span></span>'
        for i in items
    )
    return f'<div style="padding:.4rem 0 .2rem">{cells}</div>'
