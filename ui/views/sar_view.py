"""SAR segmentation page."""
from __future__ import annotations

import io

import numpy as np
import streamlit as st
from PIL import Image

from datetime import datetime, timezone

import pandas as pd
from streamlit_folium import st_folium

from poseatsea.config import CLASS_NAMES
from poseatsea.inference import sar as sar_mod
from poseatsea.scenario import sar_scenes

from .. import charts, engine, maps, theme



def _to_png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def render() -> None:
    st.markdown("## SAR oil spill segmenter")

    has_library = sar_scenes.available()
    modes = (["Scene library", "Upload a scene"] if has_library else ["Upload a scene"])
    source = st.radio("Scene source", modes, horizontal=True, label_visibility="collapsed")

    if source == "Scene library":
        _library_view()
    else:
        _upload_view()


# --------------------------------------------------------------------------
# Curated library -- precomputed, instant, no model load
# --------------------------------------------------------------------------
def _library_view() -> None:
    scenes = sar_scenes.load_scenes()
    labels = {s.label: s for s in scenes}

    c1, c2, c3 = st.columns([2.0, 1.0, 1.2])
    with c1:
        chosen = st.selectbox("Scene", list(labels), label_visibility="collapsed")
    scene = labels[chosen]
    with c2:
        alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.45, 0.05)
    with c3:
        run_live = st.toggle(
            "⚡ Run live model",
            value=False,
            help="Run the real 110 MB PyTorch SegFormer model live on this scene rather than loading precomputed masks.",
        )

    image = scene.load_image()

    if run_live:
        with st.spinner("Running SegFormer live inference..."):
            result = engine.segment_image(image, scene.pixel_resolution_m)
        mask = result.mask
        class_pixels = {CLASS_NAMES[c]: n for c, n in result.class_pixels.items()}
        oil_px = result.class_pixels.get(sar_mod.OIL_CLASS, 0)
        look_px = result.class_pixels.get(sar_mod.LOOKALIKE_CLASS, 0)
        oil_area = result.oil_area_km2
        slicks = len(result.slicks)
        footer = f"⚡ live inference {result.inference_ms:.0f} ms"
        slicks_list = result.slicks
        status_note = f"<b>Live SegFormer output:</b> {result.confidence_note()}"
        tone = "warn" if look_px > oil_px else ("bad" if result.oil_detected else "good")
    else:
        mask = scene.load_mask()
        class_pixels = scene.class_pixels
        oil_px = scene.class_pixels.get("Oil Spill", 0)
        look_px = scene.class_pixels.get("Look-alike", 0)
        oil_area = scene.oil_area_km2
        slicks = scene.slicks
        footer = f"{scene.captured} &middot; precomputed"
        slicks_list = None
        status_note = f"<b>{scene.verdict()}</b> {scene.note}"
        tone = "bad" if scene.oil_detected and scene.oil_area_km2 > 0.05 else (
            "warn" if scene.lookalike_area_km2 > 0.5 else "good")

    _headline(
        oil_px=oil_px,
        look_px=look_px,
        total_px=mask.size,
        oil_area=oil_area,
        resolution=scene.pixel_resolution_m,
        slicks=slicks,
        footer=footer,
    )

    theme.banner(status_note, tone)

    _vessel_crossref(scene)

    _imagery(image, mask, alpha, class_pixels, scene.pixel_resolution_m, slicks_list)
    _legend()
    _dark_vessels_section(scene, mask)



def _vessel_crossref(scene) -> None:
    """
    What the AIS models say about this same vessel.

    The SAR result on its own cannot attribute anything. Putting the vessel's
    own anomaly score and distance-to-slick beside the imagery is what turns
    two detections into one finding — or, for five of these six vessels, into
    a clearance.
    """
    scored = engine.scored_ais()
    track = scored[scored["mmsi"] == scene.mmsi]

    header = (f"<b>{scene.vessel}</b> "
              f"<span class='mono' style='color:{theme.MUTED}'>MMSI {scene.mmsi}</span>")

    if track.empty:
        st.markdown(
            f"<div class='pos-card tight'><span class='pos-label'>Vessel</span>"
            f"<div style='margin-top:.2rem'>{header}</div>"
            f"<div class='pos-sub'>No AIS in the analysis window for this vessel.</div></div>",
            unsafe_allow_html=True)
        return

    flagged = int(track["is_anomaly"].sum())
    peak = float(track["anomaly_score"].max())
    flag_state = track["flag"].iloc[0]
    vtype = track["vessel_type"].iloc[0]

    sc = engine.scenario()
    spill_lat, spill_lon = sc["spill_position"]
    results = {r["mmsi"]: r for r in engine.attribution(spill_lat, spill_lon, 15.0, 6.0)}
    att = results.get(scene.mmsi)

    if att and att["band"] == "primary suspect":
        tone, verdict = theme.CRITICAL, "ATTRIBUTED — PRIMARY SUSPECT"
    elif att:
        tone, verdict = theme.GOOD, f"CLEARED — {att['band'].upper()}"
    else:
        tone, verdict = theme.MUTED, "NOT RANKED"

    dist = f"{att['min_distance_km']:.1f} km from the observed slick" if att else "—"

    st.markdown(
        f"""
<div class="pos-card">
  <span class="pos-label">Cross-referenced against this vessel's AIS</span>
  <div style="display:flex;gap:.75rem;align-items:center;flex-wrap:wrap;margin:.3rem 0 .6rem">
    {header}
    {theme.pill(verdict, tone)}
    <span class="mono" style="color:{theme.MUTED}">{flag_state} &middot; {vtype}</span>
  </div>
  <div style="display:flex;gap:1.8rem;flex-wrap:wrap;font-size:.82rem">
    <span><b>{len(track)}</b> <span style="color:{theme.MUTED}">AIS pings</span></span>
    <span><b style="color:{theme.CRITICAL if flagged else theme.GOOD}">{flagged}</b>
      <span style="color:{theme.MUTED}">flagged for review</span></span>
    <span><b>{peak:.2f}</b> <span style="color:{theme.MUTED}">peak reconstruction error</span></span>
    <span><b>{dist}</b></span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _dark_vessels_section(scene, mask: np.ndarray) -> None:
    from poseatsea import dark_vessels

    st.markdown("")
    st.markdown("##### 🛰️ Non-Cooperative / Dark Vessel Detection")
    st.markdown(
        f"<div class='pos-sub' style='margin-bottom:10px'>"
        f"Cross-referencing radar vessel contacts against active Class A/B AIS transmissions. "
        f"Point targets detected by the radar segmenter with no active AIS broadcast within "
        f"the satellite overpass window are flagged as non-cooperative dark vessels.</div>",
        unsafe_allow_html=True,
    )

    ais_df = engine.scored_ais()
    sc = engine.scenario()
    spill_lat, spill_lon = sc["spill_position"]

    # Extract radar targets from mask
    targets = dark_vessels.extract_radar_targets_from_mask(
        mask,
        center_lat=scene.position[0],
        center_lon=scene.position[1],
        pixel_res_m=scene.pixel_resolution_m,
    )

    # Match against AIS
    try:
        acq_time = datetime.strptime(
            scene.captured.replace(" UTC", ""), "%Y-%m-%d %H:%M"
        ).replace(tzinfo=timezone.utc)
    except Exception:
        acq_time = None

    matched_targets = dark_vessels.match_sar_ships_to_ais(
        targets,
        ais_df,
        acquisition_time=acq_time,
        spill_lat=spill_lat,
        spill_lon=spill_lon,
        gating_radius_m=2000.0,
    )

    dark_count = sum(1 for t in matched_targets if not t.is_matched)
    coop_count = sum(1 for t in matched_targets if t.is_matched)
    near_slick_count = sum(
        1 for t in matched_targets if t.risk_level == "dark_vessel_near_slick"
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            theme.metric_card("Radar Ship Targets", str(len(matched_targets)), "detected in scene"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            theme.metric_card("AIS Cooperative", str(coop_count), "matched to transponder", theme.GOOD if coop_count else theme.MUTED),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            theme.metric_card("Dark Vessels (No AIS)", str(dark_count), "unmatched radar return", theme.WARN if dark_count else theme.GOOD),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            theme.metric_card("Dark Vessels Near Slick", str(near_slick_count), "< 15 km to slick", theme.CRITICAL if near_slick_count else theme.GOOD),
            unsafe_allow_html=True,
        )

    st.markdown("")
    mcol, tcol = st.columns([1.35, 1.0])
    with mcol:
        st.markdown("<b>Radar Discrepancy Footprint</b>", unsafe_allow_html=True)
        st_folium(
            maps.sar_dark_vessels_map(
                scene_center=tuple(scene.position),
                targets=[t.as_dict() for t in matched_targets],
                scene_size_px=tuple(scene.source_size),
                pixel_res_m=scene.pixel_resolution_m,
                spill_pos=(spill_lat, spill_lon),
                height=400,
            ),
            use_container_width=True,
            height=400,
            returned_objects=[],
            key=f"dark_map_{scene.key}",
        )
        st.markdown(
            maps.legend([
                {"color": theme.GOOD, "label": "Cooperative (AIS Verified)"},
                {"color": theme.CRITICAL, "label": "Dark Vessel (< 15 km to slick)"},
                {"color": theme.WARN, "label": "Dark Vessel (Outer AOI)"},
                {"color": theme.ACCENT, "label": "SAR Footprint"},
            ]),
            unsafe_allow_html=True,
        )

    with tcol:
        st.markdown("<b>Detected Radar Target Registry</b>", unsafe_allow_html=True)
        if matched_targets:
            rows = []
            for t in matched_targets:
                rows.append({
                    "Target": f"RDR-{t.target_id:02d}",
                    "Status": "COOPERATIVE" if t.is_matched else "DARK VESSEL",
                    "Est LOA": f"{t.estimated_length_m:.0f} m",
                    "To Slick": f"{t.distance_to_spill_km:.1f} km" if t.distance_to_spill_km is not None else "--",
                    "AIS Match": t.matched_vessel_name or "Silent (No AIS)",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=360)
        else:
            st.info("No radar ship returns detected in this scene.")



# --------------------------------------------------------------------------
# Live inference on an uploaded scene
# --------------------------------------------------------------------------
def _upload_view() -> None:
    theme.banner(
        "Upload a Sentinel-1 scene as JPG or PNG and the segmenter runs on it live. "
        "Evaluation imagery comes from the <b>Krestenitis et al. oil-spill benchmark</b> "
        "(1002 train / 110 test scenes) that these weights were trained on — any file "
        "from its <span class='mono'>test/images/</span> folder is a valid input. "
        "First run loads the 110 MB model and takes a few seconds.",
    )
    upload = st.file_uploader("SAR scene", type=["jpg", "jpeg", "png", "tif", "tiff"],
                              label_visibility="collapsed")
    if upload is None:
        _legend()
        return

    try:
        image = sar_mod.read_image(upload.getvalue())
    except ValueError as exc:
        st.error(str(exc))
        return

    c1, c2 = st.columns(2)
    with c1:
        resolution = st.slider(
            "Ground resolution (m/pixel)", 1.0, 40.0, 10.0, 1.0,
            help="Sentinel-1 GRD is 10 m in the reference dataset. Area scales with "
                 "the square of this value.")
    with c2:
        alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.45, 0.05)

    result = engine.segment_image(image, resolution)
    named = {CLASS_NAMES[c]: n for c, n in result.class_pixels.items()}

    _headline(
        oil_px=result.class_pixels.get(sar_mod.OIL_CLASS, 0),
        look_px=result.class_pixels.get(sar_mod.LOOKALIKE_CLASS, 0),
        total_px=result.mask.size,
        oil_area=result.oil_area_km2,
        resolution=resolution,
        slicks=len(result.slicks),
        footer=f"inference {result.inference_ms:.0f} ms",
    )

    oil_px = result.class_pixels.get(sar_mod.OIL_CLASS, 0)
    look_px = result.class_pixels.get(sar_mod.LOOKALIKE_CLASS, 0)
    tone = "warn" if look_px > oil_px else ("bad" if result.oil_detected else "good")
    theme.banner(f"<b>Assessment.</b> {result.confidence_note()}", tone)

    _imagery(image, result.mask, alpha, named, resolution, result.slicks)
    _legend()


# --------------------------------------------------------------------------
# Shared rendering
# --------------------------------------------------------------------------
def _headline(oil_px, look_px, total_px, oil_area, resolution, slicks, footer) -> None:
    c1, c2, c3, c4 = st.columns(4)
    detected = oil_px > 0 and oil_area > 0.05
    with c1:
        st.markdown(theme.metric_card(
            "Oil detected", "Yes" if detected else "No",
            f"{oil_px:,} px of {total_px:,}",
            theme.BAD if detected else theme.GOOD), unsafe_allow_html=True)
    with c2:
        st.markdown(theme.metric_card(
            "Approx. area", f"~{oil_area:.2f} km²",
            f"at {resolution:.0f} m/px — approximate"), unsafe_allow_html=True)
    with c3:
        st.markdown(theme.metric_card(
            "Look-alike", f"{look_px:,} px", "dark but not oil",
            theme.WARN if look_px else theme.MUTED), unsafe_allow_html=True)
    with c4:
        st.markdown(theme.metric_card(
            "Distinct slicks", str(slicks), footer), unsafe_allow_html=True)


def _imagery(image, mask, alpha, class_pixels, resolution, slicks=None) -> None:
    mask_rgb = sar_mod.mask_to_rgb(mask)
    overlay = sar_mod.overlay_on_image(image, mask, alpha=alpha)

    t1, t2, t3 = st.tabs(["Overlay", "Side by side", "Class mask"])
    with t1:
        st.image(overlay, use_container_width=True,
                 caption="Segmentation blended over the source scene at native "
                         "resolution (nearest-neighbour upsampling).")
    with t2:
        a, b = st.columns(2)
        with a:
            st.image(image, use_container_width=True, caption="Source SAR scene")
        with b:
            st.image(mask_rgb, use_container_width=True,
                     caption="Predicted classes, resampled to the source resolution")
    with t3:
        st.image(mask_rgb, use_container_width=True,
                 caption="Raw class mask in the dataset palette")

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("##### Class distribution")
        chart = charts.class_distribution(class_pixels)
        if chart is not None:
            st.altair_chart(chart, use_container_width=True)
    with right:
        st.markdown("##### Detected slicks")
        if slicks:
            rows = [{
                "#": i,
                "Pixels": s["pixels"],
                "~km²": round(s["pixels"] * (resolution / 1000.0) ** 2, 3),
                "Centroid (x, y)": f"{s['centroid_xy'][0]:.0f}, {s['centroid_xy'][1]:.0f}",
            } for i, s in enumerate(slicks, 1)]
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            counts = {k: v for k, v in class_pixels.items() if v > 0}
            st.dataframe(
                [{"Class": k, "Pixels": v,
                  "Share": f"{100 * v / max(sum(class_pixels.values()), 1):.2f}%"}
                 for k, v in counts.items()],
                use_container_width=True, hide_index=True)

    st.download_button("Download class mask (PNG)", data=_to_png_bytes(mask_rgb),
                       file_name="segmentation_mask.png", mime="image/png")


def _legend() -> None:
    st.markdown("##### Class legend")
    cols = st.columns(5)
    for col, item in zip(cols, sar_mod.legend()):
        swatch = item["hex"] if item["hex"] != "#000000" else "#2a3548"
        with col:
            st.markdown(
                f"<div class='pos-card tight'>"
                f"<span class='swatch' style='background:{swatch}'></span>"
                f"<b style='font-size:.85rem'>{item['name']}</b>"
                f"<div class='pos-sub'>{item['description']}</div></div>",
                unsafe_allow_html=True,
            )
