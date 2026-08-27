"""SAR segmentation page."""
from __future__ import annotations

import io

import numpy as np
import streamlit as st
from PIL import Image

from poseatsea.config import CLASS_NAMES, SAR_INPUT_SIZE
from poseatsea.inference import sar as sar_mod
from poseatsea.scenario import synthetic_sar

from .. import charts, engine, theme


def _to_png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def render() -> None:
    st.markdown("## SAR oil-spill segmentation")
    st.markdown(
        f"<span style='color:{theme.MUTED}'>U-Net with a MiT-B2 SegFormer encoder, "
        f"trained on the Krestenitis Sentinel-1 benchmark. Every pixel is assigned to "
        f"one of five classes.</span>",
        unsafe_allow_html=True,
    )

    source = st.radio(
        "Scene source",
        ["Upload a SAR scene", "Pipeline self-test (synthetic)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    image_rgb = None
    is_synthetic = False

    if source == "Upload a SAR scene":
        theme.banner(
            "Upload a Sentinel-1 scene as JPG or PNG. Real evaluation imagery comes from "
            "the <b>Krestenitis et al. oil-spill benchmark</b> (1002 train / 110 test scenes) "
            "that these weights were trained on &mdash; any file from its <span class='mono'>"
            "test/images/</span> folder is a valid input.",
        )
        upload = st.file_uploader("SAR scene", type=["jpg", "jpeg", "png", "tif", "tiff"],
                                  label_visibility="collapsed")
        if upload is not None:
            try:
                image_rgb = sar_mod.read_image(upload.getvalue())
            except ValueError as exc:
                st.error(str(exc))
                return
    else:
        theme.banner(
            "<b>This is not satellite data.</b> These frames reproduce the look of SAR "
            "&mdash; gamma speckle, dark formations, bright hard targets &mdash; but none of "
            "the backscatter physics the model actually learned. They confirm the pipeline "
            "executes end to end. <b>Class output on synthetic texture is not meaningful and "
            "must not be read as accuracy.</b> Supply real Sentinel-1 imagery for a genuine result.",
            "bad",
        )
        c1, c2 = st.columns([2, 1])
        with c1:
            labels = {p["label"]: p for p in synthetic_sar.PRESETS}
            chosen = st.selectbox("Scene", list(labels))
            preset = labels[chosen]
            st.caption(preset["blurb"])
        with c2:
            seed = st.number_input("Seed", 0, 9999, 7, help="Regenerate with different speckle.")
        scene = synthetic_sar.preset(preset["key"], seed=int(seed))
        image_rgb = scene["image"]
        is_synthetic = True

    if image_rgb is None:
        st.info("Choose a scene to run the segmenter.")
        _legend()
        return

    # ------------------------------------------------------------------ controls
    with st.expander("Analysis settings"):
        col1, col2 = st.columns(2)
        with col1:
            resolution = st.slider(
                "Ground resolution (m/pixel)", 1.0, 40.0, 10.0, 1.0,
                help="Sentinel-1 GRD is 10 m in the reference dataset. Area scales with "
                     "the square of this value.",
            )
        with col2:
            alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.45, 0.05)

    result = engine.segment_image(image_rgb, resolution)

    # ------------------------------------------------------------------ headline
    oil_px = result.class_pixels.get(sar_mod.OIL_CLASS, 0)
    look_px = result.class_pixels.get(sar_mod.LOOKALIKE_CLASS, 0)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(theme.metric_card(
            "Oil detected", "Yes" if result.oil_detected else "No",
            f"{oil_px:,} px of {result.mask.size:,}",
            theme.BAD if result.oil_detected else theme.GOOD), unsafe_allow_html=True)
    with c2:
        st.markdown(theme.metric_card(
            "Approx. area", f"~{result.oil_area_km2:.2f} km²",
            f"at {resolution:.0f} m/px — approximate"), unsafe_allow_html=True)
    with c3:
        st.markdown(theme.metric_card(
            "Look-alike", f"{look_px:,} px",
            "dark but not oil", theme.WARN if look_px else theme.MUTED), unsafe_allow_html=True)
    with c4:
        st.markdown(theme.metric_card(
            "Distinct slicks", str(len(result.slicks)),
            f"inference {result.inference_ms:.0f} ms"), unsafe_allow_html=True)

    if is_synthetic:
        theme.banner(
            "Figures above describe a <b>synthetic frame</b>. They demonstrate that the "
            "segmenter loaded and ran &mdash; nothing more.", "bad",
        )
    else:
        tone = "warn" if look_px > oil_px else ("bad" if result.oil_detected else "good")
        theme.banner(f"<b>Assessment.</b> {result.confidence_note()}", tone)

    # ------------------------------------------------------------------ imagery
    mask_rgb = sar_mod.mask_to_rgb(result.mask)
    overlay = sar_mod.overlay_on_image(image_rgb, result.mask, alpha=alpha)

    t1, t2, t3 = st.tabs(["Overlay", "Side by side", "Class mask"])
    with t1:
        st.image(overlay, use_container_width=True,
                 caption="Segmentation blended over the source scene at native resolution "
                         "(nearest-neighbour upsampling).")
    with t2:
        a, b = st.columns(2)
        with a:
            st.image(image_rgb, use_container_width=True, caption="Source scene")
        with b:
            st.image(mask_rgb, use_container_width=True, caption=f"Predicted classes ({SAR_INPUT_SIZE}×{SAR_INPUT_SIZE})")
    with t3:
        st.image(mask_rgb, use_container_width=True,
                 caption="Raw class mask in the dataset palette")

    # ------------------------------------------------------------------ breakdown
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("##### Class distribution")
        chart = charts.class_distribution(result.summary()["class_pixels"])
        if chart is not None:
            st.altair_chart(chart, use_container_width=True)
    with right:
        st.markdown("##### Detected slicks")
        if result.slicks:
            rows = []
            for i, s in enumerate(result.slicks, 1):
                km2 = s["pixels"] * (resolution / 1000.0) ** 2
                rows.append({
                    "#": i,
                    "Pixels": s["pixels"],
                    "~km²": round(km2, 3),
                    "Centroid (x, y)": f"{s['centroid_xy'][0]:.0f}, {s['centroid_xy'][1]:.0f}",
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.caption("No oil-class regions segmented in this scene.")

    _legend()

    st.download_button(
        "Download class mask (PNG)",
        data=_to_png_bytes(mask_rgb),
        file_name="segmentation_mask.png",
        mime="image/png",
    )

    theme.provenance(
        f"Area figures are approximate: the mask is a {SAR_INPUT_SIZE}×{SAR_INPUT_SIZE} resample of the source scene, "
        "so resampling error is carried into every pixel count."
    )


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
