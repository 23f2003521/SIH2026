"""
Synthetic radar scene generator -- a pipeline smoke test, not imagery.

No SAR imagery was supplied with this project. Rather than ship a demo that
cannot be run at all without first sourcing a Sentinel-1 scene, this module
generates a plausible-looking radar frame: gamma-distributed speckle over the
sea, a low-backscatter dark formation standing in for a slick, bright point
targets for hulls, and a textured land mass.

THIS IS NOT SATELLITE DATA. It reproduces the *look* of a SAR frame -- speckle
statistics, dark formations, bright hard targets -- but none of the physics
that the segmenter actually learned from Sentinel-1 VV backscatter. Whatever
the model outputs on these frames says something about the plumbing and nothing
about the model's accuracy. Every surface that renders one must say so.

For real evaluation, use the Krestenitis et al. Sentinel-1 oil-spill benchmark
(1002 train / 110 test scenes) that the model was trained on.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def _speckle(shape: Tuple[int, int], mean: float, looks: float,
             rng: np.random.Generator) -> np.ndarray:
    """
    Multiplicative speckle, the defining texture of a SAR intensity image.

    Multi-look SAR intensity is gamma distributed with shape = number of looks;
    higher looks means smoother imagery.
    """
    return rng.gamma(shape=looks, scale=mean / looks, size=shape)


def _blob(shape: Tuple[int, int], centre: Tuple[float, float], radius: float,
          irregularity: float, rng: np.random.Generator) -> np.ndarray:
    """A soft, irregular closed region -- slicks are never circles."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = centre
    ang = np.arctan2(yy - cy, xx - cx)
    dist = np.hypot(yy - cy, xx - cx)

    wobble = np.zeros_like(ang)
    for k in range(2, 7):
        wobble += rng.normal(0, irregularity / k) * np.sin(k * ang + rng.uniform(0, 2 * np.pi))

    mask = (dist < radius * (1.0 + wobble)).astype(np.float32)
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=radius * 0.10)


def generate_scene(
    size: int = 512,
    seed: int = 7,
    with_slick: bool = True,
    with_lookalike: bool = True,
    n_ships: int = 3,
    with_land: bool = True,
) -> Dict[str, object]:
    """
    Build one synthetic radar frame.

    Returns the RGB image plus the ground-truth geometry used to draw it, so
    the UI can be explicit about what was planted versus what was segmented.
    """
    rng = np.random.default_rng(seed)
    h = w = size
    truth: Dict[str, object] = {"slicks": [], "lookalikes": [], "ships": [], "land": False}

    # Open sea: moderate backscatter, 4-look speckle.
    sea_mean = 78.0
    img = _speckle((h, w), sea_mean, looks=4.0, rng=rng)

    # A large-scale wind-field gradient, so the sea is not uniform.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = 1.0 + 0.18 * np.sin(2 * np.pi * (0.6 * xx / w + 0.25 * yy / h))
    img *= grad

    if with_land:
        # Land in one corner: much brighter, far rougher than the sea.
        land_mask = _blob((h, w), (h * 1.06, w * 0.10), size * 0.42, 0.30, rng)
        land_tex = _speckle((h, w), 168.0, looks=1.4, rng=rng)
        img = img * (1 - land_mask) + land_tex * land_mask
        truth["land"] = True

    if with_slick:
        # Mineral oil damps capillary waves, collapsing backscatter -> dark.
        cy, cx = h * 0.44, w * 0.60
        slick = _blob((h, w), (cy, cx), size * 0.17, 0.42, rng)
        tail = _blob((h, w), (cy - size * 0.13, cx + size * 0.14), size * 0.085, 0.55, rng)
        slick = np.clip(slick + 0.85 * tail, 0, 1)
        img = img * (1 - 0.88 * slick)
        truth["slicks"].append({"centre": (cx, cy), "radius_px": size * 0.17})

    if with_lookalike:
        # A low-wind patch: also dark, but with a softer edge than oil.
        cy, cx = h * 0.76, w * 0.30
        look = _blob((h, w), (cy, cx), size * 0.11, 0.22, rng)
        look = cv2.GaussianBlur(look, (0, 0), sigmaX=size * 0.03)
        img = img * (1 - 0.62 * look)
        truth["lookalikes"].append({"centre": (cx, cy), "radius_px": size * 0.11})

    for _ in range(n_ships):
        sx = rng.uniform(w * 0.35, w * 0.95)
        sy = rng.uniform(h * 0.05, h * 0.90)
        length = rng.uniform(size * 0.012, size * 0.030)
        angle = rng.uniform(0, 180)
        hull = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(hull, (int(sx), int(sy)),
                    (int(length), max(1, int(length * 0.30))),
                    angle, 0, 360, 1.0, -1)
        hull = cv2.GaussianBlur(hull, (0, 0), sigmaX=1.1)
        img += hull * 255.0                      # hard targets saturate
        truth["ships"].append({"centre": (sx, sy), "length_px": length})

    img = np.clip(img, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return {
        "image": rgb,
        "truth": truth,
        "seed": seed,
        "disclaimer": (
            "SYNTHETIC RADAR FRAME -- generated texture, not satellite data. "
            "Segmentation output on this frame demonstrates that the pipeline runs; "
            "it is not evidence of model accuracy."
        ),
    }


PRESETS: List[Dict[str, object]] = [
    {"key": "slick_and_lookalike", "label": "Slick beside a low-wind look-alike",
     "kwargs": {"with_slick": True, "with_lookalike": True, "n_ships": 3, "with_land": True},
     "blurb": "The discrimination case the segmenter exists for."},
    {"key": "clean_sea", "label": "Clean sea, vessels only",
     "kwargs": {"with_slick": False, "with_lookalike": False, "n_ships": 5, "with_land": False},
     "blurb": "A negative control -- nothing here should segment as oil."},
    {"key": "large_slick", "label": "Large slick near the coast",
     "kwargs": {"with_slick": True, "with_lookalike": False, "n_ships": 2, "with_land": True},
     "blurb": "A coastal release, the Wakashio geometry."},
    {"key": "lookalike_only", "label": "Look-alike only, no oil",
     "kwargs": {"with_slick": False, "with_lookalike": True, "n_ships": 2, "with_land": False},
     "blurb": "A false-positive trap: dark, but not oil."},
]


def preset(key: str, seed: int = 7, size: int = 512) -> Dict[str, object]:
    for p in PRESETS:
        if p["key"] == key:
            return generate_scene(size=size, seed=seed, **p["kwargs"])  # type: ignore[arg-type]
    raise KeyError(f"Unknown preset {key!r}")
