"""
SAR oil-spill segmentation.

Architecture is pinned to U-Net + `mit_b2`, verified against the checkpoint:
encoder.patch_embed{1,2,3}.proj weights carry the (64, 128, 320) channel
progression unique to MiT-B2, and segmentation_head.0 emits 5 classes. The
state dict loads with strict=True; if that ever stops holding, the weights and
this file have diverged and results are not trustworthy.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from .. import weights
from ..config import (
    CLASS_COLORS_RGB,
    CLASS_DESCRIPTIONS,
    CLASS_NAMES,
    DEFAULT_PIXEL_RESOLUTION_M,
    DEVICE,
    NUM_CLASSES,
    SAR_ENCODER,
    SAR_INPUT_SIZE,
    SAR_WEIGHTS,
)

SEA_CLASS = 0
OIL_CLASS = 1
LOOKALIKE_CLASS = 2
SHIP_CLASS = 3
LAND_CLASS = 4


def build_model():
    import segmentation_models_pytorch as smp

    return smp.Unet(
        encoder_name=SAR_ENCODER,
        encoder_weights=None,      # our own trained weights, not ImageNet
        in_channels=3,
        classes=NUM_CLASSES,
    )


def load_model(weights_path=None):
    path = weights.resolve(weights_path or SAR_WEIGHTS)
    model = build_model()
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


# --------------------------------------------------------------------------
# Pre / post processing
# --------------------------------------------------------------------------
def read_image(path_or_bytes) -> np.ndarray:
    """Load an image from a path or raw bytes into RGB uint8."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        buf = np.frombuffer(path_or_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    else:
        bgr = cv2.imread(str(path_or_bytes), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image -- expected a readable JPG/PNG.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def preprocess(image_rgb: np.ndarray) -> torch.Tensor:
    """RGB uint8 HxWx3 -> normalised 1x3x512x512 tensor on the active device."""
    resized = cv2.resize(image_rgb, (SAR_INPUT_SIZE, SAR_INPUT_SIZE))
    arr = resized.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Integer class mask -> viewable RGB image using the dataset palette."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS_RGB.items():
        rgb[mask == cls] = color
    return rgb


def overlay_on_image(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45,
                     classes: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    """
    Blend the class palette over the original scene at its native resolution.

    The mask is upsampled with INTER_NEAREST -- bilinear would invent fractional
    class indices that correspond to no real class.
    """
    h, w = image_rgb.shape[:2]
    full = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    shown = classes if classes is not None else tuple(CLASS_COLORS_RGB)

    out = image_rgb.astype(np.float32).copy()
    for cls in shown:
        if cls == SEA_CLASS:               # never tint open water
            continue
        sel = full == cls
        if not sel.any():
            continue
        color = np.array(CLASS_COLORS_RGB[cls], dtype=np.float32)
        out[sel] = (1 - alpha) * out[sel] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def estimate_area_km2(mask: np.ndarray, cls: int = OIL_CLASS,
                      pixel_resolution_m: float = DEFAULT_PIXEL_RESOLUTION_M) -> float:
    """
    Approximate ground area of one class.

    The mask is a 512x512 resample of the source scene, so resampling error is
    baked in -- always present this as approximate.
    """
    km2_per_pixel = (pixel_resolution_m / 1000.0) ** 2
    return float((mask == cls).sum() * km2_per_pixel)


def _components(mask: np.ndarray, cls: int, top_k: int = 5) -> List[Dict[str, Any]]:
    """Connected-component breakdown of one class, largest first."""
    binary = (mask == cls).astype(np.uint8)
    if binary.sum() == 0:
        return []
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = []
    for i in range(1, count):              # label 0 is background
        cx, cy = centroids[i]
        out.append({
            "pixels": int(stats[i, cv2.CC_STAT_AREA]),
            "centroid_xy": (float(cx), float(cy)),
            "bbox": tuple(int(v) for v in stats[i, :4]),
        })
    out.sort(key=lambda s: s["pixels"], reverse=True)
    return out[:top_k]


@dataclass
class SarResult:
    mask: np.ndarray
    class_pixels: Dict[int, int]
    class_fraction: Dict[int, float]
    oil_area_km2: float
    lookalike_area_km2: float
    slicks: List[Dict[str, Any]] = field(default_factory=list)
    ships: List[Dict[str, Any]] = field(default_factory=list)
    pixel_resolution_m: float = DEFAULT_PIXEL_RESOLUTION_M
    inference_ms: float = 0.0

    @property
    def oil_detected(self) -> bool:
        return self.class_pixels.get(OIL_CLASS, 0) > 0

    @property
    def oil_pixel_fraction(self) -> float:
        return self.class_fraction.get(OIL_CLASS, 0.0)

    def confidence_note(self) -> str:
        """
        Honest framing of how oil and look-alike coverage compare.
        A coverage heuristic for the operator -- not a calibrated probability.
        """
        oil = self.class_pixels.get(OIL_CLASS, 0)
        look = self.class_pixels.get(LOOKALIKE_CLASS, 0)
        if oil == 0:
            return "No oil-spill pixels segmented in this scene."
        if look > oil:
            return ("Look-alike pixels outnumber oil pixels -- treat this scene as "
                    "ambiguous and confirm against wind and met conditions.")
        if oil < 0.001 * self.mask.size:
            return ("Oil signature covers under 0.1% of the scene -- small enough to "
                    "be resampling noise. Review at native resolution.")
        return "Coherent oil-spill signature dominates the dark formations in this scene."

    def summary(self) -> Dict[str, Any]:
        return {
            "oil_detected": self.oil_detected,
            "oil_area_km2": round(self.oil_area_km2, 4),
            "lookalike_area_km2": round(self.lookalike_area_km2, 4),
            "oil_pixel_fraction": round(self.oil_pixel_fraction, 6),
            "slick_count": len(self.slicks),
            "ship_count": len(self.ships),
            "class_pixels": {CLASS_NAMES[c]: n for c, n in self.class_pixels.items()},
            "note": self.confidence_note(),
            "inference_ms": round(self.inference_ms, 1),
        }


def segment(model, image_rgb: np.ndarray,
            pixel_resolution_m: float = DEFAULT_PIXEL_RESOLUTION_M) -> SarResult:
    """Run the segmenter over one RGB scene and summarise the result."""
    tensor = preprocess(image_rgb)
    start = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    total = mask.size
    class_pixels = {c: int((mask == c).sum()) for c in range(NUM_CLASSES)}
    class_fraction = {c: n / total for c, n in class_pixels.items()}

    return SarResult(
        mask=mask,
        class_pixels=class_pixels,
        class_fraction=class_fraction,
        oil_area_km2=estimate_area_km2(mask, OIL_CLASS, pixel_resolution_m),
        lookalike_area_km2=estimate_area_km2(mask, LOOKALIKE_CLASS, pixel_resolution_m),
        slicks=_components(mask, OIL_CLASS),
        ships=_components(mask, SHIP_CLASS, top_k=20),
        pixel_resolution_m=pixel_resolution_m,
        inference_ms=elapsed_ms,
    )


def legend() -> List[Dict[str, Any]]:
    """Palette plus prose for the UI legend."""
    return [
        {
            "index": i,
            "name": CLASS_NAMES[i],
            "rgb": CLASS_COLORS_RGB[i],
            "hex": "#%02x%02x%02x" % CLASS_COLORS_RGB[i],
            "description": CLASS_DESCRIPTIONS[i],
        }
        for i in range(NUM_CLASSES)
    ]
