"""
The curated SAR scene library.

Five real Sentinel-1 scenes, each with the class mask the shipped segmenter
produced for it, precomputed by `deploy/build_sar_scenes.py` and stored on disk.

These are genuine outputs of the same checkpoint the upload path uses -- not
illustrations, and not hand-drawn. Precomputing buys two things that matter in
a live demo: the page renders instantly, and it never has to load the 110 MB
SegFormer to show a result. Uploading a new scene still runs the model.

Each scene is tied to a vessel from the AIS scenario so the two halves of the
console agree with each other: the casualty carries the major slick, the
runner-up suspect a small discharge, and the trawler that the anomaly detector
over-flags carries a scene that is pure look-alike.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ASSETS = Path(__file__).resolve().parent.parent.parent / "assets" / "sar_samples"
MANIFEST = ASSETS / "manifest.json"


@dataclass
class SarScene:
    key: str
    title: str
    vessel: str
    mmsi: int
    captured: str
    position: List[float]
    blurb: str
    image_path: Path
    mask_path: Path
    source_size: List[int]
    pixel_resolution_m: float
    class_pixels: Dict[str, int]
    oil_area_km2: float
    lookalike_area_km2: float
    oil_detected: bool
    slicks: int
    ships: int
    note: str

    @property
    def label(self) -> str:
        return f"{self.vessel} — {self.title}"

    def load_image(self) -> np.ndarray:
        from ..inference.sar import read_image
        return read_image(self.image_path)

    def load_mask(self) -> np.ndarray:
        """The precomputed class mask: uint8, values 0-4."""
        import cv2
        mask = cv2.imread(str(self.mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {self.mask_path}")
        if mask.ndim == 3:                      # defensive: should be single-channel
            mask = mask[..., 0]
        return mask.astype(np.uint8)

    def verdict(self) -> str:
        """One line an operator can act on."""
        if self.oil_area_km2 >= 1.0:
            return f"Major slick — approximately {self.oil_area_km2:.2f} km² of mineral oil."
        if self.oil_detected and self.oil_area_km2 > 0.05:
            return f"Small slick — approximately {self.oil_area_km2:.2f} km²."
        if self.lookalike_area_km2 > 0.5:
            return ("No oil. Dark formations in this scene are classified "
                    "look-alike — calm water or wave shadow, not a spill.")
        return "No oil-spill signature in this scene."


@lru_cache(maxsize=1)
def load_scenes() -> List[SarScene]:
    if not MANIFEST.exists():
        return []
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    scenes = []
    for e in raw:
        scenes.append(SarScene(
            key=e["key"], title=e["title"], vessel=e["vessel"], mmsi=e["mmsi"],
            captured=e["captured"], position=e["position"], blurb=e["blurb"],
            image_path=ASSETS / e["image"], mask_path=ASSETS / e["mask"],
            source_size=e["source_size"], pixel_resolution_m=e["pixel_resolution_m"],
            class_pixels=e["class_pixels"], oil_area_km2=e["oil_area_km2"],
            lookalike_area_km2=e["lookalike_area_km2"], oil_detected=e["oil_detected"],
            slicks=e["slicks"], ships=e["ships"], note=e["note"],
        ))
    return scenes


def by_key(key: str) -> Optional[SarScene]:
    return next((s for s in load_scenes() if s.key == key), None)


def by_mmsi(mmsi: int) -> Optional[SarScene]:
    return next((s for s in load_scenes() if s.mmsi == int(mmsi)), None)


def available() -> bool:
    return bool(load_scenes())


def summary_rows() -> List[Dict[str, Any]]:
    """Compact table of the whole library."""
    return [{
        "Vessel": s.vessel,
        "Scene": s.title,
        "Captured": s.captured,
        "Oil (km²)": s.oil_area_km2,
        "Look-alike (km²)": s.lookalike_area_km2,
        "Slicks": s.slicks,
        "Verdict": "OIL DETECTED" if s.oil_detected and s.oil_area_km2 > 0.05 else "No oil",
    } for s in load_scenes()]
