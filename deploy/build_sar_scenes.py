#!/usr/bin/env python
"""
Precompute the curated SAR scene library.

    python deploy/build_sar_scenes.py

Runs the real segmenter once over each source scene and writes the resulting
class mask to disk as a lossless indexed PNG, alongside a manifest of the
derived statistics.

Why precompute at all, when the model works? Because the console then renders
the SAR page instantly and deterministically, without loading a 110 MB / 342 MB
SegFormer or spending two seconds per scene. These are genuine outputs of the
shipped checkpoint, not illustrations -- the page says "precomputed", and the
upload path still runs live inference for anything new.

Re-run this whenever the SAR checkpoint changes.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW = ROOT / "assets" / "sar_samples" / "raw"
OUT = ROOT / "assets" / "sar_samples" / "scenes"
MANIFEST = ROOT / "assets" / "sar_samples" / "manifest.json"

# Source scene -> curated identity. Vessel assignments follow the AIS scenario
# so the two halves of the demo agree: the casualty gets the major spill, the
# runner-up suspect gets a small discharge, and the erratic trawler that the
# anomaly detector over-flags gets a scene that is pure look-alike.
SCENES = [
    {
        "src": "WhatsApp Image 2026-08-27 at 8.32.30 AM.jpeg",
        "key": "wakashio_reef",
        "title": "Pointe d'Esny — main slick",
        "mmsi": 371284000,
        "vessel": "MV WAKASHIO",
        "captured": "2020-08-07 06:14 UTC",
        "position": [-20.4372, 57.7433],
        "blurb": ("Sentinel-1 pass over the grounding site. A large, branching "
                  "mineral-oil signature trailing from the wreck."),
    },
    {
        "src": "WhatsApp Image 2026-08-27 at 8.32.30 AM (4).jpeg",
        "key": "anchorage_discharge",
        "title": "Anchorage — small discharge",
        "mmsi": 352001899,
        "vessel": "MV BLUE BAY TRADER",
        "captured": "2020-07-25 14:02 UTC",
        "position": [-20.2650, 57.8300],
        "blurb": ("A compact slick alongside several hard targets — the "
                  "signature of an operational discharge at anchor."),
    },
    {
        "src": "WhatsApp Image 2026-08-27 at 8.32.30 AM (8).jpeg",
        "key": "lookalike_field",
        "title": "Low-wind field — look-alike",
        "mmsi": 645079210,
        "vessel": "FV SEA HARVESTER 7",
        "captured": "2020-07-25 15:40 UTC",
        "position": [-20.4750, 57.9400],
        "blurb": ("Extensive dark formations that are NOT oil. The segmenter "
                  "calls them look-alike, which is the whole point of the class."),
    },
    {
        "src": "WhatsApp Image 2026-08-27 at 8.32.30 AM (6).jpeg",
        "key": "coastal_lookalike",
        "title": "Coastal water — sheltered calm",
        "mmsi": 563114900,
        "vessel": "BULK PIONEER",
        "captured": "2020-07-25 15:10 UTC",
        "position": [-20.2100, 58.2600],
        "blurb": ("Wave shadow in the lee of the coast. Dark, adjacent to land, "
                  "and correctly not called oil."),
    },
    {
        "src": "WhatsApp Image 2026-08-27 at 8.32.30 AM (2).jpeg",
        "key": "clean_coastal",
        "title": "Clean coastal pass",
        "mmsi": 256891004,
        "vessel": "MSC CAP FLORES",
        "captured": "2020-07-25 15:25 UTC",
        "position": [-20.5300, 57.8100],
        "blurb": "Negative control — coastline and open water, no dark formations.",
    },
]


def main() -> int:
    from poseatsea.config import CLASS_NAMES, DEFAULT_PIXEL_RESOLUTION_M
    from poseatsea.inference import sar as sar_mod
    from poseatsea.registry import get_registry

    if not RAW.exists():
        print(f"Source folder not found: {RAW}")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    model = get_registry()["sar"].get()
    print(f"segmenter loaded ({get_registry()['sar'].param_count:,} params)\n")

    manifest = []
    for scene in SCENES:
        src = RAW / scene["src"]
        if not src.exists():
            print(f"  !! missing source {scene['src']}")
            return 1

        image = sar_mod.read_image(src)
        result = sar_mod.segment(model, image)

        scene_jpg = OUT / f"{scene['key']}.jpg"
        mask_png = OUT / f"{scene['key']}_mask.png"
        shutil.copy2(src, scene_jpg)
        # Single-channel PNG of raw class indices 0-4: lossless and tiny.
        cv2.imwrite(str(mask_png), result.mask)

        entry = {
            **{k: v for k, v in scene.items() if k != "src"},
            "image": f"scenes/{scene_jpg.name}",
            "mask": f"scenes/{mask_png.name}",
            "source_size": [int(image.shape[1]), int(image.shape[0])],
            "pixel_resolution_m": DEFAULT_PIXEL_RESOLUTION_M,
            "class_pixels": {CLASS_NAMES[c]: int(n) for c, n in result.class_pixels.items()},
            "oil_area_km2": round(result.oil_area_km2, 4),
            "lookalike_area_km2": round(result.lookalike_area_km2, 4),
            "oil_detected": bool(result.oil_detected),
            "slicks": len(result.slicks),
            "ships": len(result.ships),
            "note": result.confidence_note(),
        }
        manifest.append(entry)
        print(f"  {scene['key']:22s} oil {entry['oil_area_km2']:6.2f} km2  "
              f"slicks {entry['slicks']:2d}  -> {scene['vessel']}")

    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {MANIFEST.relative_to(ROOT)} ({len(manifest)} scenes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
