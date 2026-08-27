#!/usr/bin/env python
"""
Launcher.

    python run.py             # console only (the demo path)
    python run.py --api       # inference service only
    python run.py --both      # both, console on 8501, API on 8000
    python run.py --check     # verify weights load, then exit

Exists so a demo never depends on remembering streamlit/uvicorn flags.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def preflight() -> bool:
    """Fail loudly and early if the weights are missing or mismatched."""
    import warnings

    warnings.filterwarnings("ignore")
    from poseatsea.config import (AIS_AE_WEIGHTS, AIS_SCALER, DEVICE,
                                  SAR_WEIGHTS, TRAJECTORY_WEIGHTS)

    print(f"device: {DEVICE}")
    missing = [p for p in (SAR_WEIGHTS, TRAJECTORY_WEIGHTS, AIS_AE_WEIGHTS, AIS_SCALER)
               if not p.exists()]
    if missing:
        print("\nMissing model artefacts:")
        for p in missing:
            print(f"  - {p}")
        print("\nPlace all four files in models/ and retry.")
        return False

    from poseatsea.registry import get_registry

    registry = get_registry()
    ok = True
    for key in ("ais_anomaly", "trajectory", "sar"):
        handle = registry[key]
        try:
            start = time.perf_counter()
            handle.get()
            print(f"  ok  {handle.display_name:32s} {time.perf_counter() - start:5.2f}s"
                  f"  {handle.param_count:,} params")
        except Exception as exc:
            print(f"  FAIL {handle.display_name:32s} {exc}")
            ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="POSEatSea launcher")
    parser.add_argument("--api", action="store_true", help="run the inference service only")
    parser.add_argument("--both", action="store_true", help="run console and service")
    parser.add_argument("--check", action="store_true", help="verify weights load, then exit")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--api-port", type=int, default=8000)
    args = parser.parse_args()

    print("POSEatSea -- preflight")
    if not preflight():
        return 1
    print("all models verified\n")

    if args.check:
        return 0

    console = [sys.executable, "-m", "streamlit", "run", str(ROOT / "ui" / "app.py"),
               "--server.port", str(args.port)]
    service = [sys.executable, "-m", "uvicorn", "api.main:app",
               "--port", str(args.api_port), "--host", "0.0.0.0"]

    try:
        if args.api:
            print(f"inference service -> http://localhost:{args.api_port}/docs")
            return subprocess.call(service, cwd=ROOT)
        if args.both:
            print(f"inference service -> http://localhost:{args.api_port}/docs")
            proc = subprocess.Popen(service, cwd=ROOT)
            try:
                print(f"console           -> http://localhost:{args.port}")
                return subprocess.call(console, cwd=ROOT)
            finally:
                proc.terminate()
        print(f"console -> http://localhost:{args.port}")
        return subprocess.call(console, cwd=ROOT)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
