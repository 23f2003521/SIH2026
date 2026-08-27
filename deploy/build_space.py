#!/usr/bin/env python
"""
Assemble a Hugging Face Space folder from this project.

    python deploy/build_space.py ../poseatsea-space
    python deploy/build_space.py ../poseatsea-space --no-weights   # Option B

Copies only what the Space needs, swaps in the Space README and the CPU-torch
requirements, and prints the exact git commands to finish. Copying by hand is
where deployments usually break -- typically by shipping `venv/`, forgetting
that the Space README carries required YAML front matter, or leaving the
102 MB source zip in `models/`.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Directories the Space needs, copied wholesale.
PACKAGES = ["poseatsea", "ui", "api", ".streamlit"]

# Everything below is skipped inside those directories.
PRUNE_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", "venv", ".git"}

WEIGHTS = [
    "best_sar_model.pth",
    "trajectory_lstm_baseline.pth",
    "ais_phase1_autoencoder.pth",
    "ais_phase1_scaler.joblib",
]

GITATTRIBUTES = """*.pth  filter=lfs diff=lfs merge=lfs -text
*.joblib filter=lfs diff=lfs merge=lfs -text
"""


def _ignore(_dir, names):
    return [n for n in names if n in PRUNE_DIRS or n.endswith(".pyc")]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a Hugging Face Space folder")
    ap.add_argument("target", type=Path, help="Space repo folder (cloned or new)")
    ap.add_argument("--no-weights", action="store_true",
                    help="omit weights; fetch them via POSEATSEA_WEIGHTS_REPO instead")
    args = ap.parse_args()

    target: Path = args.target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    print(f"building Space in {target}\n")

    for pkg in PACKAGES:
        src = ROOT / pkg
        if not src.exists():
            print(f"  !! missing {pkg}")
            return 1
        dst = target / pkg
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=_ignore)
        print(f"  copied  {pkg}/")

    # Space README carries the YAML front matter that configures the Space --
    # it is configuration, not documentation.
    shutil.copy2(ROOT / "deploy" / "README_SPACE.md", target / "README.md")
    print("  copied  deploy/README_SPACE.md -> README.md")

    shutil.copy2(ROOT / "deploy" / "requirements-hf.txt", target / "requirements.txt")
    print("  copied  deploy/requirements-hf.txt -> requirements.txt")

    (target / ".gitattributes").write_text(GITATTRIBUTES, encoding="utf-8")
    print("  wrote   .gitattributes (LFS rules)")

    models_dir = target / "models"
    if args.no_weights:
        if models_dir.exists():
            shutil.rmtree(models_dir)
        print("\n  weights omitted -- set POSEATSEA_WEIGHTS_REPO in Space settings")
    else:
        models_dir.mkdir(exist_ok=True)
        total = 0
        for name in WEIGHTS:
            src = ROOT / "models" / name
            if not src.exists():
                print(f"  !! missing weight {name}")
                return 1
            shutil.copy2(src, models_dir / name)
            mb = src.stat().st_size / 1024 / 1024
            total += mb
            print(f"  copied  models/{name}  ({mb:.1f} MB)")
        print(f"          {total:.1f} MB of weights -- all must go through LFS")

    print("\nnext:\n")
    print(f"  cd {target}")
    print("  git lfs install")
    print('  git lfs track "*.pth" "*.joblib"')
    print("  git add -A")
    print("  git lfs ls-files          # all four artefacts must appear here")
    print('  git commit -m "POSEatSea console"')
    print("  git push\n")
    print("Authenticate with a WRITE token from huggingface.co/settings/tokens.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
