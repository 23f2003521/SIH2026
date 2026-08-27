#!/usr/bin/env python
"""
Upload the model weights to a Hugging Face model repo.

    python deploy/upload_weights.py                       # uses the default repo
    python deploy/upload_weights.py --repo you/other-repo
    python deploy/upload_weights.py --token hf_xxx        # or set HF_TOKEN

Why weights live on the Hub rather than in git:

  * `best_sar_model.pth` is 105 MB, over GitHub's 100 MB blob limit.
  * Git LFS would clear that, but Streamlit Community Cloud clones without LFS
    support, so the file would arrive as a ~130-byte pointer stub.
  * Hosting on the Hub also means a corrected checkpoint can be swapped in by
    re-uploading one file -- no commit, no redeploy.

Creates the repo if it does not exist. Needs a WRITE token from
https://huggingface.co/settings/tokens
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "23f2003521/poseatsea-weights"

ARTEFACTS = [
    "best_sar_model.pth",
    "trajectory_lstm_baseline.pth",
    "ais_phase1_autoencoder.pth",
    "ais_phase1_scaler.joblib",
]

CARD = """---
license: mit
tags:
  - oil-spill-detection
  - maritime
  - sar
  - ais
---

# POSEatSea — model weights

Weights for the POSEatSea maritime oil-spill detection and vessel attribution
console (SIH Problem Statement 26143, NTRO).

| File | Architecture | Notes |
|---|---|---|
| `best_sar_model.pth` | U-Net + MiT-B2 encoder, 5 classes | Expects 512×512 input. **See caveat below.** |
| `trajectory_lstm_baseline.pth` | LSTM(6→128)×2 + Linear(128→2) | Mauritius AOI, ~60 s ping cadence |
| `ais_phase1_autoencoder.pth` | Autoencoder 11→16→8→4→8→16→11 | Threshold 1.104481 |
| `ais_phase1_scaler.joblib` | StandardScaler, 11 features | **Mandatory** — the model returns nonsense on unscaled input |

Code: https://github.com/23f2003521/SIH2026

## Caveat on the SAR checkpoint

`best_sar_model.pth` currently holds a trained MiT-B2 encoder with an
**untrained decoder and segmentation head** — those weights are still at
`torch.nn.init` values (uniform, kurtosis −1.2, min/max exactly at the Kaiming
bound). It emits incoherent noise regardless of input resolution or
preprocessing. Replace this file with a corrected export to fix the SAR page;
no code change is needed.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Upload POSEatSea weights to the Hub")
    ap.add_argument("--repo", default=DEFAULT_REPO, help=f"default: {DEFAULT_REPO}")
    ap.add_argument("--token", default=None, help="HF write token (or set HF_TOKEN)")
    ap.add_argument("--private", action="store_true", help="create the repo private")
    args = ap.parse_args()

    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("No token. Pass --token hf_xxx, set HF_TOKEN, or run "
              "`huggingface-cli login` first.\n"
              "Create a WRITE token at https://huggingface.co/settings/tokens")
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is not installed. Run: pip install huggingface-hub")
        return 1

    missing = [n for n in ARTEFACTS if not (ROOT / "models" / n).exists()]
    if missing:
        print(f"Missing from {ROOT / 'models'}: {missing}")
        return 1

    api = HfApi(token=token)

    print(f"repo: {args.repo}")
    api.create_repo(repo_id=args.repo, repo_type="model",
                    private=args.private, exist_ok=True)

    card = ROOT / "models" / "README.md"
    card.write_text(CARD, encoding="utf-8")
    try:
        for name in ARTEFACTS + ["README.md"]:
            path = ROOT / "models" / name
            mb = path.stat().st_size / 1024 / 1024
            print(f"  uploading {name}  ({mb:.1f} MB) ...", flush=True)
            api.upload_file(path_or_fileobj=str(path), path_in_repo=name,
                            repo_id=args.repo, repo_type="model")
    finally:
        card.unlink(missing_ok=True)          # keep it out of the git checkout

    print(f"\ndone -> https://huggingface.co/{args.repo}")
    print(f"\nSet this secret in Streamlit Cloud:\n"
          f'  POSEATSEA_WEIGHTS_REPO = "{args.repo}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
