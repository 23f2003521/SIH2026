# Deployment

Target: **Streamlit Community Cloud**, free, with the large model weight served
from the Hugging Face Hub.

Live artefacts:

- Weights — <https://huggingface.co/23f2003521/poseatsea-weights> (uploaded, public)
- Code — <https://github.com/23f2003521/SIH2026>

---

## Picking a host

| | Streamlit Community Cloud | HF Space (Docker) | HF Space (Static) |
|---|---|---|---|
| Runs Python | Yes | Yes | **No** |
| Cost | Free | Free on CPU basic | Free |
| Resolves Git LFS | **No** | Yes | n/a |
| Fits this app | **Yes** | Yes | No |

**Static Spaces serve only HTML/CSS/JS** — they cannot run PyTorch, so they are
not an option for a live model demo.

### Will it fit?

Measured resident memory on this build:

| Loaded | RSS |
|---|---|
| torch + pandas + scikit-learn | 308 MB |
| \+ AIS autoencoder + trajectory LSTM | 317 MB |
| \+ streamlit | 336 MB |
| \+ SAR SegFormer | 678 MB |

Comfortably inside Streamlit Cloud's budget. The SAR segmenter is the only
heavy component — 342 MB on its own — and the registry loads it lazily, so a
session that never opens the SAR page never pays for it.

---

## How the weights are split

`best_sar_model.pth` is 105 MB, over GitHub's 100 MB blob limit. Git LFS would
clear that limit, but Streamlit Cloud clones *without* LFS support, so the file
would arrive as a ~130-byte pointer stub and `torch.load` would fail on it.

So the split is:

| Artefact | Size | Where it lives |
|---|---|---|
| `trajectory_lstm_baseline.pth` | 792 KB | committed to git |
| `ais_phase1_autoencoder.pth` | 8 KB | committed to git |
| `ais_phase1_scaler.joblib` | 4 KB | committed to git |
| `best_sar_model.pth` | 105 MB | Hugging Face Hub |

No Git LFS is used anywhere. `.gitignore` excludes only the SAR checkpoint.

`poseatsea/weights.py` resolves each artefact at load time: present on disk and
valid → use it; missing, or an unresolved LFS stub → fetch from the repo named
by `POSEATSEA_WEIGHTS_REPO`. Verified end to end — with the SAR file deleted
locally, the registry downloaded and loaded it in 20.6 s.

A further benefit: a corrected SAR checkpoint can be swapped in by re-uploading
one file to the Hub. No commit, no redeploy.

---

## 1. Weights on the Hub — done

Already uploaded to <https://huggingface.co/23f2003521/poseatsea-weights>.

To replace a checkpoint later:

```bash
python deploy/upload_weights.py --token hf_xxx
```

Creates the repo if absent and overwrites the four artefacts. Needs a **write**
token from <https://huggingface.co/settings/tokens>.

## 2. Push the code

```bash
cd C:/Users/hbp/Desktop/SIH2026/multimodal-oil-spill-detection
git push -u origin main
```

The remote is already set to `https://github.com/23f2003521/SIH2026.git` and the
commit is prepared. `venv/`, `models/files (2).zip` and the 105 MB SAR
checkpoint are all excluded.

## 3. Deploy

At <https://share.streamlit.io> → **Create app** → **Deploy a public app from GitHub**:

| Field | Value |
|---|---|
| Repository | `23f2003521/SIH2026` |
| Branch | `main` |
| Main file path | `ui/app.py` |
| Python version | `3.11` (under **Advanced settings**) |

Still under **Advanced settings → Secrets**, paste:

```toml
POSEATSEA_WEIGHTS_REPO = "23f2003521/poseatsea-weights"
```

Then **Deploy**. First build takes 5–10 minutes — torch is a large install.

Your slide link will be `https://<app-name>.streamlit.app`.

---

## Before you present

- **Wake the app.** Free apps sleep after ~7 days idle and show a wake button
  that takes 30–60 s. Open the link a few minutes early.
- **Warm the models.** Open **System → Warm up all models** once so the load
  cost is not inside your live demo. The first SAR load also pulls 105 MB from
  the Hub.
- **Know what to show.** The Incident console, AIS, Trajectory and Attribution
  pages are the working story. See the SAR note below.
- **Record a backup video.** Conference wifi is conference wifi.

---

## Troubleshooting

**Build fails resolving torch**
The `--extra-index-url` line must stay the *first* line of `requirements.txt`.
If a version has no wheel for the chosen Python, loosen `torch>=2.2,<3` to bare
`torch`, or drop the Python version to 3.11.

**`ImportError: libGL.so.1`**
Something pulled `opencv-python` instead of `opencv-python-headless`. Only the
headless build belongs on a server.

**SAR page errors with "missing and POSEATSEA_WEIGHTS_REPO is not set"**
The secret is absent or misspelled. It must be exactly
`POSEATSEA_WEIGHTS_REPO`, and the Hub repo must hold the four filenames at its
root.

**App exceeds resource limits**
Only the SAR page costs real memory (342 MB). It is also not producing valid
output at present, so leaving it closed costs nothing.

**Slow first SAR load**
Expected — it downloads 105 MB from the Hub, then caches it for the life of the
container.

---

## A note on the SAR page

`best_sar_model.pth` currently contains a trained MiT-B2 encoder with an
**untrained decoder and segmentation head** — those weights are still at
`torch.nn.init` values (uniform, kurtosis −1.2, min/max exactly at the Kaiming
bound). It emits incoherent noise regardless of input resolution or
preprocessing, so the failure is in the checkpoint, not the code.

Deploy anyway. Everything else works. When a corrected export arrives, run
`deploy/upload_weights.py` and reboot the app — one file, no code change.
