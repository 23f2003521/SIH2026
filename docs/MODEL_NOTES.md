# Model notes — what was verified, and what was found

The handoff spec describes the three models' interfaces. Everything below was
checked against the actual checkpoint bytes and the real AIS feed rather than
taken on trust, because a spec and a weight file can disagree and only one of
them runs.

Sections 1–3 confirm the spec. Sections 4–7 are findings it does not cover.

---

## 1. Architectures confirmed against the checkpoints

All three `state_dict`s load with `strict=True` — no missing keys, no unexpected
keys, no shape mismatches.

| Model | Evidence in the checkpoint |
|---|---|
| SAR segmenter | 394 tensors. `encoder.patch_embed{1,2,3}.proj.weight` carry the 64 → 128 → 320 channel progression unique to **MiT-B2**; `segmentation_head.0.weight` is `(5, 16, 3, 3)`. Split: 332 encoder / 60 decoder / 2 head. |
| Trajectory LSTM | `lstm.weight_ih_l0` is `(512, 6)` = 4 gates × 128 hidden over a 6-dim input; two layers; `head.weight` is `(2, 128)`. |
| AIS autoencoder | `encoder.0/2/4` = (16,11), (8,16), (4,8); `decoder.0/2/4` = (8,4), (16,8), (11,16). |

The scaler is a `StandardScaler` with `n_features_in_ = 11`, matching
`FEATURE_ORDER`.

Load cost on CPU: SAR **~4.6 s** and 342 MB resident; trajectory ~0.02 s;
autoencoder ~0.01 s. That asymmetry is the whole reason for the lazy registry.

---

## 2. The scaler is load-bearing

Scoring a normal transit ping:

| Path | Reconstruction error | Verdict |
|---|---|---|
| With the scaler | **0.0053** | normal (correct) |
| Skipping the scaler | **~10³ ×** larger | flags everything |

Forgetting the scaler is the most damaging silent bug available here: it
produces confident output that is entirely wrong. Asserted in
`test_unscaled_input_gives_a_different_answer`.

---

## 3. The anomaly threshold is well calibrated

Measured on the **real** AIS feed, against `AE_THRESHOLD = 1.104481`:

| Vessel (real) | Pings | Flagged | Peak error |
|---|---|---|---|
| **WAKASHIO** (the casualty) | 390 | **53.6%** | **25.08** |
| KOTA SURIA | 306 | 0.3% | 1.17 |
| VERY MARIA | 148 | 0% | 0.99 |
| AQUAVITA SOL | 43 | 0% | 0.84 |
| PALONA | 55 | 0% | 0.55 |
| DHT EDELWEISS | 100 | 0% | 0.23 |

A clean separation on data nobody tuned it against. The single KOTA SURIA flag
at 1.17 sits barely over the line — consistent with the model's stated 93%
precision rather than contradicting it.

**It fires on the strike itself, not just the aftermath.** The Wakashio's last
moving ping — 15:27:22 UTC, decelerating 8.7 → 1.7 kn — scores 2.67 and is
flagged, 84 seconds before the vessel reports 0.2 kn and stops for good.

---

## 4. Finding: the SAR checkpoint shipped untrained the first time

The initial `best_sar_model.pth` emitted salt-and-pepper noise on every input,
at every resolution. It was not a preprocessing bug.

| Tensor | Kurtosis | Reading |
|---|---|---|
| `encoder.patch_embed1.proj.weight` | 7.12 | trained |
| `encoder.block1.0.attn.q.weight` | 3.45 | trained |
| `decoder.blocks.0.conv1.0.weight` | **−1.200** | untrained |
| `decoder.blocks.2.conv1.0.weight` | **−1.198** | untrained |
| `segmentation_head.0.weight` | **−1.207** | untrained |

Kurtosis −1.2 with min/max exactly at the Kaiming bound is a uniform
distribution — i.e. untouched `torch.nn.init`. Confirmed by BatchNorm:
`num_batches_tracked = 0` across every decoder layer. A trained encoder feeding
an untrained decoder yields near-uniform logits, hence noise.

The replacement checkpoint has `num_batches_tracked = 7530` and reproduces the
reference outputs at **0.810 oil IoU / 98.7% pixel agreement**.

`test_sar_checkpoint_is_trained` now guards this, because it is a failure mode
that produces plausible-looking output and no error.

---

## 5. Finding: the segmenter only works at 512×512

Not a soft preference. The same real scene, same weights:

| Input size | Oil IoU vs reference | Mask coherence |
|---|---|---|
| **512×512** | **0.810** | 0.010 |
| 256×256 | 0.580 | 0.014 |

The updated spec states this and it is confirmed here. `SAR_INPUT_SIZE` is fixed
in `config.py` and deliberately not exposed as a UI setting.

### Measuring on the raw output understates area 3.1×

The network always emits 512×512 whatever you feed it. A Sentinel-1 frame is
1250×650 covering 12.5 km × 6.5 km, so one raw output pixel spans
**24.4 m × 12.7 m — 3.1× the area of the 10 m pixel the formula assumes.**

Counting oil pixels on the raw grid therefore understates the slick by that
factor, and does so silently: the number looks entirely plausible.

| Scene | Measured on raw 512×512 | Measured at source resolution |
|---|---|---|
| Wakashio, Pointe d'Esny | 1.38 km² | **4.28 km²** |
| Kota Suria, outer lane | 0.19 km² | **0.60 km²** |

`segment()` now resamples with `INTER_NEAREST` to the source dimensions before
any measurement, and `test_area_is_measured_at_source_resolution` asserts the
five class areas sum to the scene's true ground extent.

---

## 6. Finding: the trajectory model's cadence and heading envelope

**Not in the spec.** Probing the checkpoint with constant-heading tracks — true
step versus predicted step and bearing:

| Ping interval | Course | True step | Predicted step | Bearing error |
|---|---|---|---|---|
| 60 s | 045° | 0.340 km | 0.246 km | **1.8°** |
| 60 s | 225° | 0.340 km | 0.470 km | **8.3°** |
| 60 s | 135° | 0.340 km | 0.389 km | 139° |
| 60 s | 315° | 0.340 km | 0.248 km | 151° |
| 300 s | 045° | 1.698 km | 0.779 km | 43° |
| 300 s | 135° | 1.698 km | 5.233 km | 162° |

1. **Trained around 60 s cadence.** Step magnitude tracks truth there and
   degrades steadily beyond it.
2. **It learned the NE–SW lane and little else.** Bearing error is a few degrees
   near 045°/225° and 140–160° — near-reversed — on the NW–SE axis, which is
   sparse in Mauritius traffic.

`trajectory.assess_inputs()` enforces both: out-of-envelope windows are marked
**degraded**, and out-of-AOI windows are refused rather than answered.

**Validation on the real feed** — six vessels, 8–66 s cadence, courses 236–247°:

| Vessel | Windows | Median deviation |
|---|---|---|
| KOTA SURIA | 298 | 0.137 km |
| VERY MARIA | 140 | 0.167 km |
| DHT EDELWEISS | 92 | 0.172 km |
| AQUAVITA SOL | 35 | 0.177 km |
| PALONA | 47 | 0.182 km |
| WAKASHIO | 382 | 0.191 km |

Every vessel lands at or under the model's published **0.19 km median**, on real
tracks it never trained against. This is the strongest validation in the project.

---

## 7. Finding: the AIS sentinel values matter

Two fields in the raw feed will quietly poison the anomaly detector:

- **`rot = -128`** is the AIS code for "rate of turn not available", present on
  1,043 rows. Passed through raw it reads as a hard port swing. Mapped to 0 in
  `real_ais.clean_positions`.
- **Fractional-second timestamps** appear on ~4% of rows. `pd.to_datetime`
  infers one format from the first rows and silently drops the rest as `NaT` —
  deleting parts of a track without any error. `format="mixed"` recovers all
  1,017. Guarded by `test_every_timestamp_parses`.

The general lesson: a detector fed a subtly malformed feed will find real
anomalies in the feed's own artefacts rather than in the world.

---

## 8. Deliberate omissions

- **No hindcasting / drift modelling.** Out of scope per the handoff, and there
  is no current or wind data here to do it with.
- **No retraining or fine-tuning.** The weights are treated as final.
- **The trajectory model is not an input to the anomaly detector.** Adding
  predicted-versus-actual distance as a twelfth feature was tested during model
  development and did not improve accuracy.
- **TrAISformer is not used** — 2.6+ km error against the LSTM's 0.37 km.
- **No synthetic data anywhere.** An earlier build carried a reconstructed AIS
  scenario and a synthetic radar generator. Both were deleted once the real
  Mauritius extract and the working SAR checkpoint arrived.
