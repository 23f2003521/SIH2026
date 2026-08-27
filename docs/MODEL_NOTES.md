# Model notes — what was verified, and what was found

The handoff spec describes the three models' interfaces. Everything below was
checked against the actual checkpoint bytes rather than taken on trust, because
a spec and a weight file can disagree and only one of them runs.

Nothing here contradicts the spec. Sections 1–3 confirm it. Sections 4–6 are
findings the spec does not cover, because they only appear once you probe the
weights.

---

## 1. Architectures confirmed against the checkpoints

All three `state_dict`s load with `strict=True` — no missing keys, no unexpected
keys, no shape mismatches.

| Model | Evidence in the checkpoint |
|---|---|
| SAR segmenter | 394 tensors. `encoder.patch_embed{1,2,3}.proj.weight` carry the 64 → 128 → 320 channel progression unique to **MiT-B2**; `segmentation_head.0.weight` is `(5, 16, 3, 3)`. Top-level split: 332 encoder / 60 decoder / 2 head. |
| Trajectory LSTM | `lstm.weight_ih_l0` is `(512, 6)` = 4 gates × 128 hidden over a 6-dim input; two layers (`l0`, `l1`); `head.weight` is `(2, 128)`. |
| AIS autoencoder | `encoder.0/2/4` = (16,11), (8,16), (4,8); `decoder.0/2/4` = (8,4), (16,8), (11,16). |

The scaler is a `StandardScaler` with `n_features_in_ = 11`, matching
`FEATURE_ORDER`. Loading it under scikit-learn 1.9.0 (it was pickled under
1.6.1) emits `InconsistentVersionWarning` but unpickles correctly.

Load cost on CPU: SAR **~3.6 s** and 110 MB, trajectory ~0.02 s, autoencoder
~0.01 s. SAR forward pass ~2.2 s per 256×256 scene. This asymmetry is the whole
reason for the lazy registry.

---

## 2. The scaler is load-bearing

Scoring a normal transit ping:

| Path | Reconstruction error | Verdict |
|---|---|---|
| With the scaler | **0.0053** | normal (correct) |
| Skipping the scaler | **~10³ ×** larger | flags everything |

Forgetting the scaler is the single most damaging silent bug available in this
codebase — it produces confident output that is entirely wrong. It is asserted
in `tests/test_pipeline.py::test_unscaled_input_gives_a_different_answer`.

---

## 3. The anomaly threshold is well calibrated

Reconstruction error against `AE_THRESHOLD = 1.104481`:

| Ping signature | Score | Flagged |
|---|---|---|
| Normal transit, 11.2 kn steady | 0.005 | no |
| Slow deliberate turn | 0.127 | no |
| Moored, all diffs zero | 0.911 | no |
| Stationary aground | 1.94 | **yes** |
| Hard evasive manoeuvre | 3.36 | **yes** |
| Grounding at service speed | 8.30 | **yes** |

Three orders of magnitude separate routine transit from a grounding. The
threshold sits in a genuine gap in the distribution, not in the middle of a
cluster.

---

## 4. Finding: the trajectory model's cadence and heading envelope

**Not in the spec.** The published accuracy (mean 0.37 km) holds only inside a
narrower envelope than the AOI box implies.

Probing the checkpoint with synthetic constant-heading tracks — true step versus
predicted step and bearing:

| Ping interval | Course | True step | Predicted step | Bearing error |
|---|---|---|---|---|
| 60 s | 045° | 0.340 km | 0.246 km | **1.8°** |
| 60 s | 225° | 0.340 km | 0.470 km | **8.3°** |
| 60 s | 135° | 0.340 km | 0.389 km | 139° |
| 60 s | 315° | 0.340 km | 0.248 km | 151° |
| 120 s | 045° | 0.679 km | 0.351 km | 23° |
| 300 s | 045° | 1.698 km | 0.779 km | 43° |
| 300 s | 135° | 1.698 km | 5.233 km | 162° |

Two conclusions:

1. **The model was trained on ~60 s cadence.** Step magnitude tracks truth at
   60 s and degrades steadily beyond it. At 300 s the prediction is less than
   half the true distance on some headings and 3× too far on others.
2. **It learned the NE–SW lane and little else.** Bearing error is a few degrees
   near 045°/225° and roughly 140–160° — near-reversed — on the NW–SE axis.
   Mauritius traffic runs predominantly NE–SW, so those headings are sparse in
   training.

`trajectory.assess_inputs()` enforces both: windows outside the cadence or the
reliable course bands are marked **degraded** in the UI, and out-of-AOI windows
are refused outright. `TRAJ_RELIABLE_COURSE_BANDS` in `config.py` encodes the
bands measured above.

Validation on the reconstructed scenario (60 s cadence, courses 228–248°):
median deviation **0.066–0.134 km** per vessel, against the model's own published
median of 0.19 km. In-distribution input reproduces published accuracy.

---

## 5. Finding: synthetic radar frames are out of distribution

The segmenter's output on `scenario/synthetic_sar.py` frames, across four
presets designed to differ substantially:

| Preset | Oil px | Look-alike px |
|---|---|---|
| Slick beside a look-alike | 6,316 | 12,195 |
| **Clean sea, no slick at all** | **9,441** | 14,134 |
| Large coastal slick | 6,487 | 11,842 |
| Look-alike only | 8,112 | 12,264 |

The clean-sea negative control returns *more* oil pixels than either slick
scene. The model is not discriminating; it is responding to texture statistics
it was never trained on.

This is expected — the generator reproduces the *look* of SAR (gamma speckle,
dark formations, bright hard targets) but none of the Sentinel-1 VV backscatter
physics the model actually learned. It is retained purely as a plumbing test and
labelled as such wherever it appears.

**Real Sentinel-1 imagery from the Krestenitis benchmark is required for any
claim about segmentation accuracy.**

---

## 6. Finding: derived course must respect steerage way

Not a model property, but it changes what the models see.

The first draft of the scenario generator derived course over ground from
consecutive positions for every ping. For vessels with way on, correct. For a
vessel at anchor, the fix-to-fix displacement is metres-scale receiver dither,
so the derived bearing swung randomly through 360° — which the autoencoder
correctly read as violent turning.

Result: the anchored cargo vessel flagged on **67%** of its pings with a peak
score of 28.0, higher than the actual casualty, purely from GPS noise.

Real AIS holds the last steerage course below about 0.5 kn. Applying that
(`STEERAGE_SPEED_KN`, `STEERAGE_MIN_DISPLACEMENT_M`) dropped the anchored vessel
to **1.6%** flagged — the two remaining flags being the genuine behavioural
transition as it came to anchor.

The lesson generalises past this project: a detector fed a physically
inconsistent feed will find real anomalies in the feed's own artefacts.
`tests/test_pipeline.py::test_anchored_vessel_is_not_treated_as_anomalous`
guards the regression.

---

## 7. Deliberate omissions

- **No hindcasting / drift modelling.** Out of scope per the handoff, and there
  is no current or wind data here to do it with.
- **No retraining or fine-tuning.** The weights are treated as final.
- **The trajectory model is not an input to the anomaly detector.** Adding
  predicted-versus-actual distance as a twelfth feature was tested during model
  development and did not improve accuracy. Re-adding it without re-testing
  would be a regression.
- **TrAISformer is not used** — it measured 2.6+ km error against the LSTM's
  0.37 km.
