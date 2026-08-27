# POSEatSea — Oil Spill Detection & Vessel Attribution

**SIH Problem Statement 26143 (NTRO)** — leveraging satellite imagery and AIS data to
detect oil spills and attribute them to the responsible vessel.

An operations console built around three trained models: a SAR segmenter that finds oil
in radar imagery, an autoencoder that flags anomalous vessel behaviour, and an LSTM that
predicts where a vessel should be next. A correlation layer ties their outputs to a
single place and time and ranks the vessels that could account for an observed slick.

---

## Quick start

```bash
python -m venv venv
venv\Scripts\activate            # Windows;  source venv/bin/activate on Linux/macOS
pip install -r requirements.txt

streamlit run ui/app.py
```

Opens on <http://localhost:8501>. No internet connection is required at runtime —
charts render offline and no basemap tiles are fetched.

Verify the install:

```bash
pytest tests/ -q          # 37 tests: models, guard rails, API contract, every page
```

---

## What the console shows

| Page | What it does |
|---|---|
| **Incident console** | The whole story on one screen — traffic, flags, and the attributed vessel. |
| **SAR segmentation** | Upload a Sentinel-1 scene; get a 5-class mask, area estimate and slick breakdown. |
| **AIS anomalies** | Per-ping anomaly scoring across the fleet, down to which feature drove each flag. |
| **Trajectory** | Predicted next position against the vessel's actual track, with a deviation trace. |
| **Attribution** | Ranks candidate vessels against a slick, showing every component of the score. |
| **System** | Model residency, load timings, reported accuracy, and declared limitations. |

---

## Architecture

```
poseatsea/
  config.py              every constant the weights depend on, in one place
  registry.py            lazy, thread-safe, instrumented model registry
  fusion.py              spill-to-vessel correlation and ranking
  inference/
    sar.py               U-Net + MiT-B2 segmentation
    trajectory.py        LSTM next-position + operating-envelope guard
    ais.py               autoencoder anomaly scoring + feature engineering
  scenario/
    wakashio.py          reconstructed MV Wakashio incident
    synthetic_sar.py     synthetic radar frames (plumbing test only)
api/main.py              optional FastAPI inference service
ui/                      Streamlit console (engine, theme, charts, views)
tests/                   pipeline, API and page smoke tests
```

### Carrying the heavy models

The SAR segmenter is a 110 MB MiT-B2 SegFormer: several seconds to build and a few
hundred MB resident. Streamlit re-executes its whole script on every widget
interaction, so naive loading would rebuild it on every click.

`poseatsea/registry.py` gives the process exactly one instance of each model:

- **Lazy** — a session that only touches AIS never pays for the SAR encoder.
  Model residency is visible live in the sidebar.
- **Locked per model** — concurrent first-touches load once, not *N* times, and
  inference is serialised because the eval-mode modules are shared mutable state.
- **Instrumented** — load time and parameter counts are surfaced on the System page
  rather than hidden.
- **Strict** — every `state_dict` loads with `strict=True`, so an architecture drift
  becomes a startup error instead of a quietly wrong answer.

Streamlit's `st.cache_resource` wraps the registry; `st.cache_data` memoises the
derived analytics. The optional FastAPI service imports the *same* inference modules
and the *same* registry, so there is one implementation of each model's behaviour and
no way for the two front doors to disagree.

```bash
uvicorn api.main:app --port 8000     # docs at /docs
```

Use it when in-process loading stops being viable: several analysts sharing one GPU,
a UI that restarts without repaying the load, or a non-Streamlit consumer.

---

## The models

| Model | Architecture | Verified against checkpoint |
|---|---|---|
| SAR segmenter | U-Net, `mit_b2` encoder, 5 classes | `patch_embed{1,2,3}` channel progression 64→128→320 |
| Trajectory | LSTM(6→128) × 2 + Linear(128→2) | `lstm.weight_ih_l0` = (512, 6) |
| AIS anomaly | Autoencoder 11→16→8→4→8→16→11 | encoder/decoder layer shapes |

The AIS model **requires** its `StandardScaler`. It was trained on standardised
features and returns confident nonsense on raw input — `tests/test_pipeline.py`
asserts that skipping the scaler changes the answer by two orders of magnitude.

### Reported performance

- **AIS anomaly** — precision 0.930, recall 0.408, F1 0.567, threshold 1.104481.
  Deliberately precision-heavy: when it fires it is almost always right, but it misses
  roughly six anomalies in ten. Every surface says *flagged for review*, never
  *confirmed violation*.
- **Trajectory** — mean error 0.37 km, median 0.19 km, p90 0.63 km on unseen vessels.

### Measured operating envelope

Beyond the published figures, the trajectory checkpoint was probed directly. Two
limits emerged, and both are enforced in code rather than left to be discovered:

- **Cadence** — step size is reproduced well at ~60 s ping intervals. At 5-minute
  spacing the predicted step falls below half the true distance travelled.
- **Heading** — bearing error is small along the NE–SW lane (courses near 045° and
  225°), which dominates traffic in the training region. On NW–SE headings the model
  frequently predicts close to the *reverse* bearing. Such windows are marked
  *degraded* in the UI.

Out-of-AOI input is refused outright rather than answered wrongly.

---

## Demonstration data

No AIS feed or SAR imagery was supplied with the models, so the console ships with a
**reconstruction of the MV Wakashio grounding** (Pointe d'Esny, Mauritius,
25 July 2020) — the case study named in the project brief, and the same water and
month the trajectory model was normalised for.

Vessel particulars, voyage, grounding position and timing follow the public casualty
record. **The individual AIS pings are physically consistent synthesis, not recovered
signal.** Tracks are generated from waypoints with a speed profile; course, rate of
turn and positional deltas are all derived from the resulting geometry, so every field
the models consume agrees with every other one. The console labels this on every page.

The scenario also carries five contemporaneous vessels, which is what makes the
demonstration honest — attribution only means something if the system had innocent
traffic available to blame and did not blame it.

One deliberate result: **an inshore trawler carries the highest raw anomaly score in
the window, not the casualty.** Tight repeated turns are kinematically anomalous and
entirely lawful. Behaviour alone would nominate the wrong vessel; only correlation with
the spill's position resolves it. That is the argument for fusion, made concrete.

### SAR imagery — read this before demoing

The synthetic radar generator (`scenario/synthetic_sar.py`) is a **plumbing test, not
imagery**. Those frames are out of distribution for the segmenter, which responds to
them almost identically regardless of what was drawn — the clean-sea control returns
*more* oil pixels than the slick scenes. They prove the model loads and runs; they say
nothing about accuracy.

**For a real demonstration, supply Sentinel-1 scenes** from the Krestenitis et al.
oil-spill benchmark the weights were trained on (1002 train / 110 test). Any file from
its `test/images/` folder is a valid input to the upload path.

---

## Declared limitations

1. **No drift or hindcast modelling.** Attribution assumes the slick lies where it was
   observed. There is no ocean-current or wind model here. Faking one would be worse
   than omitting it — a plausible backtrack with no physics behind it would send an
   investigation to the wrong vessel with false confidence. Wiring in real drift data
   (for example INCOIS current products) would replace the fixed search radius with a
   time-reversed probability field; that is the natural next component.
2. **The trajectory model is regional.** Valid only inside the Mauritius AOI it was
   normalised for. Elsewhere requires retraining.
3. **The anomaly detector misses more than it catches.** Recall 0.41. It nominates
   vessels for review; it cannot clear one.
4. **Area estimates are approximate.** The mask is a 256×256 resample of the source
   scene, and ground resolution is an operator-supplied assumption.
5. **The demonstration AIS is reconstructed**, as described above.
6. **Attribution weights are a policy choice, not a measurement**, and are exposed in
   the UI so they can be argued with.

---

## Attribution scoring

A transparent weighted sum, not a learned model — every component is returned
alongside the total so an analyst can see why a vessel ranked where it did.

| Component | Weight | Measures |
|---|---|---|
| Proximity | 0.40 | Closest approach to the observed slick |
| Anomaly | 0.30 | Peak reconstruction error *while near the slick* |
| Dwell | 0.20 | Share of its observed time spent inside the radius |
| Deviation | 0.10 | Worst departure from its predicted track |

Anomalies only count as evidence if they occurred near the slick — which is precisely
what clears the trawler.
