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

Opens on <http://localhost:8501>. Maps use Esri satellite tiles and need network at
view time; set `POSEATSEA_OFFLINE_MAPS=1` to fall back to a plain dark canvas.

Verify the install:

```bash
pytest tests/ -q          # 46 tests: models, guard rails, API contract, every page
```

---

## What the console shows

| Page | What it does |
|---|---|
| **Overview** | The whole story on one satellite map — traffic, flags, and the attributed vessel. |
| **Route Deviation (LSTM)** | Predicted next position against the vessel's actual track, with a deviation trace. |
| **AIS Anomaly Detection** | Per-ping scoring across the fleet, down to which feature drove each flag. |
| **SAR Oil Spill Segmenter** | Five real Sentinel-1 scenes, or upload your own for live inference. |
| **Attribution Pipeline** | Ranks candidate vessels against a slick, showing every component of the score. |
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
    real_ais.py          the Mauritius AOI feed (canonical data source)
    wakashio.py          documented casualty facts
    sar_scenes.py        precomputed Sentinel-1 scene library
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

## Data — all of it real

**AIS.** A Mauritius AOI extract covering 1–31 July 2020: 27,979 records,
231 vessels, the same region and month the trajectory model was normalised for.
**The MV Wakashio's grounding is in this feed, broadcast by the ship itself.**

Straight from the data, nothing asserted by hand:

- She transits south-west at 11.6–12.0 kn on courses 245–247°.
- Her last ping with way on is 2020-07-25 **15:27:22 UTC** at −20.44421,
  57.74328 — already down to 1.7 kn.
- **84 seconds later** she reports 0.2 kn, and never moves again.
- From there she broadcasts navigational status **6 — Aground** for six days.

15:27 UTC is 19:27 local, matching the casualty record's 19:25 to within two
minutes. The console derives the grounding position and time from the feed
rather than hardcoding them; a test asserts they land on the documented site.

The five other vessels on screen — KOTA SURIA, VERY MARIA, DHT EDELWEISS,
PALONA, AQUAVITA SOL — are real ships that were really there that day, chosen
for track density. They are the control group: attribution only means something
if the system had innocent traffic available to blame and did not blame it. It
scores the Wakashio at **0.936** and clears every one of them by proximity.

Only the *background* is documentary rather than measured: owner, tonnage,
voyage, and what happened after she stopped moving. Those come from the public
casualty record and live in `scenario/wakashio.py`.

**SAR.** Five real Sentinel-1 scenes with the masks the shipped checkpoint
produced for them, precomputed by `deploy/build_sar_scenes.py` so the page
renders without loading the 110 MB segmenter. Uploading a scene runs the model
live.

Each scene names a real vessel from the feed and is cross-referenced against
that vessel's own AIS result, which is where the two halves meet:

| Vessel | SAR oil | AIS flagged | Distance to slick | Verdict |
|---|---|---|---|---|
| **WAKASHIO** | **4.28 km²** | 209 pings | 0.0 km | **Primary suspect** |
| KOTA SURIA | 0.60 km² | 1 ping | 27.7 km | Cleared by proximity |
| VERY MARIA | none | 0 | 30.1 km | Cleared |
| DHT EDELWEISS | none | 0 | 28.3 km | Cleared |
| PALONA | none | 0 | 44.9 km | Cleared |

The KOTA SURIA row is the useful one: the segmenter *does* find oil in that
scene and the detector *did* flag one of its pings, yet the vessel was 27.7 km
away and is cleared. Neither model alone gets that right.

Areas are measured after the 512×512 network output is resampled back to the
source scene's resolution. Measuring on the raw output instead understates
ground area by the scene's aspect ratio — 3.1× on a 1250×650 Sentinel-1 frame —
which is a silent error, so a test pins it.

The SAR frames come from the Krestenitis benchmark, so pairing one with a
vessel's operating area is presentational — the segmentation output is genuine,
the geographic pairing is for the demo, and the UI states this.

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
5. **The AIS is real**, as described above; only the documentary background is not.
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
