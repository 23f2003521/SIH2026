# POSEatSea — Gap Analysis & Roadmap (SIH 26143)

**Scope:** Oil-spill detection from satellite imagery + AIS-based vessel attribution.
**Question this doc answers:** given what POSEatSea already has, what are the highest-impact things we have NOT implemented yet?

Audit date: 2026-09-24. Based on a direct code audit of this repository (entrypoint `ui/app.py` / `run.py`, 4-page Streamlit app) cross-referenced against concrete, verified techniques from external repos (OpenDrift/OpenOil, PyGNOME, `nansencenter/openwind`, MMSegmentation, eo-learn, xView3-reference, `krestenitis/oil-spill-detection-dataset`, GFW loitering/encounter methodology, TrAISformer, InVEST). Items that could not be verified from source are marked **Not verified** rather than guessed.

---

## 1. Current state (what POSEatSea already has)

Confirmed by reading the actual code, not commit messages:

- **SAR segmentation** — U-Net + `mit_b2` (SegFormer encoder) via `segmentation_models_pytorch`, 5-class output (sea/oil/look-alike/ship/land), `poseatsea/inference/sar.py`. Trained on the Krestenitis et al. benchmark.
- **Slick geometry** — `cv2.connectedComponentsWithStats` for per-slick pixel count, centroid, bbox, area in km² (`sar.py:_components`).
- **Ship detection** — a byproduct of the same segmentation model (class 3 = "Ship"), cross-referenced against AIS MMSI in `sar_view.py:_vessel_crossref`.
- **AIS ingestion** — `poseatsea/scenario/real_ais.py` loads/cleans real AIS data for the Mauritius/Wakashio scenario.
- **AIS anomaly detection** — an 11→16→8→4→8→16→11 autoencoder (`poseatsea/inference/ais.py`), reconstruction-error threshold, documented model card (P=0.93, R=0.41). General kinematic anomaly, not loitering-specific.
- **AIS gap tracking** — `trajectory.py` computes `coverage_gap`, but only to exclude noisy accuracy measurements — **not** used as a dark-vessel suspicion signal.
- **Trajectory prediction** — 2-layer LSTM (`trajectory.py:LSTMTrajectoryModel`), predicts next single position from an 8-ping window.
- **Vessel-to-slick attribution** — `poseatsea/fusion.py:attribute()`, a transparent rule-based weighted sum: proximity 0.40, anomaly 0.30, dwell 0.20, deviation 0.10 → confidence bands.
- **Frontend** — folium maps (Esri World Imagery, offline fallback) + Altair charts, 4 pages (Overview, AIS Anomaly Detection, Route Deviation, SAR Segmenter).
- **Deployment** — `poseatsea/registry.py` lazy model loading, FastAPI service (`api/main.py`), Hugging Face Hub weight fallback (`poseatsea/weights.py`).

**The single biggest gap, stated in the code itself:** `fusion.py`'s own docstring says *"It does NOT hindcast. There is no ocean-current or wind-drift model anywhere in this project... [the straight-line proximity] assumption degrades badly as a slick ages."* There is no weather/ocean-current data anywhere in the codebase, no drift physics, no slick age/weathering model, no multi-date SAR comparison, and no PDF/forensic report export.

> Note (positioning, not a technique source): `ashishkhedkar/OASIS` is a **competing** SIH26143 team's repo (design-only as of audit), using OceanParcels for drift. Worth tracking, not worth copying from.

---

## 2. Table A — ML / DL / Scientific Pipeline (priority: HIGH → LOW)

| Priority | Missing Feature | SIH Requirement | Repository / Technique | What We Add to POSEatSea | Exact Implementation Idea | Expected Impact | Time | Dependencies | Milestone / Definition of Done |
|---|---|---|---|---|---|---|---|---|---|
| **HIGH** | Backward drift hindcasting (spill origin estimation) | Core "vessel attribution" ask — currently zero physics-based origin estimate | OpenDrift `OpenOil` — `basemodel.run()` supports backward simulation via negative timestep (time-reversal of advection); weathering is auto-skipped on backward runs | New `poseatsea/drift/hindcast.py` wrapping OpenOil, seeded from slick centroid/polygon at detection time | `pip install opendrift`; `readers.reader_netCDF_CF_generic` for wind/current fields; `seed_elements()` at slick centroid; `run(time_step=-900, duration=timedelta(hours=-48))`; export particle cloud per timestep as origin-candidate polygons | Replaces straight-line proximity in `fusion.py` with a physically grounded origin estimate — the core scientific gap | 40–60h | Wind/current data feed (row below); `opendrift` + netCDF4/cartopy install | Given a slick polygon + timestamp, hindcast returns an origin-candidate polygon consumable by `fusion.py` |
| **HIGH** | Weather/ocean current data integration | Needed for drift hindcast + wind-based look-alike rejection | ERA5/CMEMS reanalysis, or `nansencenter/openwind`'s `SARWind` class + `cmod5n_inverse` (CMOD5 geophysical model) for SAR-derived wind speed | `poseatsea/data/ocean_weather.py` fetching wind (10m u/v) + surface current (u/v) grids for scenario bbox/time window | `cdsapi` (ERA5) or Copernicus Marine client call, cached to disk for demo reliability (avoid live-API dependency during judging) | Unblocks both drift hindcasting and wind-validated look-alike rejection | 16–24h | API keys (CDS/CMEMS); pre-fetch cache for offline demo | `get_wind_current(lat, lon, time)` returns a usable grid |
| **HIGH** | SAR-only vessel with no AIS match ("dark vessel" flag) | Directly targets AIS-evasion / dark-vessel attribution — cheap, reuses existing outputs | None needed — pure geometry on existing `SarResult.ships` | Extend `sar_view.py:_vessel_crossref()` to explicitly flag unmatched SAR ship detections | `cv2.minAreaRect` on Ship-class connected components → length/heading estimate; compare position/heading against AIS tracks within a radius/time window; flag "no AIS signal" | Likely the single most compelling piece of forensic evidence for judges — physical presence with no broadcast | 8–12h | None | SAR ship list cross-referenced against AIS; unmatched detections surfaced as evidence rows |
| **HIGH** | SAR look-alike rejection via wind validation | Reduces false positives — the model's own `confidence_note()` already flags look-alikes as ambiguous | `nansencenter/openwind` `SARWind` / CMOD5 (low wind speed, roughly <3 m/s, is where biogenic look-alikes form) | Post-hoc filter on oil-class polygons using co-located wind speed | Sample wind field at slick centroid; if below threshold, downgrade confidence in `SarResult.confidence_note()` | Directly cuts false-positive oil detections — a core judging criterion | 8–12h | Weather row above | Wind-validated confidence flag attached per slick |
| MED-HIGH | Class-balanced / focal+dice segmentation loss | Krestenitis dataset is class-imbalanced (5 classes: sea/oil/look-alike/ship/land); current training loss unverified from inspectable code (only inference code is in-repo) | MMSegmentation `focal_loss.py`, `dice_loss.py`, `ohem_cross_entropy_loss.py` | Retrain existing U-Net + `mit_b2` with a focal+dice combo | New `poseatsea/train/losses.py`; retrain, compare per-class IoU, especially oil and look-alike | Better recall on minority classes → fewer missed/mislabeled slicks | 12–20h | Existing training notebook (not audited here), GPU | New checkpoint beats baseline mIoU on oil + look-alike classes |
| MED-HIGH | Repurpose AIS blackout as a suspicion signal | `coverage_gap` currently only excludes noisy trajectory-accuracy measurements — not used as evidence | GFW methodology (conceptual reuse only, not code) | New `dark_vessel_score` component in `fusion.py` | Flag gaps that bracket the slick's location/time within drift-uncertainty radius; add ~0.10–0.15 weight | Turns an already-computed value into a named attribution feature — "vessel went dark near spill" | 10–16h | None (bonus if hindcast row exists, for spatial gating) | `fusion.py` score includes a dark-vessel component, tested on a synthetic gap |
| MED-HIGH | GFW-style loitering / encounter detector | Distinguishes rendezvous/illegal-transfer behavior near a spill from ordinary transit | GFW published thresholds — loitering: avg speed <2kn, ≥20nm offshore; encounter: two vessels <500m for ≥2h at <2kn, positions interpolated onto a 10-minute grid | `poseatsea/inference/loitering.py`, separate from the general autoencoder anomaly score | Rolling-speed loitering flag; pairwise-distance encounter detector on the interpolated AIS grid | Interpretable, named evidence — stronger forensic story than a black-box anomaly score alone | 12–18h | None | Loitering/encounter events computed and surfaced as evidence items |
| MEDIUM | Origin-probability uncertainty ensemble | Communicates confidence in the origin estimate, not a false-precision single point | PyGNOME's uncertain `SpillContainerPair` — a duplicated particle ensemble with randomized weathering/movement rates (`gnome/model.py`, `gnome/weatherers/`) | Run the drift hindcast as N=20–50 perturbed runs (±10–20% wind/current, ± detection-time jitter) | Rasterize particle density across runs into an origin-probability raster | Defensible probabilistic evidence instead of overclaiming precision | 10–16h | Drift hindcast row | Origin-probability raster produced, feeds the Table B heatmap |
| MEDIUM | Multi-temporal SAR change detection | Reduces false positives from persistent dark features (calm water, land shadow) | `eo-learn` `EOPatch` / `EOTask` / `EOWorkflow` — pipeline infrastructure, not an algorithm itself | Two-scene workflow diffing oil/look-alike masks between dates | Build a minimal `EOWorkflow`, difference class masks between two dated scenes, flag persistent look-alikes as non-spill | Moderate — valuable but blocked without multi-date imagery for the demo scenario | 16–24h + data acquisition | A second dated SAR scene for the same AOI (may not exist for the fixed Wakashio scenario) | Proof-of-concept change mask for at least one before/after pair |
| LOW | Slick age / weathering estimate | Supporting forensic detail — is observed slick state consistent with the claimed spill time? | OpenOil `evaporation_noaa()` / `emulsification_noaa()` (ported from PyGNOME / NOAA ADIOS oil database) | Run OpenOil weathering functions given oil type + elapsed time from candidate origin | Compare predicted weathered fraction/area to observed slick area | Nice-to-have; fully dependent on hindcast existing, scientifically approximate | 12–20h | Drift hindcast + weather rows | Age/weathering estimate shown as an auxiliary metric |
| LOW | Coastal vulnerability model | Replaces the hardcoded 5km reef ring with a real computed index | InVEST `coastal_vulnerability` model (`src/natcap/invest/coastal_vulnerability/coastal_vulnerability.py`, has its own `MODEL_SPEC`) | Run InVEST offline for the scenario coastline, export as a layer | Requires shoreline/bathymetry/habitat GIS data — real data-wrangling cost | Demo polish + credibility; not urgent vs. core detection/attribution gaps | 16–24h | Coastline/bathymetry/habitat GIS data for scenario AOI | Vulnerability layer replaces the static ring |
| LOW | Transformer trajectory prediction | Multi-modal prediction (vessel could turn either way) vs. single-point LSTM | TrAISformer (`CIA-Oceanix/TrAISformer`, arXiv:2109.03958) — discretizes AIS state into a "four-hot" vector, transformer predicts next discretized state | Optional upgrade path, not a replacement | Discretize lat/lon/speed/course into bins, train a small transformer, output top-k modes | Marginal for the SIH demo — current LSTM already ships with a documented model card | 30–50h | Sufficient AIS training volume, GPU | Transformer beats LSTM on held-out median displacement error |

**Deliberately excluded (not verified, not guessed):** xView3-reference's specific detection architecture/loss — its own README states the training notebook "is not intended to recommend a particular strategy or approach," so no concrete architecture claim is sourced. `krestenitis/oil-spill-detection-dataset`'s exact baseline model/loss — the original repo could not be reached to confirm.

---

## 3. Table B — Streamlit / Frontend / Demo Features (priority: HIGH → LOW)

| Priority | New Streamlit Feature | Purpose | Data / Backend Required | UI Implementation | SIH Demo Impact | Time | Milestone / Definition of Done |
|---|---|---|---|---|---|---:|---|
| **HIGH** | Forensic PDF report export | SIH deliverable needs a tangible, downloadable case output | Existing `fusion.py` evidence + attribution + SAR result + map snapshot | "Generate Forensic Report" button, `reportlab`/WeasyPrint compiling incident summary, evidence list, attribution table, map snapshot | High — a downloadable PDF is often the literal graded artifact | 16–24h | Click a button → downloadable PDF summarizing the full case |
| **HIGH** | Origin-probability heatmap layer | Visualize the backward-drift ensemble spatially instead of a single line | Drift hindcast + uncertainty ensemble (Table A) | folium `HeatMap` layer, opacity slider (matches existing SAR overlay pattern) | High — the core visual payoff of "here's where it came from" | 8–12h (once backend exists) | Heatmap toggle on the map showing the probable origin zone |
| **HIGH** | Dark-vessel / SAR-AIS cross-reference view | Surface SAR ships with no AIS match, and tracks intersecting the origin zone | SAR dark-vessel flag + hindcast polygon (Table A) | Map badges for unmatched SAR ships; highlight vessel tracks crossing the origin-probability zone | High — the "smoking gun" visualization for attribution | 12–16h | Map clearly distinguishes AIS-matched vs. unmatched SAR vessel detections |
| MED-HIGH | AIS time-slider + animated trajectory playback | Currently only static polylines — no way to scrub through time | None new — AIS ping data already loaded via `real_ais.py` | `st.slider` bound to timestamp, redraw `traffic_map` per frame, or folium `TimestampedGeoJson` for playback | High — a standard expected feature in maritime-tracking demos, currently fully absent | 10–16h | Time slider scrubs vessel positions; optional animated play button |
| MEDIUM | Weather/current overlay | Show wind/current vectors used in the drift story, for transparency | Weather/current data feed (Table A) | folium vector/arrow or wind-barb layer, toggle alongside existing layers | Medium — supports scientific credibility of the drift narrative | 8–12h (once data exists) | Wind/current arrows render on the map for the scenario time window |
| MEDIUM | Before/after SAR imagery comparison | Supports multi-temporal change detection visually | Multi-temporal change detection (Table A) | Extend `sar_view.py` tabs with a swipe/slider comparison (e.g. `streamlit-image-comparison`) | Medium — good if multi-date imagery is available, otherwise blocked | 6–10h (UI only) | Swipeable before/after comparison for a scenario with 2+ dated scenes |
| MEDIUM | Candidate-vessel side-by-side comparison | `attribution_bars` shows all vessels but no head-to-head evidence table | None new — existing `fusion.py` output | Multiselect 2–3 MMSIs → side-by-side evidence table (proximity/anomaly/dwell/deviation/dark-vessel columns) | Medium — helps narrate "why vessel A over vessel B" during judging Q&A | 6–10h | Select 2 vessels, see evidence side by side |
| LOW | Graduated coastal-risk overlay | Replace the hardcoded 5km reef circle with a real computed vulnerability layer | Coastal vulnerability model (Table A) | Choropleth/graduated color overlay along the coastline instead of a fixed circle | Low-medium — polish item | 6–10h (once data exists) | Coastline colored by vulnerability index instead of a fixed ring |
| LOW | Generalize incident timeline | "Sequence" card is hardcoded to the single Wakashio `INCIDENT` dict — not reusable for other scenarios | None new — restructure the existing scenario config pattern | Derive timeline entries from detection timestamp + `fusion.py` outputs generically, not a fixed dict | Low for the one rehearsed demo scenario; raises credibility if judges ask "does this only work for Mauritius?" | 6–10h | Timeline card renders from any processed scenario, not just Wakashio |

---

## 4. Roadmap

### M1 — Critical SIH Gap
**Features:** OpenDrift/OpenOil backward hindcast; weather/current data connector; SAR-vs-AIS dark-vessel flag; wind-based look-alike rejection. *(Table A, rows 1–4)*
**Hours:** ~110–160h
**Dependencies:** `opendrift` + netCDF4/cartopy install; wind/current data source (CDS/CMEMS API or pre-cached for demo)
**Expected output:** Given a detected slick, the system produces a physics-grounded origin-probability zone and flags vessels present in SAR with no matching AIS signal — replacing the current pure-heuristic proximity score with a scientifically defensible attribution chain.

### M2 — Attribution Improvement
**Features:** Dark-vessel + loitering/encounter signals folded into `fusion.py`; drift-derived origin probability replacing straight-line proximity; uncertainty ensembles. *(Table A, rows 5–8)*
**Hours:** ~50–70h
**Dependencies:** M1 for spatial gating on some components; the loitering/blackout detector is independent and can start immediately.
**Expected output:** `fusion.py`'s attribution score becomes multi-evidence and independently auditable line-by-line — physics-based proximity, dark-vessel presence, loitering/encounter events, AIS blackout timing — instead of a single distance number.

### M3 — Prediction / Response
**Features:** Segmentation loss retraining (class-balanced/focal+dice); multi-temporal change detection; slick age/weathering; coastal vulnerability model. *(Table A, rows 4b, 9–11)*
**Hours:** ~55–80h
**Dependencies:** Training GPU access; multi-date SAR imagery for change detection (a real blocker if unavailable for the demo scenario); InVEST GIS inputs for coastal vulnerability.
**Expected output:** Improved segmentation accuracy on minority classes (oil/look-alike), false-positive reduction where multi-date data allows, and a scientifically grounded coastal-risk layer.

### M4 — Demo / Streamlit
**Features:** PDF forensic report; origin-probability heatmap; dark-vessel/AIS cross-reference view; AIS time-slider + playback; weather overlay; before/after imagery; candidate comparison; coastal-risk overlay upgrade; timeline generalization. *(all of Table B)*
**Hours:** ~80–110h
**Dependencies:** Several items need M1–M3 backends first (the heatmap needs the drift ensemble; the weather overlay needs the data feed; before/after needs multi-date SAR).
**Expected output:** A demo-ready dashboard that visually narrates the full forensic chain — detection → origin → dark-vessel flag → evidence → exportable report — instead of today's disconnected per-page views.

**Total estimated effort:** ~295–420h across M1–M4.
