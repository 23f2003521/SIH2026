---
title: POSEatSea — Oil Spill Detection & Vessel Attribution
emoji: 🛰️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.41.1
app_file: ui/app.py
pinned: false
license: mit
short_description: Satellite SAR + AIS fusion to detect oil spills and attribute them to vessels
---

# POSEatSea

**SIH Problem Statement 26143 (NTRO)** — detecting oil spills in satellite radar
imagery and attributing them to the responsible vessel using AIS traffic.

Three trained models behind one console:

- **SAR segmenter** — U-Net with a MiT-B2 SegFormer encoder, 5-class Sentinel-1
  segmentation (sea / oil / look-alike / ship / land).
- **AIS anomaly autoencoder** — 11-feature reconstruction-error detector,
  precision 0.93, recall 0.41.
- **Trajectory LSTM** — predicts a vessel's next AIS position from its last 8 pings.

A transparent correlation layer ties their outputs to one place and time and
ranks the vessels that could account for an observed slick.

## Data

Real AIS: a Mauritius AOI extract for July 2020, containing the **MV Wakashio**'s
own broadcasts through her grounding at Pointe d'Esny on 25 July 2020 and the six
days of navigational status 6 (aground) that followed.

## Declared limitations

- No drift or hindcast modelling — attribution assumes the slick lies where it
  was observed.
- The trajectory model is valid only inside the Mauritius AOI it was normalised for.
- The anomaly detector nominates vessels for review; with 0.41 recall it cannot
  clear one.

Full detail on the **System** page inside the app.
