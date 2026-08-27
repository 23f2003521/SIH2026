# Maritime Oil Spill Detection & Vessel Attribution — Build Spec

**Read this entire document before writing any code.** This is the complete, self-contained specification for building the prototype application for SIH Problem Statement 26143 (NTRO — Leveraging satellite imagery and AIS data to detect oil spills and attribute them to the responsible vessel). Three trained ML models are provided as ready-to-use weight files; your job is to build the application around them, not to train or modify the models themselves.

---

## 1. Project Context (read this first)

The problem statement asks for a pipeline that: (a) detects oil spills in satellite radar imagery, (b) traces a detected spill back toward its likely origin, and (c) identifies which vessel is likely responsible using AIS (Automatic Identification System — the GPS-like tracking data ships broadcast) traffic data.

**What has been built and handed to you:** two of the three pieces from that pipeline, fully trained and ready for inference:
1. A model that takes a satellite radar image and identifies which pixels are oil spill, look-alike (false positive), ship, land, or open sea.
2. A model that flags whether a given ship's AIS behavior looks anomalous (e.g. consistent with running aground / discharging oil).

A third capability — predicting a vessel's next position from its recent movement history — is also provided as a **standalone feature**, not wired into the anomaly detector (see Section 5 for why, and Section 7 for how to use it anyway in the UI).

**What is explicitly NOT built and NOT in scope for this prototype:** tracing a spill backward to its origin point/time using ocean current and wind data ("hindcasting"). Do not attempt to build this. If the UI needs a placeholder for it, a static "coming soon" panel is fine — do not fabricate a working feature.

**Your job:** build a working demo application that lets a user (1) upload a SAR image and see it segmented, (2) input or select AIS vessel data and see anomaly flags, and (3) see a vessel's predicted next position vs. its actual path. Tie these into one coherent UI. Exact tech stack recommendation is in Section 6, but the priority is a **working, demoable app**, not a particular framework.

---

## 2. Files provided and what each one is

| File | What it is | Required to run |
|---|---|---|
| `best_sar_model.pth` | SAR segmentation model weights | Yes |
| `trajectory_lstm_baseline.pth` | Trajectory prediction model weights | Yes |
| `ais_phase1_autoencoder.pth` | AIS anomaly detection model weights | Yes |
| `ais_phase1_scaler.joblib` | Feature scaler that MUST be used alongside the AIS model — the model was trained on scaled features and will produce meaningless output on raw, unscaled input | Yes |

None of these files are human-readable or directly inspectable — they are PyTorch/scikit-learn binary serializations. You load them programmatically using the exact code in Sections 3–5 below. **Do not attempt to open, edit, or "look inside" these files** — treat them as opaque binaries and only interact with them through `torch.load()` / `joblib.load()` as shown.

**A note on file extensions:** if any of these files arrive with a `.zip` extension instead of `.pth`, this is a known artifact of how Kaggle's download button labels PyTorch checkpoint files — they are not actually zip archives to be extracted. Simply rename the extension back to `.pth`; the file's byte content is already a valid PyTorch checkpoint (PyTorch's `.pth` format is internally zip-based, which is why this mislabeling happens). Do not extract/unzip them.

---

## 3. Component 1 — SAR Oil Spill Segmentation

### What it does, in plain terms
Takes one satellite radar (SAR) image of the ocean and produces a same-purpose output image where every pixel is classified into one of 5 categories. This tells you *where* in the image oil is, how much of it there is, and rules out common false positives (calm-water "look-alikes" that resemble oil in radar but aren't).

### The 5 output classes
| Class index | Meaning | Suggested display color |
|---|---|---|
| 0 | Sea Surface (normal open water) | Black |
| 1 | **Oil Spill** (the target class) | Cyan `(0,255,255)` |
| 2 | Look-alike (false positive — calm water, algae, etc. that resembles oil in radar) | Red `(255,0,0)` |
| 3 | Ship | Brown `(153,76,0)` |
| 4 | Land | Green `(0,153,0)` |

These exact colors match the convention used in the original training dataset (Krestenitis et al. Sentinel-1 oil spill benchmark) — use them for visual consistency if you're building a colored overlay.

### Dependencies
```
torch
segmentation-models-pytorch
opencv-python
numpy
```

### Exact loading and inference code
```python
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_CLASSES = 5
CLASS_NAMES = ['Sea Surface', 'Oil Spill', 'Look-alike', 'Ship', 'Land']
CLASS_COLORS_RGB = {
    0: (0, 0, 0),
    1: (0, 255, 255),
    2: (255, 0, 0),
    3: (153, 76, 0),
    4: (0, 153, 0),
}

def load_sar_model(weights_path='best_sar_model.pth'):
    # encoder_name is HARDCODED to 'mit_b2' -- this is the confirmed winning
    # architecture from training. Do not change this or the weights will
    # fail to load (architecture must match exactly what was trained).
    model = smp.Unet(
        encoder_name='mit_b2',
        encoder_weights=None,   # we're loading OUR trained weights, not ImageNet ones
        in_channels=3,
        classes=NUM_CLASSES
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_sar_oil_spill(model, image_path):
    """
    Input: path to a raw SAR .jpg image, any original resolution.
    Output: 2D numpy array, shape (512, 512), integer values 0-4 (see class table above).
             NOTE: output is ALWAYS 512x512 regardless of input image size -- the
             function resizes internally. This MUST be 512x512, matching the resolution
             the model was trained at (see OilSpillDataset512 in the training notebook) --
             running inference at a different resolution (e.g. 256x256) produces garbage,
             incoherent noise output, not a slightly-worse result. This is not a tunable
             parameter -- do not change it.
             If you need to overlay this on the original image, resize this mask back up
             to the original image's dimensions using
             cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST) --
             use NEAREST, not linear/cubic, or you will invent fractional class values
             that don't correspond to any real class.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512)).astype(np.float32) / 255.0  # MUST be 512x512, see note above
    image = np.transpose(image, (2, 0, 1))
    tensor_img = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor_img)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    return pred_mask

def mask_to_rgb(mask):
    """Convert the (512,512) integer class mask into a viewable (512,512,3) RGB image."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS_RGB.items():
        rgb[mask == cls] = color
    return rgb

def estimate_oil_spill_area_km2(mask, pixel_ground_resolution_m=10):
    """
    Rough area estimate: counts Oil Spill (class 1) pixels and converts to km^2.
    pixel_ground_resolution_m defaults to 10m (Sentinel-1's native resolution per
    the reference dataset), but since the mask is resized to 512x512 from the
    original image, this is an APPROXIMATION, not an exact measurement -- flag this
    clearly in the UI (e.g. "~X km^2, approximate") rather than presenting it as precise.
    """
    oil_pixel_count = (mask == 1).sum()
    km2_per_pixel = (pixel_ground_resolution_m / 1000) ** 2
    return oil_pixel_count * km2_per_pixel

# --- Example usage ---
# model = load_sar_model('best_sar_model.pth')
# mask = predict_sar_oil_spill(model, 'path/to/some_sar_image.jpg')
# rgb_overlay = mask_to_rgb(mask)
# area = estimate_oil_spill_area_km2(mask)
```

### Test data
If you need sample SAR images to test with and don't have real ones on hand, request the original training/test dataset from the team ("Oil Spill Detection Dataset" — Krestenitis et al. Sentinel-1 benchmark, 1002 train / 110 test images). Any `.jpg` from the `test/images/` folder is a valid, realistic input.

---

## 4. Component 2 — AIS Trajectory Prediction

### What it does, in plain terms
Given a vessel's last 8 AIS position reports (pings), predicts where that vessel's *next* ping will be. Comparing this prediction to the vessel's actual next reported position tells you how "predictable" vs. "erratic" its movement is — useful as a standalone visualization ("here's where we expected this ship to be vs. where it actually went") even though it isn't currently wired into the anomaly detector (see Section 5).

### Dependencies
```
torch
numpy
pandas
```

### Critical: normalization constants
The model was trained on lat/lon/speed normalized to a 0–1 range based on the specific geographic bounding box and speed distribution of the training data (Mauritius AOI, July 2020). **These exact constants must be used at inference time or predictions will be meaningless.** They were computed as follows and are given here as fixed values — do not attempt to recompute them from different data:

```python
LAT_MIN = -20.565386666666665
LAT_MAX = -20.05773333333333
LON_MIN = 57.725333333333325
LON_MAX = 58.37872
SPEED_MAX = 19.3   # 99.9th percentile of training speed data, used as a clip ceiling
SEQ_LEN = 8         # the model expects EXACTLY 8 historical pings, no more, no fewer
```

**Important caveat for whoever builds the UI:** these bounds are specific to the Mauritius AOI used in training. This model will only produce sensible predictions for vessels operating within (or very close to) this same geographic area. If the demo needs to work with vessels/regions outside this box, that requires retraining on different data — flag this to the team rather than silently producing nonsense predictions for out-of-region inputs.

### Exact loading and inference code
```python
import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMTrajectoryModel(nn.Module):
    """
    Architecture MUST match this exactly -- these are the confirmed dimensions
    from the saved weights file (verified: lstm.weight_ih_l0 shape (512,6),
    hidden_dim=128, num_layers=2, output dim=2).
    """
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

def load_trajectory_model(weights_path='trajectory_lstm_baseline.pth'):
    model = LSTMTrajectoryModel()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def normalize_features(seq):
    """seq: numpy array shape (1, 8, 5) -- columns are (lat, lon, speed, course, rot)."""
    out = seq.copy().astype(np.float32)
    out[..., 0] = (seq[..., 0] - LAT_MIN) / (LAT_MAX - LAT_MIN + 1e-9)
    out[..., 1] = (seq[..., 1] - LON_MIN) / (LON_MAX - LON_MIN + 1e-9)
    out[..., 2] = np.clip(seq[..., 2], 0, SPEED_MAX) / SPEED_MAX
    course_rad = np.radians(seq[..., 3])
    sin_c, cos_c = np.sin(course_rad), np.cos(course_rad)
    rot_norm = np.clip(seq[..., 4], -128, 128) / 128.0
    return np.concatenate([out[..., :3], sin_c[..., None], cos_c[..., None], rot_norm[..., None]], axis=-1)

def denorm_latlon(lat_norm, lon_norm):
    lat = lat_norm * (LAT_MAX - LAT_MIN) + LAT_MIN
    lon = lon_norm * (LON_MAX - LON_MIN) + LON_MIN
    return lat, lon

def predict_next_position(model, history_df):
    """
    Input: history_df -- a pandas DataFrame of EXACTLY 8 rows (a vessel's last 8 AIS
           pings, in chronological order -- oldest first, most recent last), with
           columns ['latitude', 'longitude', 'speed', 'course', 'rot'].
    Output: (predicted_lat, predicted_lon) -- a tuple of two floats.
    """
    assert len(history_df) == 8, f"history_df must have exactly 8 rows, got {len(history_df)}"
    seq = history_df[['latitude','longitude','speed','course','rot']].values[None, ...]
    x = torch.tensor(normalize_features(seq), dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(x).cpu().numpy()
    return denorm_latlon(pred[0,0], pred[0,1])

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km -- use this to compare predicted vs. actual position."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

# --- Example usage ---
# model = load_trajectory_model('trajectory_lstm_baseline.pth')
# # history_df must have exactly 8 rows, sorted oldest-to-newest
# pred_lat, pred_lon = predict_next_position(model, history_df)
# actual_lat, actual_lon = 21.45, -71.2   # the vessel's real next reported position
# deviation_km = haversine_km(pred_lat, pred_lon, actual_lat, actual_lon)
```

### Measured accuracy (for display / documentation, not required for the code to run)
On vessels the model never saw during training: average error 0.37 km, median error 0.19 km, 90th-percentile error 0.63 km.

---

## 5. Component 3 — AIS Anomaly Detection

### What it does, in plain terms
Given a single AIS ping's kinematic data (speed, heading, turn rate, and how much those changed from the previous ping), flags whether that ping looks behaviorally anomalous — the kind of pattern consistent with a vessel running aground, drifting uncontrolled, or otherwise behaving outside normal operation. This is the model that answers "is this vessel doing something suspicious right now."

**Important:** this model does NOT take a trajectory-prediction feature as input (an earlier version tested that and it underperformed — see the note below). It only needs the 11 features listed in the next section.

### Dependencies
```
torch
joblib
scikit-learn   # needed to unpickle the scaler -- see version note below
numpy
pandas
```

**Version note:** the scaler (`ais_phase1_scaler.joblib`) was saved with scikit-learn 1.6.1. If your environment has a different scikit-learn version, you will likely see an `InconsistentVersionWarning` when loading it — this is a warning, not a fatal error, and the scaler still functions correctly. If you want to eliminate the warning entirely, pin `scikit-learn==1.6.1` in your requirements.

### The 11 required input features, in this exact order
This order matters — the model and scaler expect a fixed-position array, not named columns:

```python
FEATURE_ORDER = ['speed', 'course', 'rot', 'msg_type', 'status', 'accuracy',
                  'course_diff', 'rot_diff', 'speed_diff', 'lat_diff', 'long_diff']
```

| Feature | Meaning | How to compute |
|---|---|---|
| `speed` | Vessel speed (knots) | Direct from AIS ping |
| `course` | Heading (degrees, 0-360) | Direct from AIS ping |
| `rot` | Rate of turn | Direct from AIS ping |
| `msg_type` | AIS message type code | Direct from AIS ping |
| `status` | AIS navigation status code | Direct from AIS ping |
| `accuracy` | AIS position accuracy flag | Direct from AIS ping |
| `course_diff` | Change in course since the vessel's previous ping | `current_course - previous_course` (0 if this is the vessel's first ping) |
| `rot_diff` | Change in rate-of-turn since previous ping | Same pattern |
| `speed_diff` | Change in speed since previous ping | Same pattern |
| `lat_diff` | Change in latitude since previous ping | Same pattern |
| `long_diff` | Change in longitude since previous ping | Same pattern |

If working from a pandas DataFrame of multiple pings per vessel, the `_diff` columns can be computed in one line each:
```python
df = df.sort_values(['mmsi', 'timestamp'])
df['course_diff'] = df.groupby('mmsi')['course'].diff().fillna(0)
df['rot_diff']    = df.groupby('mmsi')['rot'].diff().fillna(0)
df['speed_diff']  = df.groupby('mmsi')['speed'].diff().fillna(0)
df['lat_diff']    = df.groupby('mmsi')['latitude'].diff().fillna(0)
df['long_diff']   = df.groupby('mmsi')['longitude'].diff().fillna(0)
```

### Exact loading and inference code
```python
import torch
import torch.nn as nn
import joblib
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FEATURE_ORDER = ['speed', 'course', 'rot', 'msg_type', 'status', 'accuracy',
                  'course_diff', 'rot_diff', 'speed_diff', 'lat_diff', 'long_diff']

class Autoencoder(nn.Module):
    """
    Architecture MUST match this exactly -- confirmed from saved weights:
    input_dim=11, latent_dim=4, hidden layers 16 -> 8 -> 4 -> 8 -> 16.
    """
    def __init__(self, input_dim=11, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8), nn.ReLU(),
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# The exact anomaly-score threshold determined during training/evaluation.
# A row is flagged anomalous if its reconstruction error is >= this value.
AE_THRESHOLD = 1.104481

def load_ais_anomaly_model(weights_path='ais_phase1_autoencoder.pth', scaler_path='ais_phase1_scaler.joblib'):
    model = Autoencoder(input_dim=11)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict_ais_anomaly(model, scaler, row_dict):
    """
    Input: row_dict -- a Python dict with EXACTLY the 11 keys in FEATURE_ORDER
           (see table above for what each means and how to compute it).
    Output: {'is_anomaly': bool, 'score': float}
            'score' is a raw reconstruction-error value, NOT a probability --
            higher means more anomalous. Only surface 'is_anomaly' directly in
            the UI; keep 'score' available for debugging/threshold tuning only.
    """
    x = np.array([[row_dict[f] for f in FEATURE_ORDER]])
    x_scaled = scaler.transform(x)
    with torch.no_grad():
        x_t = torch.tensor(x_scaled, dtype=torch.float32).to(device)
        recon = model(x_t).cpu().numpy()
    score = float(np.mean((x_scaled - recon) ** 2))
    return {'is_anomaly': bool(score >= AE_THRESHOLD), 'score': score}

# --- Example usage ---
# model, scaler = load_ais_anomaly_model()
# row = {
#     'speed': 0.2, 'course': 145.0, 'rot': -8.0,
#     'msg_type': 1, 'status': 0, 'accuracy': 1,
#     'course_diff': 54.0, 'rot_diff': 12.0, 'speed_diff': -9.8,
#     'lat_diff': 0.0002, 'long_diff': -0.0001
# }
# result = predict_ais_anomaly(model, scaler, row)
# # result == {'is_anomaly': True/False, 'score': <float>}
```

### Measured accuracy
On held-out data: F1 = 0.567, Precision = 0.930, Recall = 0.408. In plain terms: **when this model flags a vessel, it's usually right** (93% of flags are real anomalies, few false alarms) — but it misses more than half of true anomalies (catches about 4 in 10). This is a defensible trade-off for a first-pass screening tool where a human reviews flagged vessels rather than the system acting fully autonomously — reflect that framing in the UI (e.g. "flagged for review," not "confirmed violation").

### Why this model doesn't use the trajectory prediction feature
An earlier version added "distance between predicted and actual position" (from Component 2) as a 12th input feature, hypothesizing it would improve detection. It was tested and did not improve the primary accuracy metric, so it was dropped from the model actually being shipped. This is documented here so nobody re-adds it expecting an improvement without re-testing.

---

## 6. Suggested Application Architecture

**Recommendation: a Streamlit application.** Reasoning: fastest path to a working, demoable multi-tab UI in Python with no separate frontend build step, and the team has prior familiarity with Streamlit from earlier prototyping in this project. This is a recommendation, not a hard requirement — build with Flask+React or any other stack if there's a strong reason to, but Streamlit is the lowest-effort path to a working demo given the time constraints of this hackathon.

### Suggested structure
```
app.py                          # main Streamlit entrypoint, defines tabs/pages
models/
    best_sar_model.pth
    trajectory_lstm_baseline.pth
    ais_phase1_autoencoder.pth
    ais_phase1_scaler.joblib
sar_inference.py                # Section 3's code, as a module
trajectory_inference.py         # Section 4's code, as a module
ais_inference.py                # Section 5's code, as a module
requirements.txt
```

### Suggested `requirements.txt`
```
torch
segmentation-models-pytorch
opencv-python
scikit-learn
joblib
numpy
pandas
streamlit
matplotlib
```

### Suggested UI layout (3 tabs minimum)

**Tab 1 — SAR Oil Spill Segmentation**
- File uploader for a `.jpg` SAR image.
- On upload: run `predict_sar_oil_spill()`, show the original image and the colored segmentation mask side by side (use `mask_to_rgb()`).
- Show the estimated oil spill area (`estimate_oil_spill_area_km2()`), clearly labeled as approximate.
- Show a color legend (the 5-class table from Section 3).

**Tab 2 — AIS Anomaly Inspector**
- Either a file uploader for an AIS CSV, or a small set of pre-loaded sample vessels to pick from (ask the team for sample AIS data if none is provided).
- Let the user select a vessel and a specific ping (e.g. via a slider/dropdown over that vessel's ping history).
- Compute the 11 features (Section 5) and run `predict_ais_anomaly()`.
- Display a clear flagged/not-flagged indicator, not just a raw score. Suggested framing: green "Normal Behavior" vs. red "Flagged for Review."

**Tab 3 — Trajectory Prediction**
- For a selected vessel with at least 8 historical pings, run `predict_next_position()`.
- Plot the vessel's actual historical track, plus a marker showing the predicted next position, on a simple lat/lon scatter or map plot.
- If the vessel's actual next position is also available (e.g. testing against historical data), show both actual and predicted, with the distance between them (`haversine_km()`) displayed.

**Optional Tab 4 — Combined Incident View**
- If time permits: let a user pick a SAR result and a set of AIS-flagged vessels together, to visually tell the "here's the spill, here are the suspect vessels" story end-to-end. This is a presentation/demo aid, not a new model — it's just displaying Tabs 1–3's outputs together.

---

## 7. Explicitly out of scope — do not build these

- **Hindcasting / drift simulation** (tracing a spill backward to its origin using ocean current/wind data). Not built, not part of this handoff. If asked to add it, flag back to the team rather than improvising a fake version.
- **Real-time satellite or AIS data feeds.** All three models operate on static, already-collected data (uploaded images, historical/sample AIS records). Do not attempt to integrate live data sources unless explicitly asked.
- **Retraining or fine-tuning any of the three models.** Treat the provided weight files as final for this prototype.
- **The TrAISformer trajectory model.** A transformer-based alternative to the LSTM was tested and performed worse (average error 2.6+ km vs. the LSTM's 0.37 km) — it is not part of this handoff and should not be used even if referenced elsewhere in project history.

---

## 8. Definition of done

The prototype is complete when:
1. A user can upload a SAR image and see a correctly colored, correctly labeled segmentation result.
2. A user can select or input AIS data for a vessel and see a clear anomaly flag (not a raw unexplained number).
3. A user can see a vessel's predicted next position plotted against its actual track.
4. All three components load their respective model files successfully with no architecture-mismatch errors (if you see a `size mismatch` or `unexpected key` error loading any `state_dict`, you have altered one of the architecture definitions in Sections 3–5 — revert to the exact code given here).
5. **The SAR model's output looks like a coherent segmentation** — large, contiguous regions of color roughly following visible shapes in the image (a spill's dark streak, a coastline). If the output instead looks like random high-frequency salt-and-pepper noise across the whole image with no coherent shapes, **the input resolution is wrong** — it must be exactly 512×512 (Section 3). This failure mode does not throw an error; it silently produces garbage, so a visual sanity check is required, not just "did the code run without crashing."
6. The app runs end-to-end without requiring internet access or any data/service not listed in this document.
