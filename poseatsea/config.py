"""
POSEatSea -- central configuration.

Every magic number that the three trained models depend on lives here, in one
place, so that no inference module can silently drift away from the values the
weights were actually trained with.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("POSEATSEA_MODELS_DIR", PROJECT_ROOT / "models"))

SAR_WEIGHTS = MODELS_DIR / "best_sar_model.pth"
TRAJECTORY_WEIGHTS = MODELS_DIR / "trajectory_lstm_baseline.pth"
AIS_AE_WEIGHTS = MODELS_DIR / "ais_phase1_autoencoder.pth"
AIS_SCALER = MODELS_DIR / "ais_phase1_scaler.joblib"


def resolve_device() -> torch.device:
    """CUDA when available, else CPU. Override with POSEATSEA_DEVICE."""
    forced = os.getenv("POSEATSEA_DEVICE")
    if forced:
        return torch.device(forced)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = resolve_device()

# --------------------------------------------------------------------------
# SAR segmentation  (Krestenitis et al. Sentinel-1 oil-spill benchmark)
# --------------------------------------------------------------------------
SAR_ENCODER = "mit_b2"          # confirmed against checkpoint patch_embed dims
# The model was trained at 512x512 (OilSpillDataset512). This is NOT a tunable
# knob: running inference at another resolution does not degrade the result, it
# returns incoherent noise. See docs/model_interface_spec.md section 3.
SAR_INPUT_SIZE = 512
NUM_CLASSES = 5

CLASS_NAMES = ["Sea Surface", "Oil Spill", "Look-alike", "Ship", "Land"]
CLASS_COLORS_RGB = {
    0: (0, 0, 0),         # sea surface
    1: (0, 255, 255),     # oil spill        <- target class
    2: (255, 0, 0),       # look-alike
    3: (153, 76, 0),      # ship
    4: (0, 153, 0),       # land
}
CLASS_DESCRIPTIONS = {
    0: "Normal open water.",
    1: "Confirmed dark-formation signature consistent with mineral oil.",
    2: "Dark formation that mimics oil in radar (low wind, algae, wave shadow).",
    3: "Vessel hull returning a bright radar signature.",
    4: "Coastal / land mass.",
}

# Sentinel-1 GRD ground resolution used by the reference dataset.
DEFAULT_PIXEL_RESOLUTION_M = 10.0

# --------------------------------------------------------------------------
# Trajectory LSTM  (Mauritius AOI, July 2020)
# --------------------------------------------------------------------------
LAT_MIN = -20.565386666666665
LAT_MAX = -20.05773333333333
LON_MIN = 57.725333333333325
LON_MAX = 58.37872
SPEED_MAX = 19.3                # 99.9th pct of training speed, used as clip ceiling
SEQ_LEN = 8                     # model consumes exactly 8 pings

TRAJ_INPUT_DIM = 6
TRAJ_HIDDEN_DIM = 128
TRAJ_NUM_LAYERS = 2

# Published held-out accuracy, surfaced in the UI so operators can calibrate trust.
TRAJ_MEAN_ERROR_KM = 0.37
TRAJ_MEDIAN_ERROR_KM = 0.19
TRAJ_P90_ERROR_KM = 0.63

# Empirically probed operating envelope (see docs/MODEL_NOTES.md). The model was
# trained on ~60 s ping cadence along the NE-SW lane; outside that it degrades.
TRAJ_NOMINAL_INTERVAL_S = 60.0
TRAJ_RELIABLE_COURSE_BANDS = ((200.0, 290.0), (20.0, 110.0))

# --------------------------------------------------------------------------
# AIS anomaly autoencoder
# --------------------------------------------------------------------------
FEATURE_ORDER = [
    "speed", "course", "rot", "msg_type", "status", "accuracy",
    "course_diff", "rot_diff", "speed_diff", "lat_diff", "long_diff",
]
AE_INPUT_DIM = 11
AE_LATENT_DIM = 4
AE_THRESHOLD = 1.104481         # reconstruction-error cut determined at training

# Held-out performance -- deliberately precision-heavy.
AE_F1 = 0.567
AE_PRECISION = 0.930
AE_RECALL = 0.408

# AIS navigational status codes (ITU-R M.1371) used for human-readable output.
NAV_STATUS = {
    0: "Under way using engine",
    1: "At anchor",
    2: "Not under command",
    3: "Restricted manoeuvrability",
    4: "Constrained by draught",
    5: "Moored",
    6: "Aground",
    7: "Engaged in fishing",
    8: "Under way sailing",
    15: "Undefined",
}
