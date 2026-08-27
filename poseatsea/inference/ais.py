"""
AIS behavioural anomaly detection.

An 11-feature undercomplete autoencoder (11 -> 16 -> 8 -> 4 -> 8 -> 16 -> 11),
verified against the checkpoint's encoder/decoder weight shapes. A ping is
flagged when its reconstruction error clears AE_THRESHOLD.

The model is deliberately precision-heavy (P=0.93, R=0.41): when it fires it is
almost always right, but it misses roughly six in ten true anomalies. Every
surface in this application therefore says "flagged for review", never
"confirmed violation", and absence of a flag is never treated as an all-clear.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .. import weights
from ..config import (
    AE_INPUT_DIM,
    AE_LATENT_DIM,
    AE_PRECISION,
    AE_RECALL,
    AE_THRESHOLD,
    AIS_AE_WEIGHTS,
    AIS_SCALER,
    DEVICE,
    FEATURE_ORDER,
    NAV_STATUS,
)

# Columns a raw AIS frame must carry before features can be derived.
RAW_COLUMNS = ["mmsi", "timestamp", "latitude", "longitude",
               "speed", "course", "rot", "msg_type", "status", "accuracy"]

DIFF_SOURCES = {
    "course_diff": "course",
    "rot_diff": "rot",
    "speed_diff": "speed",
    "lat_diff": "latitude",
    "long_diff": "longitude",
}


class Autoencoder(nn.Module):
    """Shapes are fixed by the checkpoint -- do not change them."""

    def __init__(self, input_dim: int = AE_INPUT_DIM, latent_dim: int = AE_LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8), nn.ReLU(),
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def load_model(weights_path=None, scaler_path=None):
    """Return (model, scaler). The scaler is mandatory, not optional."""
    import warnings

    import joblib

    wpath = weights.resolve(weights_path or AIS_AE_WEIGHTS)
    spath = weights.resolve(scaler_path or AIS_SCALER)

    model = Autoencoder()
    model.load_state_dict(torch.load(wpath, map_location=DEVICE), strict=True)
    model.to(DEVICE)
    model.eval()

    # The scaler was pickled under scikit-learn 1.6.1; a newer runtime warns but
    # still unpickles a StandardScaler correctly. Silence only that warning.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*version.*")
        scaler = joblib.load(spath)

    if getattr(scaler, "n_features_in_", AE_INPUT_DIM) != AE_INPUT_DIM:
        raise ValueError(
            f"Scaler expects {scaler.n_features_in_} features, model expects {AE_INPUT_DIM}."
        )
    return model, scaler


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the five per-vessel difference features the model expects.

    Differences are taken within each MMSI after sorting by time, so one
    vessel's first ping never differences against another vessel's last.
    """
    out = df.copy()
    if "mmsi" not in out.columns:
        out["mmsi"] = 0
    sort_keys = ["mmsi", "timestamp"] if "timestamp" in out.columns else ["mmsi"]
    out = out.sort_values(sort_keys).reset_index(drop=True)

    for target, source in DIFF_SOURCES.items():
        if source not in out.columns:
            raise KeyError(f"Column {source!r} is required to derive {target!r}.")
        out[target] = out.groupby("mmsi")[source].diff().fillna(0.0)

    # Course is circular: a 359 -> 1 degree step is +2 degrees, not -358.
    out["course_diff"] = ((out["course_diff"] + 180.0) % 360.0) - 180.0
    return out


def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the diff features unless the frame already carries all of them."""
    if all(c in df.columns for c in FEATURE_ORDER):
        return df.copy()
    return add_derived_features(df)


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing:
        raise KeyError(f"Missing model features: {missing}")
    return df[FEATURE_ORDER].to_numpy(dtype=np.float64)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
@dataclass
class AnomalyResult:
    is_anomaly: bool
    score: float
    threshold: float = AE_THRESHOLD
    per_feature_error: Optional[Dict[str, float]] = None

    @property
    def severity(self) -> str:
        """Coarse band over the reconstruction error, for triage ordering."""
        if not self.is_anomaly:
            return "normal"
        ratio = self.score / self.threshold
        if ratio >= 5.0:
            return "critical"
        if ratio >= 2.0:
            return "high"
        return "elevated"

    def top_contributors(self, k: int = 3) -> List[tuple]:
        """Features carrying the most reconstruction error -- the 'why'."""
        if not self.per_feature_error:
            return []
        ranked = sorted(self.per_feature_error.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:k]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "is_anomaly": self.is_anomaly,
            "severity": self.severity,
            "score": round(self.score, 6),
            "threshold": self.threshold,
            "top_contributors": [
                {"feature": f, "error": round(e, 4)} for f, e in self.top_contributors()
            ],
        }


def _reconstruction_errors(model, scaler, x_raw: np.ndarray):
    """Return (per-row mean error, per-row per-feature squared error)."""
    x_scaled = scaler.transform(x_raw)
    with torch.no_grad():
        tensor = torch.tensor(x_scaled, dtype=torch.float32).to(DEVICE)
        recon = model(tensor).cpu().numpy()
    sq = (x_scaled - recon) ** 2
    return sq.mean(axis=1), sq


def score_row(model, scaler, row: Dict[str, float]) -> AnomalyResult:
    """Score a single ping supplied as a dict of the 11 features."""
    missing = [f for f in FEATURE_ORDER if f not in row]
    if missing:
        raise KeyError(f"Missing model features: {missing}")
    x = np.array([[float(row[f]) for f in FEATURE_ORDER]], dtype=np.float64)
    means, sq = _reconstruction_errors(model, scaler, x)
    score = float(means[0])
    return AnomalyResult(
        is_anomaly=bool(score >= AE_THRESHOLD),
        score=score,
        per_feature_error={f: float(sq[0, i]) for i, f in enumerate(FEATURE_ORDER)},
    )


def score_frame(model, scaler, df: pd.DataFrame) -> pd.DataFrame:
    """
    Score every ping in a frame.

    Returns a copy with `anomaly_score`, `is_anomaly` and `severity` appended,
    plus `top_feature` naming the largest error contributor for that ping.
    """
    prepared = ensure_features(df)
    x = feature_matrix(prepared)
    means, sq = _reconstruction_errors(model, scaler, x)

    out = prepared.copy()
    out["anomaly_score"] = means
    out["is_anomaly"] = means >= AE_THRESHOLD
    out["severity"] = [
        AnomalyResult(bool(s >= AE_THRESHOLD), float(s)).severity for s in means
    ]
    out["top_feature"] = [FEATURE_ORDER[i] for i in sq.argmax(axis=1)]
    return out


def vessel_rollup(scored: pd.DataFrame) -> pd.DataFrame:
    """Per-vessel summary of a scored frame, worst offenders first."""
    if scored.empty:
        return pd.DataFrame()
    grouped = scored.groupby("mmsi").agg(
        pings=("anomaly_score", "size"),
        flagged=("is_anomaly", "sum"),
        max_score=("anomaly_score", "max"),
        mean_score=("anomaly_score", "mean"),
    ).reset_index()
    grouped["flagged_pct"] = (grouped["flagged"] / grouped["pings"] * 100).round(1)
    if "vessel_name" in scored.columns:
        names = scored.groupby("mmsi")["vessel_name"].first()
        grouped["vessel_name"] = grouped["mmsi"].map(names)
    return grouped.sort_values("max_score", ascending=False).reset_index(drop=True)


def status_label(code) -> str:
    try:
        return NAV_STATUS.get(int(code), f"Status {int(code)}")
    except (TypeError, ValueError):
        return "Unknown"


def model_card() -> Dict[str, Any]:
    """Performance framing surfaced next to every verdict in the UI."""
    return {
        "precision": AE_PRECISION,
        "recall": AE_RECALL,
        "threshold": AE_THRESHOLD,
        "reading": (
            f"About {AE_PRECISION:.0%} of flags are genuine anomalies, but the model "
            f"catches only {AE_RECALL:.0%} of them. Treat a flag as strong evidence "
            f"and the absence of one as no evidence either way."
        ),
    }
