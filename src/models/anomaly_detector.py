"""
Isolation Forest anomaly detector on machine telemetry sensor data.

Trained on baseline (non-degraded) sensor readings. Flags hourly readings
that deviate significantly from expected machine behavior. Results are
written back to the machine_telemetry table (anomaly_flag, anomaly_score).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

SENSOR_FEATURES = [
    "temperature_c", "vibration_hz", "pressure_bar", "power_kw", "rpm"
]

MODEL_DIR = Path("models")


def train(
    telemetry: pd.DataFrame,
    cfg: dict,
) -> tuple[IsolationForest, StandardScaler]:
    """Fit Isolation Forest on telemetry sensor features."""
    anom_cfg = cfg["models"]["anomaly"]

    X = telemetry[SENSOR_FEATURES].dropna()
    log.info("Training Isolation Forest on %d telemetry rows...", len(X))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=anom_cfg["n_estimators"],
        contamination=anom_cfg["contamination"],
        random_state=anom_cfg["random_state"],
        n_jobs=-1,
    )
    model.fit(X_scaled)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "isolation_forest.joblib")
    joblib.dump(scaler, MODEL_DIR / "anomaly_scaler.joblib")
    log.info("Anomaly detector saved to models/")

    return model, scaler


def score(
    telemetry: pd.DataFrame,
    model: IsolationForest,
    scaler: StandardScaler,
    cfg: dict,
) -> pd.DataFrame:
    """Score all telemetry rows. Returns DataFrame with anomaly_flag and anomaly_score."""
    tel = telemetry.copy()
    threshold = cfg["kpis"]["anomaly_score_threshold"]

    has_all = all(c in tel.columns for c in SENSOR_FEATURES)
    valid_mask = tel[SENSOR_FEATURES].notna().all(axis=1) if has_all else pd.Series(False, index=tel.index)

    scores = np.full(len(tel), 0.0)
    flags = np.zeros(len(tel), dtype=bool)

    if valid_mask.any():
        X = tel.loc[valid_mask, SENSOR_FEATURES]
        X_scaled = scaler.transform(X)
        raw_scores = model.score_samples(X_scaled)  # more negative = more anomalous
        scores[valid_mask] = raw_scores
        flags[valid_mask] = raw_scores < threshold

    tel["anomaly_score"] = scores.round(4)
    tel["anomaly_flag"] = flags

    n_anomalies = flags.sum()
    log.info(
        "Anomaly scoring complete: %d/%d rows flagged (%.2f%%)",
        n_anomalies, len(tel), 100 * n_anomalies / len(tel),
    )
    return tel


def load_model() -> tuple[IsolationForest, StandardScaler]:
    model = joblib.load(MODEL_DIR / "isolation_forest.joblib")
    scaler = joblib.load(MODEL_DIR / "anomaly_scaler.joblib")
    return model, scaler


def run_training(
    config_path: str = "config/config.yaml",
    data_dir: str = "data/raw",
    use_db: bool = True,
) -> tuple[IsolationForest, StandardScaler]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if use_db:
        from src.db import get_connection
        conn = get_connection(cfg)
        tel = pd.read_sql("SELECT * FROM machine_telemetry", conn)
        conn.close()
    else:
        tel = pd.read_csv(Path(data_dir) / "machine_telemetry.csv")

    return train(tel, cfg)
