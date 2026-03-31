"""
Tests for model training and scoring logic.
Uses small synthetic datasets — no DB required.
"""

import numpy as np
import pandas as pd

from src.features.pipeline import FEATURE_COLS, TARGET_COL

MINIMAL_CFG = {
    "models": {
        "anomaly": {"n_estimators": 10, "contamination": 0.05, "random_state": 42},
        "xgboost": {
            "n_estimators": 10, "max_depth": 3, "learning_rate": 0.1,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "scale_pos_weight": 4, "random_state": 42, "n_jobs": 1,
        },
        "lightgbm": {
            "n_estimators": 10, "max_depth": 3, "learning_rate": 0.1,
            "num_leaves": 7, "min_child_samples": 2,
            "random_state": 42, "n_jobs": 1,
        },
        "cv_splits": 3,
        "train_test_split_quantile": 0.80,
        "ensemble_weights": [0.5, 0.5],
    },
    "kpis": {"high_risk_score_threshold": 0.5, "anomaly_score_threshold": -0.1},
}


def _make_features(n: int = 200) -> pd.DataFrame:
    """Create a minimal feature DataFrame with all required columns."""
    rng = np.random.default_rng(42)
    data = {col: rng.normal(0, 1, n) for col in FEATURE_COLS}
    data["work_order_id"] = range(1, n + 1)
    data[TARGET_COL] = (rng.random(n) > 0.75).astype(int)  # ~25% positive
    return pd.DataFrame(data)


def _make_telemetry(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "machine_id": rng.integers(1, 6, n),
        "recorded_at": pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC"),
        "temperature_c": rng.normal(65, 5, n),
        "vibration_hz": rng.normal(12, 2, n),
        "pressure_bar": rng.normal(8, 1, n),
        "power_kw": rng.normal(45, 5, n),
        "rpm": rng.normal(3000, 100, n),
    })


def test_anomaly_detector_train_and_score():
    from src.models.anomaly_detector import score, train

    tel = _make_telemetry(100)
    model, scaler = train(tel, MINIMAL_CFG)

    scored = score(tel, model, scaler, MINIMAL_CFG)
    assert "anomaly_flag" in scored.columns
    assert "anomaly_score" in scored.columns
    assert scored["anomaly_flag"].dtype == bool
    assert len(scored) == len(tel)


def test_anomaly_score_range():
    from src.models.anomaly_detector import score, train

    tel = _make_telemetry(100)
    model, scaler = train(tel, MINIMAL_CFG)
    scored = score(tel, model, scaler, MINIMAL_CFG)
    # Isolation Forest scores are bounded (typically -1 to 0)
    assert scored["anomaly_score"].between(-2, 1).all()


def test_risk_classifier_train_returns_metrics():
    from src.models.risk_classifier import train

    features_df = _make_features(200)
    metrics = train(features_df, MINIMAL_CFG)

    for key in ["test_auc_roc", "test_auc_pr", "test_f1", "cv_auc_mean"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_risk_classifier_score_probabilities():
    from src.models.risk_classifier import score, train

    features_df = _make_features(200)
    train(features_df, MINIMAL_CFG)
    result = score(features_df, MINIMAL_CFG)

    assert "risk_score" in result.columns
    assert "risk_label" in result.columns
    assert result["risk_score"].between(0, 1).all()
    assert set(result["risk_label"].unique()).issubset({0, 1})
    assert len(result) == len(features_df)


def test_risk_classifier_score_count_matches():
    from src.models.risk_classifier import score, train

    features_df = _make_features(200)
    train(features_df, MINIMAL_CFG)
    result = score(features_df, MINIMAL_CFG)
    assert len(result) == 200
