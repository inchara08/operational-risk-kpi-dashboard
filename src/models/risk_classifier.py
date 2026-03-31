"""
XGBoost + LightGBM soft-voting ensemble risk classifier.

Predicts the probability that a work order will result in failure or
SLA-breaching delay. Uses TimeSeriesSplit cross-validation to prevent
temporal data leakage — a common mistake in operational ML that this
project explicitly avoids.

Model outputs:
  - risk_score  : ensemble probability (0–1)
  - risk_label  : 1 if risk_score >= threshold, else 0
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.features.pipeline import FEATURE_COLS, TARGET_COL

log = logging.getLogger(__name__)

MODEL_DIR = Path("models")


def _build_xgb(cfg: dict) -> XGBClassifier:
    xgb_cfg = cfg["models"]["xgboost"]
    return XGBClassifier(
        n_estimators=xgb_cfg["n_estimators"],
        max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"],
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        scale_pos_weight=xgb_cfg["scale_pos_weight"],
        random_state=xgb_cfg["random_state"],
        n_jobs=xgb_cfg["n_jobs"],
        eval_metric="aucpr",
        verbosity=0,
    )


def _build_lgb(cfg: dict) -> LGBMClassifier:
    lgb_cfg = cfg["models"]["lightgbm"]
    return LGBMClassifier(
        n_estimators=lgb_cfg["n_estimators"],
        max_depth=lgb_cfg["max_depth"],
        learning_rate=lgb_cfg["learning_rate"],
        num_leaves=lgb_cfg["num_leaves"],
        min_child_samples=lgb_cfg["min_child_samples"],
        random_state=lgb_cfg["random_state"],
        n_jobs=lgb_cfg["n_jobs"],
        verbose=-1,
    )


def train(
    features_df: pd.DataFrame,
    cfg: dict,
) -> dict:
    """
    Train ensemble with TimeSeriesSplit CV. Returns metrics dict.
    Models are saved to models/ directory.
    """
    n_splits = cfg["models"]["cv_splits"]
    split_q = cfg["models"]["train_test_split_quantile"]
    weights = cfg["models"]["ensemble_weights"]
    threshold = cfg["kpis"]["high_risk_score_threshold"]

    X = features_df[FEATURE_COLS].fillna(0).values
    y = features_df[TARGET_COL].values

    # Time-based train/test split at 80th percentile row index
    split_idx = int(len(X) * split_q)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    log.info(
        "Train: %d rows | Test: %d rows | Positive rate: train=%.2f%% test=%.2f%%",
        len(X_train), len(X_test),
        100 * y_train.mean(), 100 * y_test.mean(),
    )

    # Cross-validation on training set only
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_aucs = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        xgb_fold = _build_xgb(cfg)
        lgb_fold = _build_lgb(cfg)
        xgb_fold.fit(X_train[tr_idx], y_train[tr_idx])
        lgb_fold.fit(X_train[tr_idx], y_train[tr_idx])
        fold_proba = (
            weights[0] * xgb_fold.predict_proba(X_train[val_idx])[:, 1]
            + weights[1] * lgb_fold.predict_proba(X_train[val_idx])[:, 1]
        )
        auc = roc_auc_score(y_train[val_idx], fold_proba)
        cv_aucs.append(auc)
        log.info("  Fold %d AUC-ROC: %.4f", fold, auc)

    log.info("CV mean AUC-ROC: %.4f ± %.4f", np.mean(cv_aucs), np.std(cv_aucs))

    # Train final models on full training set
    log.info("Training final XGBoost model...")
    xgb_final = _build_xgb(cfg)
    xgb_final.fit(X_train, y_train)

    log.info("Training final LightGBM model...")
    lgb_final = _build_lgb(cfg)
    lgb_final.fit(X_train, y_train)

    # Evaluate on held-out test set
    xgb_proba = xgb_final.predict_proba(X_test)[:, 1]
    lgb_proba = lgb_final.predict_proba(X_test)[:, 1]
    ensemble_proba = weights[0] * xgb_proba + weights[1] * lgb_proba

    auc_roc = roc_auc_score(y_test, ensemble_proba)
    auc_pr = average_precision_score(y_test, ensemble_proba)
    preds = (ensemble_proba >= threshold).astype(int)
    f1 = f1_score(y_test, preds, zero_division=0)

    metrics = {
        "cv_auc_mean": float(np.mean(cv_aucs)),
        "cv_auc_std": float(np.std(cv_aucs)),
        "test_auc_roc": float(auc_roc),
        "test_auc_pr": float(auc_pr),
        "test_f1": float(f1),
        "test_n": int(len(y_test)),
        "test_positive_rate": float(y_test.mean()),
        "threshold": threshold,
        "feature_names": FEATURE_COLS,
        "ensemble_weights": weights,
    }

    log.info(
        "Test metrics: AUC-ROC=%.4f | AUC-PR=%.4f | F1=%.4f",
        auc_roc, auc_pr, f1,
    )

    # Save models and metrics
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(xgb_final, MODEL_DIR / "xgboost_classifier.joblib")
    joblib.dump(lgb_final, MODEL_DIR / "lightgbm_classifier.joblib")

    Path("results").mkdir(exist_ok=True)
    with open("results/model_metrics.json", "w") as fp:
        json.dump(metrics, fp, indent=2)
    log.info("Models and metrics saved.")

    return metrics


def score(
    features_df: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """Load saved models and score the full feature set."""
    weights = cfg["models"]["ensemble_weights"]
    threshold = cfg["kpis"]["high_risk_score_threshold"]

    xgb_model = joblib.load(MODEL_DIR / "xgboost_classifier.joblib")
    lgb_model = joblib.load(MODEL_DIR / "lightgbm_classifier.joblib")

    X = features_df[FEATURE_COLS].fillna(0).values
    xgb_proba = xgb_model.predict_proba(X)[:, 1]
    lgb_proba = lgb_model.predict_proba(X)[:, 1]
    ensemble_proba = weights[0] * xgb_proba + weights[1] * lgb_proba

    result = features_df[["work_order_id"]].copy()
    result["risk_score"] = ensemble_proba.round(4)
    result["risk_label"] = (ensemble_proba >= threshold).astype(int)

    log.info(
        "Scored %d work orders | High-risk: %d (%.1f%%)",
        len(result),
        result["risk_label"].sum(),
        100 * result["risk_label"].mean(),
    )
    return result


def run_training(
    config_path: str = "config/config.yaml",
    processed_dir: str = "data/processed",
) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    features_path = Path(processed_dir) / "features.parquet"
    log.info("Loading features from %s...", features_path)
    features_df = pd.read_parquet(features_path)

    return train(features_df, cfg)
