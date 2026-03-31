"""
Model evaluation: generates SHAP beeswarm plot, confusion matrix, and AUC curve.
Saves all images to results/images/ for README and dashboard embedding.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_curve,
    auc,
)

from src.features.pipeline import FEATURE_COLS, TARGET_COL

log = logging.getLogger(__name__)
IMAGES_DIR = Path("results/images")


def generate_shap_plot(features_df: pd.DataFrame) -> None:
    """SHAP beeswarm summary plot using XGBoost model."""
    try:
        import shap
    except ImportError:
        log.warning("shap not installed — skipping SHAP plot")
        return

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    model = joblib.load("models/xgboost_classifier.joblib")
    X = features_df[FEATURE_COLS].fillna(0)

    # Use a sample of 5000 rows for speed
    sample = X.sample(min(5000, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, sample,
        feature_names=FEATURE_COLS,
        show=False,
        max_display=15,
        plot_type="dot",
    )
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("SHAP summary plot saved.")


def generate_confusion_matrix(features_df: pd.DataFrame, cfg: dict) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    threshold = cfg["kpis"]["high_risk_score_threshold"]
    split_q = cfg["models"]["train_test_split_quantile"]
    weights = cfg["models"]["ensemble_weights"]

    xgb = joblib.load("models/xgboost_classifier.joblib")
    lgb = joblib.load("models/lightgbm_classifier.joblib")

    split_idx = int(len(features_df) * split_q)
    test_df = features_df.iloc[split_idx:]
    X_test = test_df[FEATURE_COLS].fillna(0).values
    y_test = test_df[TARGET_COL].values

    proba = weights[0] * xgb.predict_proba(X_test)[:, 1] + weights[1] * lgb.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Low Risk", "High Risk"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Risk Classifier — Confusion Matrix (Hold-out Test Set)", fontsize=11)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Confusion matrix saved.")


def generate_roc_curve(features_df: pd.DataFrame, cfg: dict) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    split_q = cfg["models"]["train_test_split_quantile"]
    weights = cfg["models"]["ensemble_weights"]

    xgb = joblib.load("models/xgboost_classifier.joblib")
    lgb = joblib.load("models/lightgbm_classifier.joblib")

    split_idx = int(len(features_df) * split_q)
    test_df = features_df.iloc[split_idx:]
    X_test = test_df[FEATURE_COLS].fillna(0).values
    y_test = test_df[TARGET_COL].values

    for name, model in [("XGBoost", xgb), ("LightGBM", lgb)]:
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    # Ensemble
    ens_proba = weights[0] * xgb.predict_proba(X_test)[:, 1] + weights[1] * lgb.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, ens_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2.5, linestyle="--", label=f"Ensemble (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Risk Classifier")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("ROC curve saved.")


def run_all(
    config_path: str = "config/config.yaml",
    processed_dir: str = "data/processed",
) -> None:
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    features_df = pd.read_parquet(Path(processed_dir) / "features.parquet")

    generate_shap_plot(features_df)
    generate_confusion_matrix(features_df, cfg)
    generate_roc_curve(features_df, cfg)
    log.info("All evaluation plots generated in results/images/")
