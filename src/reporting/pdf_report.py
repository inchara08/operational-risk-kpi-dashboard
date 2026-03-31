"""
4-page PDF report using matplotlib.
Suitable for stakeholder delivery without requiring a browser.

Pages:
  1. Executive Summary — KPI scorecard table + OEE bar chart
  2. Risk Model Summary — metrics table + confusion matrix image
  3. Plant Performance — MTTR, defect rate, SLA breach bar charts
  4. Top High-Risk Work Orders — table of top 20 by risk score
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

log = logging.getLogger(__name__)


def _load_kpi_data(cfg: dict, use_db: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if use_db:
        from src.db import get_connection
        conn = get_connection(cfg)
        kpi = pd.read_sql(
            "SELECT k.*, p.plant_name FROM kpi_snapshots k JOIN plants p USING (plant_id)", conn
        )
        wo_top = pd.read_sql(
            "SELECT wo.work_order_id, wo.risk_score, wo.priority, wo.status, "
            "wo.product_type, wo.planned_units, p.plant_name, m.machine_code "
            "FROM work_orders wo "
            "JOIN plants p USING (plant_id) JOIN machines m USING (machine_id) "
            "WHERE wo.risk_label = 1 "
            "ORDER BY wo.risk_score DESC LIMIT 20",
            conn,
        )
        conn.close()
        return kpi, wo_top
    else:
        return pd.DataFrame(), pd.DataFrame()


def generate(cfg: dict, out_dir: str = "reports", use_db: bool = True) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    kpi, wo_top = _load_kpi_data(cfg, use_db)

    out_path = Path(out_dir) / "operational_risk_report.pdf"

    with PdfPages(out_path) as pdf:
        _page_executive_summary(pdf, kpi, cfg)
        _page_risk_model(pdf, cfg)
        _page_plant_performance(pdf, kpi)
        _page_top_risk_orders(pdf, wo_top)

    log.info("PDF report saved: %s", out_path)


def _page_executive_summary(pdf, kpi: pd.DataFrame, cfg: dict) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Operational Risk & KPI Analytics — Executive Summary", fontsize=14, fontweight="bold")

    if len(kpi) == 0:
        plt.text(0.5, 0.5, "No KPI data available", ha="center", va="center", transform=fig.transFigure)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    kpi["snapshot_date"] = pd.to_datetime(kpi["snapshot_date"])
    plant_avg = (
        kpi.groupby("plant_name")[["oee_score", "mttr_hours", "sla_breach_rate", "defect_rate"]]
        .mean()
        .round(4)
        .reset_index()
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    # OEE bar
    ax1 = fig.add_subplot(gs[0, :])
    colors = ["#e74c3c" if v < cfg["kpis"]["oee_threshold"] else "#2ecc71" for v in plant_avg["oee_score"]]
    ax1.bar(plant_avg["plant_name"], plant_avg["oee_score"], color=colors)
    ax1.axhline(cfg["kpis"]["oee_threshold"], color="red", linestyle="--", label=f"Target ({cfg['kpis']['oee_threshold']:.0%})")
    ax1.set_ylabel("OEE Score")
    ax1.set_title("Average OEE by Plant (Red = Below Target)")
    ax1.set_ylim(0, 1)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax1.legend()

    # KPI scorecard table
    ax2 = fig.add_subplot(gs[1, :])
    ax2.axis("off")
    table_data = plant_avg.copy()
    table_data["oee_score"] = (table_data["oee_score"] * 100).round(1).astype(str) + "%"
    table_data["sla_breach_rate"] = (table_data["sla_breach_rate"] * 100).round(2).astype(str) + "%"
    table_data["defect_rate"] = (table_data["defect_rate"] * 100).round(2).astype(str) + "%"
    table_data["mttr_hours"] = table_data["mttr_hours"].round(2).astype(str) + "h"
    table_data.columns = ["Plant", "OEE", "MTTR", "SLA Breach Rate", "Defect Rate"]
    tbl = ax2.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.5)
    ax2.set_title("KPI Scorecard Summary", pad=10)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_risk_model(pdf, cfg: dict) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Risk Model — Performance Summary", fontsize=14, fontweight="bold")

    metrics_path = Path("results/model_metrics.json")
    cm_path = Path("results/images/confusion_matrix.png")
    roc_path = Path("results/images/roc_curve.png")

    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)

        ax_metrics = fig.add_axes([0.05, 0.55, 0.4, 0.35])
        ax_metrics.axis("off")
        rows = [
            ["Model", "XGBoost + LightGBM Ensemble"],
            ["CV AUC-ROC (mean)", f"{m.get('cv_auc_mean', 0):.4f} ± {m.get('cv_auc_std', 0):.4f}"],
            ["Test AUC-ROC", f"{m.get('test_auc_roc', 0):.4f}"],
            ["Test AUC-PR", f"{m.get('test_auc_pr', 0):.4f}"],
            ["Test F1 Score", f"{m.get('test_f1', 0):.4f}"],
            ["High-Risk Threshold", f"{m.get('threshold', 0.65)}"],
            ["Test Set Positive Rate", f"{m.get('test_positive_rate', 0):.2%}"],
            ["CV Strategy", "TimeSeriesSplit (5 folds)"],
        ]
        tbl = ax_metrics.table(cellText=rows, cellLoc="left", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.5, 1.8)
        ax_metrics.set_title("Model Metrics", pad=8)

    if cm_path.exists():
        cm_img = plt.imread(cm_path)
        ax_cm = fig.add_axes([0.5, 0.5, 0.45, 0.4])
        ax_cm.imshow(cm_img)
        ax_cm.axis("off")
        ax_cm.set_title("Confusion Matrix")

    if roc_path.exists():
        roc_img = plt.imread(roc_path)
        ax_roc = fig.add_axes([0.05, 0.05, 0.85, 0.4])
        ax_roc.imshow(roc_img)
        ax_roc.axis("off")
        ax_roc.set_title("ROC Curves")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_plant_performance(pdf, kpi: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Plant Performance Deep-Dive", fontsize=14, fontweight="bold")

    if len(kpi) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    plant_avg = kpi.groupby("plant_name")[["mttr_hours", "defect_rate", "sla_breach_rate"]].mean().reset_index()

    axes[0].barh(plant_avg["plant_name"], plant_avg["mttr_hours"], color="#3498db")
    axes[0].set_title("Avg MTTR (hours)")
    axes[0].set_xlabel("Hours")

    axes[1].barh(plant_avg["plant_name"], plant_avg["defect_rate"] * 100, color="#e74c3c")
    axes[1].set_title("Avg Defect Rate (%)")
    axes[1].set_xlabel("Defect Rate %")

    axes[2].barh(plant_avg["plant_name"], plant_avg["sla_breach_rate"] * 100, color="#e67e22")
    axes[2].set_title("Avg SLA Breach Rate (%)")
    axes[2].set_xlabel("SLA Breach %")

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_top_risk_orders(pdf, wo_top: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(14, 8.5))
    fig.suptitle("Top 20 Highest-Risk Work Orders", fontsize=14, fontweight="bold")

    if len(wo_top) == 0:
        plt.text(0.5, 0.5, "No high-risk work orders found", ha="center", va="center", transform=fig.transFigure)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    ax = fig.add_subplot(111)
    ax.axis("off")

    display_cols = ["work_order_id", "plant_name", "machine_code", "priority", "status", "risk_score", "product_type"]
    available = [c for c in display_cols if c in wo_top.columns]
    table_data = wo_top[available].head(20)
    table_data = table_data.copy()
    if "risk_score" in table_data.columns:
        table_data["risk_score"] = table_data["risk_score"].round(4)

    tbl = ax.table(
        cellText=table_data.values,
        colLabels=[c.replace("_", " ").title() for c in available],
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.4)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
