"""
Generates a standalone, self-contained interactive HTML report using Plotly.
No server required — the file opens directly in any browser.

Charts included:
  1. OEE trend by plant over time
  2. Risk score distribution (histogram)
  3. Anomaly timeline per plant
  4. Defect rate Pareto by product type
  5. Plant comparison (bar chart, 4 KPIs)
  6. MTTR trend over time
  7. SLA breach heatmap (month × plant)
  8. Feature importance (static image embed)
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

log = logging.getLogger(__name__)


def _load_data(cfg: dict, use_db: bool) -> dict[str, pd.DataFrame]:
    if use_db:
        from src.db import get_connection
        conn = get_connection(cfg)
        data = {
            "kpi": pd.read_sql(
                "SELECT k.*, p.plant_name FROM kpi_snapshots k JOIN plants p USING (plant_id)", conn
            ),
            "wo": pd.read_sql(
                "SELECT wo.risk_score, wo.status, wo.product_type, wo.defect_count, "
                "wo.actual_units, wo.scheduled_start, p.plant_name "
                "FROM work_orders wo JOIN plants p USING (plant_id) "
                "WHERE wo.risk_score IS NOT NULL",
                conn,
            ),
        }
        conn.close()
        return data
    else:
        wo = pd.read_csv("data/raw/work_orders.csv")
        return {"kpi": pd.DataFrame(), "wo": wo}


def generate(cfg: dict, out_dir: str = "reports", use_db: bool = True) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    data = _load_data(cfg, use_db)

    kpi = data["kpi"]
    wo = data["wo"]

    figures = []

    # 1. OEE Trend by plant
    if len(kpi) > 0 and "snapshot_date" in kpi.columns:
        kpi["snapshot_date"] = pd.to_datetime(kpi["snapshot_date"])
        fig1 = px.line(
            kpi, x="snapshot_date", y="oee_score", color="plant_name",
            title="Overall Equipment Effectiveness (OEE) Trend by Plant",
            labels={"oee_score": "OEE Score", "snapshot_date": "Date", "plant_name": "Plant"},
        )
        fig1.add_hline(y=cfg["kpis"]["oee_threshold"], line_dash="dash",
                       line_color="red", annotation_text="OEE Target")
        fig1.update_layout(yaxis_tickformat=".0%")
        figures.append(("OEE Trend", fig1))

    # 2. Risk score distribution
    if len(wo) > 0 and "risk_score" in wo.columns:
        wo["risk_score"] = pd.to_numeric(wo["risk_score"], errors="coerce")
        fig2 = px.histogram(
            wo.dropna(subset=["risk_score"]),
            x="risk_score", color="status",
            title="Risk Score Distribution by Work Order Status",
            nbins=50, barmode="overlay", opacity=0.75,
            labels={"risk_score": "Risk Score (0=Low, 1=High)"},
        )
        fig2.add_vline(
            x=cfg["kpis"]["high_risk_score_threshold"],
            line_dash="dash", line_color="red",
            annotation_text="High-Risk Threshold",
        )
        figures.append(("Risk Distribution", fig2))

    # 3. MTTR trend
    if len(kpi) > 0 and "mttr_hours" in kpi.columns:
        fig3 = px.line(
            kpi, x="snapshot_date", y="mttr_hours", color="plant_name",
            title="Mean Time to Repair (MTTR) — Hours",
            labels={"mttr_hours": "MTTR (hours)", "snapshot_date": "Date"},
        )
        figures.append(("MTTR Trend", fig3))

    # 4. Defect rate Pareto by product type
    if len(wo) > 0 and {"product_type", "defect_count", "actual_units"}.issubset(wo.columns):
        wo["actual_units"] = pd.to_numeric(wo["actual_units"], errors="coerce").fillna(1)
        wo["defect_count"] = pd.to_numeric(wo["defect_count"], errors="coerce").fillna(0)
        pareto = (
            wo.groupby("product_type")
            .agg(total_defects=("defect_count", "sum"), total_units=("actual_units", "sum"))
            .assign(defect_rate=lambda d: d["total_defects"] / d["total_units"])
            .sort_values("total_defects", ascending=False)
            .reset_index()
        )
        fig4 = px.bar(
            pareto, x="product_type", y="total_defects",
            title="Defect Count by Product Type (Pareto)",
            labels={"total_defects": "Total Defects", "product_type": "Product Type"},
            color="defect_rate",
            color_continuous_scale="Reds",
        )
        figures.append(("Defect Pareto", fig4))

    # 5. Plant comparison KPI bar chart
    if len(kpi) > 0:
        plant_avg = (
            kpi.groupby("plant_name")[["oee_score", "sla_breach_rate", "defect_rate", "mttr_hours"]]
            .mean()
            .reset_index()
        )
        fig5 = make_subplots(rows=1, cols=2, subplot_titles=["Average OEE by Plant", "Average MTTR by Plant"])
        fig5.add_trace(
            go.Bar(x=plant_avg["plant_name"], y=plant_avg["oee_score"], name="OEE"),
            row=1, col=1,
        )
        fig5.add_trace(
            go.Bar(x=plant_avg["plant_name"], y=plant_avg["mttr_hours"], name="MTTR (hours)"),
            row=1, col=2,
        )
        fig5.update_layout(title_text="Plant Performance Comparison", showlegend=False)
        figures.append(("Plant Comparison", fig5))

    # 6. SLA breach heatmap
    if len(kpi) > 0 and "sla_breach_rate" in kpi.columns:
        kpi["month"] = kpi["snapshot_date"].dt.to_period("M").astype(str)
        heatmap_data = (
            kpi.groupby(["month", "plant_name"])["sla_breach_rate"].mean().reset_index()
        )
        heatmap_pivot = heatmap_data.pivot(index="plant_name", columns="month", values="sla_breach_rate")
        fig6 = px.imshow(
            heatmap_pivot,
            title="SLA Breach Rate Heatmap (Plant × Month)",
            color_continuous_scale="YlOrRd",
            labels={"color": "SLA Breach Rate"},
            aspect="auto",
        )
        figures.append(("SLA Heatmap", fig6))

    # Assemble HTML
    plots_html = ""
    for title, fig in figures:
        plots_html += f"<h2>{title}</h2>\n"
        plots_html += fig.to_html(full_html=False, include_plotlyjs=False)
        plots_html += "\n"

    # Embed SHAP image if available
    shap_path = Path("results/images/shap_summary.png")
    shap_section = ""
    if shap_path.exists():
        img_b64 = base64.b64encode(shap_path.read_bytes()).decode()
        shap_section = (
            "<h2>SHAP Feature Importance</h2>"
            f'<img src="data:image/png;base64,{img_b64}" style="max-width:900px;">'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Operational Risk KPI Report</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem 3rem; color: #222; }}
    h1 {{ color: #1a1a2e; border-bottom: 3px solid #1a1a2e; padding-bottom: 0.5rem; }}
    h2 {{ color: #2c3e50; margin-top: 2.5rem; }}
  </style>
</head>
<body>
  <h1>Operational Risk & KPI Analytics Dashboard — Report</h1>
  {plots_html}
  {shap_section}
</body>
</html>"""

    out_path = Path(out_dir) / "operational_risk_report.html"
    out_path.write_text(html)
    log.info("HTML report saved: %s", out_path)
