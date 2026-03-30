"""
Plotly Dash dashboard — 5 pages.

Pages:
  1. Executive Summary  — KPI cards, OEE trend, plant comparison
  2. Risk Heatmap       — risk score scatter, top high-risk table
  3. Equipment Intel    — anomaly timeline, sensor trend lines
  4. Quality Drill-Down — defect Pareto, SLA breach calendar
  5. Model Insights     — SHAP plot, confusion matrix, ROC curve

Run:  python main.py dashboard
      → http://localhost:8050
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, dcc, html


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────

def _load(cfg: dict) -> dict[str, pd.DataFrame]:
    """Load data from PostgreSQL (or return empty frames if unavailable)."""
    try:
        import psycopg2
        db = cfg["database"]
        conn = psycopg2.connect(
            host=db["host"], port=db["port"], dbname=db["dbname"],
            user=db["user"], password=os.environ.get("PGPASSWORD", ""),
        )

        kpi = pd.read_sql(
            "SELECT k.*, p.plant_name FROM kpi_snapshots k JOIN plants p USING (plant_id)", conn
        )
        wo = pd.read_sql(
            "SELECT wo.work_order_id, wo.risk_score, wo.risk_label, wo.priority, "
            "wo.status, wo.product_type, wo.scheduled_start, wo.planned_units, "
            "wo.defect_count, wo.actual_units, wo.downtime_minutes, "
            "p.plant_name, m.machine_code, m.machine_type "
            "FROM work_orders wo "
            "JOIN plants p USING (plant_id) JOIN machines m USING (machine_id)",
            conn,
        )
        tel = pd.read_sql(
            "SELECT t.machine_id, t.recorded_at, t.temperature_c, t.vibration_hz, "
            "t.anomaly_flag, t.anomaly_score, p.plant_name, m.machine_code "
            "FROM machine_telemetry t "
            "JOIN machines m USING (machine_id) JOIN plants p USING (plant_id) "
            "LIMIT 200000",
            conn,
        )
        conn.close()
    except Exception as e:
        print(f"[dashboard] DB unavailable ({e}) — using empty frames")
        kpi, wo, tel = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    return {"kpi": kpi, "wo": wo, "tel": tel}


def _load_image_b64(path: str) -> str | None:
    p = Path(path)
    if p.exists():
        return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
    return None


def _load_metrics() -> dict:
    p = Path("results/model_metrics.json")
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


# ──────────────────────────────────────────────────────────────
# Layout helpers
# ──────────────────────────────────────────────────────────────

def _kpi_card(title: str, value: str, color: str = "#2c3e50") -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.H6(title, className="card-subtitle text-muted"),
            html.H3(value, style={"color": color, "fontWeight": "bold"}),
        ]),
        className="shadow-sm",
    )


# ──────────────────────────────────────────────────────────────
# Page builders
# ──────────────────────────────────────────────────────────────

def _page_executive(data: dict, cfg: dict) -> html.Div:
    kpi = data["kpi"]

    if len(kpi) == 0:
        return html.Div("No KPI data available. Run the pipeline first.", className="p-4 text-danger")

    kpi["snapshot_date"] = pd.to_datetime(kpi["snapshot_date"])
    latest = kpi.groupby("plant_name").last().reset_index()

    avg_oee = f"{kpi['oee_score'].mean():.1%}"
    avg_mttr = f"{kpi['mttr_hours'].mean():.1f}h"
    avg_sla = f"{kpi['sla_breach_rate'].mean():.2%}"
    avg_defect = f"{kpi['defect_rate'].mean():.2%}"
    total_high_risk = f"{int(kpi['high_risk_wo_count'].sum()):,}"

    kpi_row = dbc.Row([
        dbc.Col(_kpi_card("Avg OEE", avg_oee, "#27ae60"), md=2),
        dbc.Col(_kpi_card("Avg MTTR", avg_mttr, "#e67e22"), md=2),
        dbc.Col(_kpi_card("SLA Breach Rate", avg_sla, "#e74c3c"), md=2),
        dbc.Col(_kpi_card("Defect Rate", avg_defect, "#8e44ad"), md=2),
        dbc.Col(_kpi_card("High-Risk WOs", total_high_risk, "#c0392b"), md=2),
    ], className="mb-4 g-3")

    fig_oee = px.line(
        kpi, x="snapshot_date", y="oee_score", color="plant_name",
        title="OEE Trend by Plant",
        labels={"oee_score": "OEE", "snapshot_date": "Date", "plant_name": "Plant"},
    )
    fig_oee.add_hline(y=cfg["kpis"]["oee_threshold"], line_dash="dash", line_color="red",
                      annotation_text="Target")
    fig_oee.update_layout(yaxis_tickformat=".0%", height=350)

    plant_avg = kpi.groupby("plant_name")[["oee_score", "mttr_hours", "sla_breach_rate"]].mean().reset_index()
    fig_cmp = px.bar(
        plant_avg, x="plant_name", y="oee_score",
        title="Average OEE by Plant",
        color="oee_score", color_continuous_scale="RdYlGn",
        labels={"oee_score": "OEE", "plant_name": "Plant"},
    )
    fig_cmp.update_layout(yaxis_tickformat=".0%", height=350, showlegend=False)

    return html.Div([
        html.H4("Executive Summary", className="mb-3"),
        kpi_row,
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_oee), md=7),
            dbc.Col(dcc.Graph(figure=fig_cmp), md=5),
        ]),
    ])


def _page_risk(data: dict, cfg: dict) -> html.Div:
    wo = data["wo"]
    if len(wo) == 0:
        return html.Div("No work order data available.", className="p-4 text-danger")

    wo["risk_score"] = pd.to_numeric(wo.get("risk_score"), errors="coerce")
    wo["planned_units"] = pd.to_numeric(wo.get("planned_units"), errors="coerce")

    valid = wo.dropna(subset=["risk_score", "planned_units"])
    fig_scatter = px.scatter(
        valid.sample(min(5000, len(valid)), random_state=42),
        x="planned_units", y="risk_score",
        color="plant_name", symbol="priority",
        title="Risk Score vs. Planned Units (sampled 5k)",
        labels={"risk_score": "Risk Score", "planned_units": "Planned Units"},
        opacity=0.6,
        hover_data=["status", "product_type"],
    )
    fig_scatter.add_hline(y=cfg["kpis"]["high_risk_score_threshold"], line_dash="dash",
                          line_color="red", annotation_text="High-Risk Threshold")
    fig_scatter.update_layout(height=400)

    top_risk = (
        wo[wo.get("risk_label", pd.Series(0)) == 1]
        .nlargest(20, "risk_score")[
            ["work_order_id", "plant_name", "machine_code", "priority", "status", "risk_score", "product_type"]
        ]
        if "risk_label" in wo.columns else wo.nlargest(20, "risk_score")[
            ["work_order_id", "plant_name", "priority", "status", "risk_score"]
        ]
    )

    table = dbc.Table.from_dataframe(
        top_risk.round({"risk_score": 4}),
        striped=True, bordered=True, hover=True, size="sm",
    )

    return html.Div([
        html.H4("Risk Heatmap", className="mb-3"),
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_scatter))]),
        html.H5("Top 20 High-Risk Work Orders", className="mt-4 mb-2"),
        table,
    ])


def _page_equipment(data: dict) -> html.Div:
    tel = data["tel"]
    if len(tel) == 0:
        return html.Div("No telemetry data available.", className="p-4 text-danger")

    tel["recorded_at"] = pd.to_datetime(tel["recorded_at"], utc=True, errors="coerce")
    tel["anomaly_flag"] = pd.to_numeric(tel["anomaly_flag"], errors="coerce").fillna(0).astype(bool)

    # Downsample for performance
    sample = tel.sample(min(20000, len(tel)), random_state=42).sort_values("recorded_at")

    fig_temp = px.line(
        sample, x="recorded_at", y="temperature_c", color="plant_name",
        title="Machine Temperature Over Time (Sampled)",
        labels={"temperature_c": "Temp (°C)", "recorded_at": "Time"},
    )
    fig_temp.update_layout(height=350)

    anomalies = tel[tel["anomaly_flag"] == True].sample(min(5000, tel["anomaly_flag"].sum()), random_state=42)
    fig_anom = px.scatter(
        anomalies, x="recorded_at", y="vibration_hz",
        color="plant_name",
        title="Anomaly Events — Vibration at Flagged Readings",
        labels={"vibration_hz": "Vibration (Hz)", "recorded_at": "Time"},
        opacity=0.7,
    )
    fig_anom.update_layout(height=350)

    return html.Div([
        html.H4("Equipment Intelligence", className="mb-3"),
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_temp))]),
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_anom))]),
    ])


def _page_quality(data: dict) -> html.Div:
    wo = data["wo"]
    if len(wo) == 0:
        return html.Div("No data available.", className="p-4 text-danger")

    wo["defect_count"] = pd.to_numeric(wo.get("defect_count"), errors="coerce").fillna(0)
    wo["actual_units"] = pd.to_numeric(wo.get("actual_units"), errors="coerce").fillna(1)

    pareto = (
        wo.groupby("product_type")["defect_count"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig_pareto = px.bar(
        pareto, x="product_type", y="defect_count",
        title="Defect Count by Product Type (Pareto)",
        labels={"defect_count": "Total Defects", "product_type": "Product Type"},
        color="defect_count", color_continuous_scale="Reds",
    )

    kpi = data["kpi"]
    if len(kpi) > 0 and "sla_breach_rate" in kpi.columns:
        kpi["snapshot_date"] = pd.to_datetime(kpi["snapshot_date"])
        kpi["month"] = kpi["snapshot_date"].dt.to_period("M").astype(str)
        heatmap_data = kpi.groupby(["month", "plant_name"])["sla_breach_rate"].mean().reset_index()
        pivot = heatmap_data.pivot(index="plant_name", columns="month", values="sla_breach_rate")
        fig_heat = px.imshow(
            pivot, title="SLA Breach Rate Heatmap",
            color_continuous_scale="YlOrRd", aspect="auto",
        )
    else:
        fig_heat = go.Figure()
        fig_heat.update_layout(title="SLA Breach Heatmap (no data)")

    return html.Div([
        html.H4("Quality Drill-Down", className="mb-3"),
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_pareto))]),
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_heat))]),
    ])


def _page_model(metrics: dict) -> html.Div:
    shap_src = _load_image_b64("results/images/shap_summary.png")
    cm_src = _load_image_b64("results/images/confusion_matrix.png")
    roc_src = _load_image_b64("results/images/roc_curve.png")

    metric_rows = []
    if metrics:
        for k, v in {
            "CV AUC-ROC": f"{metrics.get('cv_auc_mean', 0):.4f} ± {metrics.get('cv_auc_std', 0):.4f}",
            "Test AUC-ROC": f"{metrics.get('test_auc_roc', 0):.4f}",
            "Test AUC-PR": f"{metrics.get('test_auc_pr', 0):.4f}",
            "Test F1": f"{metrics.get('test_f1', 0):.4f}",
            "Validation Strategy": "TimeSeriesSplit (5 folds, no leakage)",
        }.items():
            metric_rows.append(html.Tr([html.Td(k, className="fw-bold"), html.Td(v)]))

    images = []
    for label, src in [("SHAP Feature Importance", shap_src), ("Confusion Matrix", cm_src), ("ROC Curves", roc_src)]:
        if src:
            images.append(dbc.Col([html.H6(label), html.Img(src=src, style={"maxWidth": "100%"})], md=4))

    return html.Div([
        html.H4("Model Insights", className="mb-3"),
        dbc.Row([
            dbc.Col([
                html.H5("Performance Metrics"),
                dbc.Table(html.Tbody(metric_rows), bordered=True, striped=True, size="sm"),
            ], md=4),
        ], className="mb-4"),
        dbc.Row(images) if images else html.P("Run `python main.py train` to generate model visualizations."),
    ])


# ──────────────────────────────────────────────────────────────
# App factory
# ──────────────────────────────────────────────────────────────

def create_app(cfg: dict) -> dash.Dash:
    data = _load(cfg)
    metrics = _load_metrics()

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY],
        suppress_callback_exceptions=True,
        title="Operational Risk Dashboard",
    )

    NAV_ITEMS = [
        ("Executive Summary", "executive"),
        ("Risk Heatmap", "risk"),
        ("Equipment Intel", "equipment"),
        ("Quality Drill-Down", "quality"),
        ("Model Insights", "model"),
    ]

    app.layout = dbc.Container([
        dbc.NavbarSimple(
            brand="Operational Risk & KPI Analytics Dashboard",
            brand_href="#",
            color="dark",
            dark=True,
            className="mb-4",
            children=[
                dbc.NavItem(dbc.NavLink(label, href=f"#{page_id}", external_link=True))
                for label, page_id in NAV_ITEMS
            ],
        ),
        dcc.Tabs(
            id="tabs",
            value="tab-executive",
            children=[
                dcc.Tab(label=label, value=f"tab-{page_id}")
                for label, page_id in NAV_ITEMS
            ],
            colors={"border": "#dee2e6", "primary": "#2c3e50", "background": "#f8f9fa"},
        ),
        html.Div(id="tab-content", className="mt-4"),
    ], fluid=True)

    @app.callback(Output("tab-content", "children"), Input("tabs", "value"))
    def render_tab(tab: str):
        if tab == "tab-executive":
            return _page_executive(data, cfg)
        elif tab == "tab-risk":
            return _page_risk(data, cfg)
        elif tab == "tab-equipment":
            return _page_equipment(data)
        elif tab == "tab-quality":
            return _page_quality(data)
        elif tab == "tab-model":
            return _page_model(metrics)
        return html.Div("Unknown tab")

    return app
