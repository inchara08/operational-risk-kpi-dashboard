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
from dash import Input, Output, dash_table, dcc, html


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────

def _load(cfg: dict) -> dict[str, pd.DataFrame]:
    """Load data from PostgreSQL (or return empty frames if unavailable)."""
    try:
        from src.db import get_connection
        conn = get_connection(cfg)

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

def _kpi_card(title: str, value: str, color: str = "#2c3e50", subtitle: str = "") -> dbc.Card:
    body = [
        html.H6(title, className="card-subtitle text-muted"),
        html.H3(value, style={"color": color, "fontWeight": "bold"}),
    ]
    if subtitle:
        body.append(html.Small(subtitle, style={"color": "#888", "fontSize": "0.75rem"}))
    return dbc.Card(dbc.CardBody(body), className="shadow-sm")


# ──────────────────────────────────────────────────────────────
# Page builders
# ──────────────────────────────────────────────────────────────

def _page_executive(data: dict, cfg: dict) -> html.Div:
    kpi = data["kpi"]

    if len(kpi) == 0:
        return html.Div("No KPI data available. Run the pipeline first.", className="p-4 text-danger")

    kpi["snapshot_date"] = pd.to_datetime(kpi["snapshot_date"])

    # Month-over-month context
    kpi["_month"] = kpi["snapshot_date"].dt.to_period("M")
    latest_m = kpi["_month"].max()
    prev_m = latest_m - 1
    cur = kpi[kpi["_month"] == latest_m]
    prv = kpi[kpi["_month"] == prev_m]

    def _mom(col, fmt="{:+.1%}", scale=1):
        c, p = cur[col].mean() * scale, prv[col].mean() * scale
        delta = c - p
        arrow = "↑" if delta > 0 else "↓"
        return f"{arrow} {abs(delta):{fmt[2:-1]}} MoM"

    threshold = cfg["kpis"]["oee_threshold"]
    avg_oee = f"{cur['oee_score'].mean():.1%}"
    oee_sub = f"{_mom('oee_score', '{:+.1%}')} · target {threshold:.0%}"

    avg_mttr = f"{cur['mttr_hours'].mean():.1f}h"
    mttr_delta = cur["mttr_hours"].mean() - prv["mttr_hours"].mean()
    mttr_sub = f"{'↑' if mttr_delta > 0 else '↓'} {abs(mttr_delta):.1f}h MoM"

    avg_sla = f"{cur['sla_breach_rate'].mean():.2%}"
    sla_delta = cur["sla_breach_rate"].mean() - prv["sla_breach_rate"].mean()
    sla_sub = f"{'↑' if sla_delta > 0 else '↓'} {abs(sla_delta):.2%} MoM"

    avg_defect = f"{cur['defect_rate'].mean():.2%}"
    def_delta = cur["defect_rate"].mean() - prv["defect_rate"].mean()
    def_sub = f"{'↑' if def_delta > 0 else '↓'} {abs(def_delta):.2%} MoM"

    hr_count = int(kpi["high_risk_wo_count"].sum())
    hr_total = int(kpi["throughput_units"].sum()) or 1
    hr_pct = hr_count / hr_total
    top_plant = kpi.groupby("plant_name")["high_risk_wo_count"].sum().idxmax()
    top_share = kpi[kpi["plant_name"] == top_plant]["high_risk_wo_count"].sum() / hr_count
    hr_sub = f"Top: {top_plant.split()[0]} ({top_share:.0%})"

    kpi_row = dbc.Row([
        dbc.Col(_kpi_card("Avg OEE", avg_oee, "#27ae60", oee_sub), md=2),
        dbc.Col(_kpi_card("Avg MTTR", avg_mttr, "#e67e22", mttr_sub), md=2),
        dbc.Col(_kpi_card("SLA Breach Rate", avg_sla, "#e74c3c", sla_sub), md=2),
        dbc.Col(_kpi_card("Defect Rate", avg_defect, "#8e44ad", def_sub), md=2),
        dbc.Col(_kpi_card("High-Risk WOs", f"{hr_count:,} ({hr_pct:.1%})", "#c0392b", hr_sub), md=2),
    ], className="mb-4 g-3")

    all_plants = sorted(kpi["plant_name"].unique().tolist())

    plant_avg = kpi.groupby("plant_name")[["oee_score", "mttr_hours", "sla_breach_rate"]].mean().reset_index()
    fig_cmp = px.bar(
        plant_avg, x="plant_name", y="oee_score",
        title="Average OEE by Plant",
        color="plant_name", color_discrete_map=PLANT_COLORS,
        labels={"oee_score": "OEE", "plant_name": "Plant"},
    )
    fig_cmp.update_layout(yaxis_tickformat=".0%", height=350, showlegend=False)

    return html.Div([
        html.H4("Executive Summary", className="mb-3"),
        kpi_row,
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col(
                        dcc.Dropdown(
                            id="plant-filter",
                            options=[{"label": p, "value": p} for p in all_plants],
                            value=all_plants,
                            multi=True,
                            placeholder="Filter plants...",
                            clearable=False,
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        dcc.RadioItems(
                            id="oee-smoothing",
                            options=[
                                {"label": " Raw", "value": 1},
                                {"label": " 7-day avg", "value": 7},
                                {"label": " 30-day avg", "value": 30},
                            ],
                            value=7,
                            inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "16px"},
                        ),
                        md=6, className="d-flex align-items-center",
                    ),
                ], className="mb-2"),
                dcc.Graph(id="oee-chart"),
            ], md=7),
            dbc.Col(dcc.Graph(figure=fig_cmp), md=5),
        ]),
    ])


def _page_risk(data: dict, cfg: dict) -> html.Div:
    wo = data["wo"]
    if len(wo) == 0:
        return html.Div("No work order data available.", className="p-4 text-danger")

    all_plants = sorted(wo["plant_name"].dropna().unique().tolist())
    all_priorities = ["critical", "high", "medium", "low"]

    return html.Div([
        html.H4("Risk Heatmap", className="mb-3"),
        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id="risk-plant-filter",
                    options=[{"label": p, "value": p} for p in all_plants],
                    value=all_plants,
                    multi=True,
                    placeholder="Filter plants...",
                    clearable=False,
                ),
                md=6,
            ),
            dbc.Col(
                dcc.Dropdown(
                    id="risk-priority-filter",
                    options=[{"label": p.capitalize(), "value": p} for p in all_priorities],
                    value=all_priorities,
                    multi=True,
                    placeholder="Filter priorities...",
                    clearable=False,
                ),
                md=6,
            ),
        ], className="mb-3"),
        html.Div(id="risk-insight-box", className="mb-3"),
        dcc.Graph(id="risk-scatter"),
        html.H5("Top 20 High-Risk Work Orders", className="mt-4 mb-2"),
        html.Div(id="risk-top-table"),
    ])


def _page_equipment(data: dict) -> html.Div:
    tel = data["tel"]
    if len(tel) == 0:
        return html.Div("No telemetry data available.", className="p-4 text-danger")

    all_plants = sorted(tel["plant_name"].dropna().unique().tolist())
    return html.Div([
        html.H4("Equipment Intelligence", className="mb-3"),
        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id="equip-plant-filter",
                    options=[{"label": p, "value": p} for p in all_plants],
                    value=all_plants,
                    multi=True,
                    placeholder="Filter plants...",
                    clearable=False,
                ),
                md=6,
            ),
        ], className="mb-3"),
        html.Div(id="equip-insight-box", className="mb-3"),
        dcc.Graph(id="equip-temp-chart"),
        dcc.Graph(id="equip-vibration-chart"),
        html.Div(id="equip-corr-table", className="mt-3 mb-4"),
    ])


def _page_quality(data: dict) -> html.Div:
    wo = data["wo"]
    if len(wo) == 0:
        return html.Div("No data available.", className="p-4 text-danger")

    all_plants = sorted(wo["plant_name"].dropna().unique().tolist())
    all_products = sorted(wo["product_type"].dropna().unique().tolist())

    return html.Div([
        html.H4("Quality Drill-Down", className="mb-3"),
        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id="quality-plant-filter",
                    options=[{"label": p, "value": p} for p in all_plants],
                    value=all_plants,
                    multi=True,
                    placeholder="Filter plants...",
                    clearable=False,
                ),
                md=6,
            ),
            dbc.Col(
                dcc.Dropdown(
                    id="quality-product-filter",
                    options=[{"label": p, "value": p} for p in all_products],
                    value=all_products,
                    multi=True,
                    placeholder="Filter product types...",
                    clearable=False,
                ),
                md=6,
            ),
        ], className="mb-3"),
        html.Div(id="quality-insight-box", className="mb-3"),
        dcc.Graph(id="quality-pareto-chart"),
        dcc.Graph(id="quality-sla-heatmap"),
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


# Consistent plant color palette — used across all charts
PLANT_COLORS = {
    "North Assembly Plant":  "#2196F3",   # blue
    "South Fabrication Hub": "#FF9800",   # orange
    "Midwest Press Works":   "#E53935",   # red
    "West Coast Packaging":  "#43A047",   # green
    "Central CNC Center":    "#9C27B0",   # purple
}


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

    @app.callback(
        Output("oee-chart", "figure"),
        Input("plant-filter", "value"),
        Input("oee-smoothing", "value"),
    )
    def update_oee_chart(selected_plants, window):
        kpi = data["kpi"].copy()
        if len(kpi) == 0:
            return go.Figure()
        kpi["snapshot_date"] = pd.to_datetime(kpi["snapshot_date"])
        if selected_plants:
            kpi = kpi[kpi["plant_name"].isin(selected_plants)]
        kpi = kpi.sort_values(["plant_name", "snapshot_date"])
        window = window or 7
        kpi["oee_smooth"] = (
            kpi.groupby("plant_name")["oee_score"]
            .transform(lambda s: s.rolling(window, min_periods=1).mean())
        )
        label = "OEE" if window == 1 else f"OEE ({window}-day avg)"
        fig = px.line(
            kpi, x="snapshot_date", y="oee_smooth", color="plant_name",
            title="OEE Trend by Plant",
            labels={"oee_smooth": label, "snapshot_date": "Date", "plant_name": "Plant"},
            color_discrete_map=PLANT_COLORS,
        )
        threshold = cfg["kpis"]["oee_threshold"]
        fig.add_hline(
            y=threshold, line_dash="dash", line_color="red",
            annotation_text=f"Target OEE: {threshold:.0%}",
            annotation_font_color="red",
            annotation_bgcolor="rgba(255,255,255,0.85)",
            annotation_bordercolor="red",
        )

        # Per-plant gap labels at the right edge
        x_max = kpi["snapshot_date"].max()
        for plant, grp in kpi.groupby("plant_name"):
            last_val = grp.sort_values("snapshot_date")["oee_smooth"].iloc[-1]
            gap = last_val - threshold
            color = "#27ae60" if gap >= 0 else "#e74c3c"
            arrow = "▲" if gap >= 0 else "▼"
            fig.add_annotation(
                x=x_max, y=last_val,
                text=f" {arrow} {gap:+.0%}",
                showarrow=False,
                xanchor="left",
                font={"size": 10, "color": color},
            )

        fig.update_layout(yaxis_tickformat=".0%", height=350)

        # Annotate the injected degradation window if Midwest Press Works is visible
        if not selected_plants or "Midwest Press Works" in selected_plants:
            fig.add_vrect(
                x0="2022-04-01", x1="2022-08-01",
                fillcolor="red", opacity=0.08, line_width=0,
                layer="below",
            )
            fig.add_annotation(
                x="2022-06-01", y=1.0,
                xref="x", yref="paper",
                text="<b>Equipment Degradation Event</b><br><sub>Midwest Press Works · Failure rate 3×, defect rate 2×</sub>",
                showarrow=True, arrowhead=2, arrowcolor="#c0392b",
                ax=0, ay=-40,
                bgcolor="rgba(255,235,235,0.9)",
                bordercolor="#c0392b", borderwidth=1,
                font={"size": 11, "color": "#c0392b"},
            )

        return fig

    @app.callback(
        Output("risk-scatter", "figure"),
        Output("risk-insight-box", "children"),
        Output("risk-top-table", "children"),
        Input("risk-plant-filter", "value"),
        Input("risk-priority-filter", "value"),
    )
    def update_risk_page(plants, priorities):
        wo = data["wo"].copy()
        if len(wo) == 0:
            empty = go.Figure()
            return empty, "", html.P("No data")

        wo["risk_score"] = pd.to_numeric(wo.get("risk_score"), errors="coerce")
        wo["planned_units"] = pd.to_numeric(wo.get("planned_units"), errors="coerce")
        filtered = wo.dropna(subset=["risk_score", "planned_units"])

        if plants:
            filtered = filtered[filtered["plant_name"].isin(plants)]
        if priorities:
            filtered = filtered[filtered["priority"].isin(priorities)]

        if len(filtered) == 0:
            return go.Figure(), dbc.Alert("No data matches the selected filters.", color="secondary"), html.P("No data")

        # Sample for performance
        sample = filtered.sample(min(6000, len(filtered)), random_state=42)
        threshold = cfg["kpis"]["high_risk_score_threshold"]

        normal = sample[sample["risk_score"] <= threshold]
        high = sample[sample["risk_score"] > threshold]

        fig = go.Figure()

        # Normal points — grey, de-emphasised
        fig.add_trace(go.Scatter(
            x=normal["planned_units"], y=normal["risk_score"],
            mode="markers",
            marker=dict(color="#adb5bd", size=5, opacity=0.25),
            name="Normal",
            hovertemplate="Plant: %{customdata[0]}<br>Priority: %{customdata[1]}<br>Status: %{customdata[2]}<br>Score: %{y:.3f}<extra></extra>",
            customdata=normal[["plant_name", "priority", "status"]].values,
        ))

        # High-risk points — colored by plant
        for plant, grp in high.groupby("plant_name"):
            color = PLANT_COLORS.get(plant, "#888")
            fig.add_trace(go.Scatter(
                x=grp["planned_units"], y=grp["risk_score"],
                mode="markers",
                marker=dict(color=color, size=6, opacity=0.7),
                name=f"{plant} (high-risk)",
                legendgroup=plant,
                hovertemplate=f"<b>{plant}</b><br>Priority: %{{customdata[0]}}<br>Status: %{{customdata[1]}}<br>Score: %{{y:.3f}}<extra></extra>",
                customdata=grp[["priority", "status"]].values,
            ))

        # Binned trend line
        sample["bucket"] = (sample["planned_units"] // 100 * 100).astype(int)
        trend = sample.groupby("bucket")["risk_score"].mean().reset_index().sort_values("bucket")
        fig.add_trace(go.Scatter(
            x=trend["bucket"], y=trend["risk_score"],
            mode="lines",
            line=dict(color="#495057", width=2, dash="dot"),
            name="Avg risk (binned)",
        ))

        fig.add_hline(
            y=threshold, line_dash="dash", line_color="red",
            annotation_text=f"High-Risk Threshold: {threshold}",
            annotation_font_color="red",
            annotation_bgcolor="rgba(255,255,255,0.85)",
            annotation_bordercolor="red",
        )
        fig.update_layout(
            title="Risk Score vs. Planned Units",
            xaxis_title="Planned Units",
            yaxis_title="Risk Score",
            height=420,
            legend=dict(groupclick="toggleitem"),
        )

        # ── Dynamic insight ──────────────────────────────────────
        pct_above = (filtered["risk_score"] > threshold).mean()
        corr = filtered["planned_units"].corr(filtered["risk_score"])
        top_plant = filtered.groupby("plant_name")["risk_score"].mean().idxmax() if plants else "N/A"

        if abs(corr) < 0.05:
            vol_msg = "Risk is largely independent of production volume."
        elif corr > 0:
            vol_msg = f"Risk tends to increase with production volume (r={corr:.2f})."
        else:
            vol_msg = f"Risk decreases with higher production volume (r={corr:.2f})."

        insight_text = (
            f"{pct_above:.1%} of orders exceed the risk threshold. "
            f"{vol_msg} "
            f"Highest average risk plant: {top_plant}."
        )
        insight = dbc.Alert([html.B("Insight: "), insight_text], color="warning", className="py-2")

        # ── Top-risk table ───────────────────────────────────────
        cols_show = ["work_order_id", "plant_name", "machine_code", "priority", "status", "risk_score", "product_type"]
        cols_show = [c for c in cols_show if c in filtered.columns]
        top_risk = (
            filtered.nlargest(20, "risk_score")[cols_show]
            .round({"risk_score": 4})
        )
        table = dash_table.DataTable(
            data=top_risk.to_dict("records"),
            columns=[{"name": c.replace("_", " ").title(), "id": c} for c in cols_show],
            sort_action="native",
            page_size=20,
            style_header={"backgroundColor": "#2c3e50", "color": "white", "fontWeight": "bold"},
            style_cell={"fontSize": "0.85rem", "padding": "6px"},
            style_data_conditional=[
                {
                    "if": {
                        "filter_query": f"{{risk_score}} > {threshold}",
                        "column_id": "risk_score",
                    },
                    "backgroundColor": "#ffeaea",
                    "color": "#c0392b",
                    "fontWeight": "bold",
                }
            ],
        )

        return fig, insight, table

    _TEMP_THRESHOLD = 95.0
    _VIB_THRESHOLD = 35.0

    @app.callback(
        Output("equip-temp-chart", "figure"),
        Output("equip-vibration-chart", "figure"),
        Output("equip-insight-box", "children"),
        Output("equip-corr-table", "children"),
        Input("equip-plant-filter", "value"),
    )
    def update_equipment_charts(selected_plants):
        tel = data["tel"].copy()
        if len(tel) == 0:
            empty = go.Figure()
            return empty, empty, "", html.P("No data")

        tel["recorded_at"] = pd.to_datetime(tel["recorded_at"], utc=True, errors="coerce")
        tel["anomaly_flag"] = pd.to_numeric(tel["anomaly_flag"], errors="coerce").fillna(0).astype(bool)
        tel["temperature_c"] = pd.to_numeric(tel["temperature_c"], errors="coerce")
        tel["vibration_hz"] = pd.to_numeric(tel["vibration_hz"], errors="coerce")

        if selected_plants:
            tel = tel[tel["plant_name"].isin(selected_plants)]
        if len(tel) == 0:
            empty = go.Figure()
            return empty, empty, dbc.Alert("No data for selected plants.", color="secondary"), html.P("No data")

        # ── Temperature chart ────────────────────────────────────
        # Daily means per plant for rolling avg
        tel["date"] = tel["recorded_at"].dt.normalize()
        daily = (
            tel.groupby(["plant_name", "date"])[["temperature_c", "vibration_hz"]]
            .mean()
            .reset_index()
            .sort_values(["plant_name", "date"])
        )
        daily["temp_7d"] = (
            daily.groupby("plant_name")["temperature_c"]
            .transform(lambda s: s.rolling(7, min_periods=1).mean())
        )

        # Raw sample (faded grey dots)
        raw_sample = tel.dropna(subset=["temperature_c"]).sample(
            min(15000, len(tel)), random_state=42
        ).sort_values("recorded_at")

        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=raw_sample["recorded_at"], y=raw_sample["temperature_c"],
            mode="markers",
            marker=dict(color="#adb5bd", size=2, opacity=0.15),
            name="Raw readings",
            showlegend=True,
        ))

        # 7-day rolling avg per plant
        for plant, grp in daily.groupby("plant_name"):
            color = PLANT_COLORS.get(plant, "#888")
            fig_temp.add_trace(go.Scatter(
                x=grp["date"], y=grp["temp_7d"],
                mode="lines",
                line=dict(color=color, width=2),
                name=f"{plant} (7d avg)",
            ))

        # Threshold line
        fig_temp.add_hline(
            y=_TEMP_THRESHOLD, line_dash="dash", line_color="orange",
            annotation_text=f"Overheating threshold ({_TEMP_THRESHOLD:.0f}°C)",
            annotation_font_color="orange",
            annotation_bgcolor="rgba(255,255,255,0.85)",
            annotation_bordercolor="orange",
        )

        # Shade windows where rolling avg exceeds threshold
        for plant, grp in daily.groupby("plant_name"):
            color = PLANT_COLORS.get(plant, "#888")
            overheat = grp[grp["temp_7d"] > _TEMP_THRESHOLD].sort_values("date")
            if len(overheat) == 0:
                continue
            # Find contiguous windows
            overheat = overheat.copy()
            overheat["gap"] = (overheat["date"].diff() > pd.Timedelta(days=2)).cumsum()
            for _, window in overheat.groupby("gap"):
                fig_temp.add_vrect(
                    x0=window["date"].min(), x1=window["date"].max(),
                    fillcolor=color, opacity=0.08, line_width=0, layer="below",
                )

        fig_temp.update_layout(
            title="Machine Temperature Over Time — 7-Day Rolling Average",
            xaxis_title="Date", yaxis_title="Temperature (°C)",
            height=380,
        )

        # ── Vibration chart ──────────────────────────────────────
        anomalies = tel[tel["anomaly_flag"] == True].dropna(subset=["vibration_hz"])
        if len(anomalies) == 0:
            fig_vib = go.Figure()
            fig_vib.update_layout(title="No anomaly readings for selected plants.", height=350)
        else:
            p99_vib = anomalies["vibration_hz"].quantile(0.99)
            normal_anom = anomalies[anomalies["vibration_hz"] <= p99_vib]
            extreme_anom = anomalies[anomalies["vibration_hz"] > p99_vib]

            # Sample normal anomalies for performance
            normal_sample = normal_anom.sample(min(8000, len(normal_anom)), random_state=42)

            fig_vib = go.Figure()
            for plant, grp in normal_sample.groupby("plant_name"):
                color = PLANT_COLORS.get(plant, "#888")
                fig_vib.add_trace(go.Scatter(
                    x=grp["recorded_at"], y=grp["vibration_hz"],
                    mode="markers",
                    marker=dict(color=color, size=4, opacity=0.5),
                    name=plant,
                    hovertemplate=f"<b>{plant}</b><br>Vibration: %{{y:.2f}} Hz<extra></extra>",
                ))

            # Top 1% — bright red, larger
            if len(extreme_anom) > 0:
                fig_vib.add_trace(go.Scatter(
                    x=extreme_anom["recorded_at"], y=extreme_anom["vibration_hz"],
                    mode="markers",
                    marker=dict(color="#e53935", size=9, opacity=0.9),
                    name="Top 1% vibration",
                    hovertemplate="<b>Extreme vibration</b><br>Plant: %{customdata}<br>%{y:.2f} Hz<extra></extra>",
                    customdata=extreme_anom["plant_name"].values,
                ))

            fig_vib.add_hline(
                y=_VIB_THRESHOLD, line_dash="dash", line_color="red",
                annotation_text=f"Abnormal vibration ({_VIB_THRESHOLD:.0f} Hz)",
                annotation_font_color="red",
                annotation_bgcolor="rgba(255,255,255,0.85)",
                annotation_bordercolor="red",
            )
            fig_vib.update_layout(
                title="Anomaly Events — Vibration at Flagged Readings",
                xaxis_title="Date", yaxis_title="Vibration (Hz)",
                height=380,
            )

        # ── Insight box ──────────────────────────────────────────
        pct_overheat = (tel["temperature_c"] > _TEMP_THRESHOLD).mean()
        if len(anomalies) > 0:
            pct_high_vib = (anomalies["vibration_hz"] > _VIB_THRESHOLD).mean()
            corr_map = (
                anomalies.groupby("plant_name")
                .apply(lambda g: g["temperature_c"].corr(g["vibration_hz"]))
                .dropna()
            )
            highest_corr_plant = corr_map.idxmax() if len(corr_map) > 0 else "N/A"
            highest_corr_val = corr_map.max() if len(corr_map) > 0 else 0.0
        else:
            pct_high_vib, highest_corr_plant, highest_corr_val = 0.0, "N/A", 0.0
            corr_map = pd.Series(dtype=float)

        if selected_plants and "Midwest Press Works" in selected_plants:
            narrative = (
                "Midwest Press Works shows a sustained temperature spike in mid-2022 (Apr–Aug), "
                "coinciding with elevated vibration levels — consistent with the injected equipment "
                "degradation window (failure rate 3×, defect rate 2×). Filter to this plant alone to isolate the signal."
            )
        else:
            narrative = (
                f"{pct_overheat:.1%} of readings exceed the overheating threshold ({_TEMP_THRESHOLD:.0f}°C). "
                f"{pct_high_vib:.1%} of flagged anomalies exceed the abnormal vibration level ({_VIB_THRESHOLD:.0f} Hz). "
                f"Strongest temp–vibration correlation: {highest_corr_plant} (r={highest_corr_val:.2f})."
            )
        insight = dbc.Alert([html.B("Insight: "), narrative], color="warning", className="py-2")

        # ── Correlation table ────────────────────────────────────
        if len(corr_map) > 0:
            corr_df = (
                corr_map.reset_index()
                .rename(columns={"plant_name": "Plant", 0: "Temp–Vibration Correlation"})
                .sort_values("Temp–Vibration Correlation", ascending=False)
            )
            corr_table = html.Div([
                html.H6("Temp–Vibration Correlation by Plant (anomaly readings)", className="mt-2 mb-1"),
                dash_table.DataTable(
                    data=corr_df.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in corr_df.columns],
                    sort_action="native",
                    page_size=5,
                    style_header={"backgroundColor": "#2c3e50", "color": "white", "fontWeight": "bold"},
                    style_cell={"fontSize": "0.85rem", "padding": "6px"},
                ),
            ])
        else:
            corr_table = html.P("No anomaly data for correlation.", className="text-muted")

        return fig_temp, fig_vib, insight, corr_table

    @app.callback(
        Output("quality-pareto-chart", "figure"),
        Output("quality-sla-heatmap", "figure"),
        Output("quality-insight-box", "children"),
        Input("quality-plant-filter", "value"),
        Input("quality-product-filter", "value"),
    )
    def update_quality_charts(plants, product_types):
        wo = data["wo"].copy()
        kpi = data["kpi"].copy()

        if len(wo) == 0:
            empty = go.Figure()
            return empty, empty, ""

        wo["defect_count"] = pd.to_numeric(wo.get("defect_count"), errors="coerce").fillna(0)

        # Apply plant filter to both
        if plants:
            wo = wo[wo["plant_name"].isin(plants)]
            kpi = kpi[kpi["plant_name"].isin(plants)] if len(kpi) > 0 else kpi

        # Apply product type filter to Pareto only
        wo_pareto = wo[wo["product_type"].isin(product_types)] if product_types else wo

        # ── Pareto chart ─────────────────────────────────────────
        pareto = (
            wo_pareto.groupby("product_type")["defect_count"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        if len(pareto) == 0:
            fig_pareto = go.Figure()
            fig_pareto.update_layout(title="No defect data for selected filters.")
            insight = dbc.Alert("No defect data for the selected filters.", color="secondary")
        else:
            total_defects = pareto["defect_count"].sum() or 1
            pareto["pct"] = pareto["defect_count"] / total_defects * 100
            pareto["cum_pct"] = pareto["pct"].cumsum()

            colors = ["#e53935" if i < 3 else "#90a4ae" for i in range(len(pareto))]

            fig_pareto = go.Figure()
            fig_pareto.add_trace(go.Bar(
                x=pareto["product_type"], y=pareto["defect_count"],
                marker_color=colors,
                text=[f"{p:.1f}%" for p in pareto["pct"]],
                textposition="outside",
                name="Defect Count",
            ))
            fig_pareto.add_trace(go.Scatter(
                x=pareto["product_type"], y=pareto["cum_pct"],
                mode="lines+markers",
                yaxis="y2",
                line=dict(color="#1565C0", width=2),
                marker=dict(size=6),
                name="Cumulative %",
            ))
            fig_pareto.add_hline(
                y=80, yref="y2", line_dash="dash", line_color="#757575",
                annotation_text="80% of defects",
                annotation_position="top right",
                annotation_font_color="#757575",
            )
            fig_pareto.update_layout(
                title="Defect Pareto — Product Type (top 3 in red)",
                xaxis_title="Product Type",
                yaxis_title="Total Defects",
                yaxis2=dict(
                    title="Cumulative %",
                    overlaying="y",
                    side="right",
                    range=[0, 105],
                    ticksuffix="%",
                ),
                height=420,
                legend=dict(orientation="h", y=-0.2),
            )

            # Dynamic insight
            n_types = len(pareto)
            top3_pct = pareto.head(min(3, n_types))["pct"].sum()
            top1_name = pareto.iloc[0]["product_type"]
            top1_pct = pareto.iloc[0]["pct"]
            insight_text = (
                f"Top {min(3, n_types)} product types account for {top3_pct:.0f}% of all defects. "
                f"{top1_name} is the largest contributor at {top1_pct:.1f}%."
            )
            insight = dbc.Alert([html.B("Insight: "), insight_text], color="warning", className="py-2")

        # ── SLA heatmap ──────────────────────────────────────────
        if len(kpi) > 0 and "sla_breach_rate" in kpi.columns:
            kpi["snapshot_date"] = pd.to_datetime(kpi["snapshot_date"])
            kpi["month"] = kpi["snapshot_date"].dt.to_period("M").astype(str)
            heatmap_data = (
                kpi.groupby(["month", "plant_name"])["sla_breach_rate"]
                .mean()
                .reset_index()
            )
            pivot = heatmap_data.pivot(index="plant_name", columns="month", values="sla_breach_rate")
            fig_heat = px.imshow(
                pivot,
                title="SLA Breach Rate by Plant & Month",
                color_continuous_scale="YlOrRd",
                aspect="auto",
                zmin=0, zmax=0.5,
                text_auto=".0%",
            )
            fig_heat.update_traces(textfont_size=9)
            fig_heat.update_coloraxes(colorbar_title="SLA Breach<br>Rate")
            fig_heat.update_layout(height=300)

            # Annotate Midwest degradation window only if that plant is visible
            if not plants or "Midwest Press Works" in (plants or []):
                pivot_cols = list(pivot.columns)
                mid_col = "2022-06" if "2022-06" in pivot_cols else (
                    next((c for c in pivot_cols if c.startswith("2022-0")), None)
                )
                if mid_col and "Midwest Press Works" in pivot.index:
                    fig_heat.add_annotation(
                        x=mid_col,
                        y="Midwest Press Works",
                        text="<b>Degradation peak</b>",
                        showarrow=True, arrowhead=2, arrowcolor="#c0392b",
                        ax=60, ay=-35,
                        bgcolor="rgba(255,235,235,0.9)",
                        bordercolor="#c0392b", borderwidth=1,
                        font=dict(size=10, color="#c0392b"),
                    )
        else:
            fig_heat = go.Figure()
            fig_heat.update_layout(title="SLA Breach Heatmap (no data)")

        return fig_pareto, fig_heat, insight

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
