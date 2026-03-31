# Operational Risk & KPI Analytics Dashboard

**End-to-end operational analytics platform** — predictive risk modeling on 2.1M+ manufacturing work order events, real-time KPI dashboards, and automated data validation pipelines.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue?logo=postgresql)
![Plotly Dash](https://img.shields.io/badge/Dashboard-Plotly%20Dash-informational)
![CI](https://github.com/inchara08/operational-risk-kpi-dashboard/actions/workflows/pipeline.yml/badge.svg)

---

## Results

| Model | AUC-ROC | AUC-PR | F1 (threshold=0.65) |
|---|---|---|---|
| XGBoost | 0.719 | 0.526 | 0.371 |
| LightGBM | 0.718 | 0.524 | 0.368 |
| **Ensemble** | **0.719** | **0.526** | **0.371** |

- Anomaly detector flags **100% of telemetry rows scored** across 1.3M readings
- Pipeline processes **2.3M rows end-to-end in ~15 minutes** on a standard laptop
- **TimeSeriesSplit cross-validation** (5 folds) — no temporal data leakage
- Risk scores written back to PostgreSQL; live KPI snapshots updated per run

<img width="1417" height="776" alt="image" src="https://github.com/user-attachments/assets/380cce00-44f2-44bf-906a-74fb7ff15235" />
<img width="1419" height="747" alt="image" src="https://github.com/user-attachments/assets/373680a0-086a-462a-b507-a615cf787067" />
<img width="1427" height="583" alt="image" src="https://github.com/user-attachments/assets/fa04d2f0-b553-4709-88a6-d2587d004248" />
<img width="1429" height="539" alt="image" src="https://github.com/user-attachments/assets/ef2af2dd-1123-42b6-8a6e-09a37a31ce5a" />
<img width="1436" height="812" alt="image" src="https://github.com/user-attachments/assets/bd57ec48-228b-46f9-b645-bf8b58efe7cc" />
<img width="1440" height="766" alt="image" src="https://github.com/user-attachments/assets/7b4cdbdc-0b9d-49d8-b1a6-12b71c864ce7" />

---

## Architecture

```
Raw Data (Synthetic)
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  INGESTION  generator.py → 500k work orders, 1.2M telemetry,   │
│             400k inspections │ loader.py → PostgreSQL COPY      │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  VALIDATION  schema_validator + business_rules → HTML report    │
│              Halts pipeline on CRITICAL failures                │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  FEATURES  26 features across 5 groups:                        │
│    - Work order static (priority, product type, time)          │
│    - Schedule pressure (lead time, queue depth, overdue flag)  │
│    - Machine history (rolling 30d failures, MTTR, last failure)│
│    - Telemetry aggregates (sensor stats, anomaly count 24h)    │
│    - Plant context (utilization rate, rolling defect/risk)     │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  MODELS                                                         │
│    IsolationForest → anomaly_flag per telemetry row            │
│    XGBoost + LightGBM Ensemble → risk_score per work order     │
│    SHAP explainability → feature importance visualization      │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  KPI CALCULATOR  OEE · MTTR · MTBF · SLA Breach · Defect Rate │
│                  Writes to kpi_snapshots → PostgreSQL           │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  DELIVERY                                                       │
│    Plotly Dash (5 pages) · HTML report (self-contained)        │
│    PDF report (stakeholder delivery) · Power BI compatible     │
└─────────────────────────────────────────────────────────────────┘
```

---

## KPIs Tracked

| KPI | Formula | Insight |
|---|---|---|
| **OEE** | Availability × Performance × Quality | Primary manufacturing efficiency metric |
| **MTTR** | Total downtime / # failures | Maintenance responsiveness |
| **MTBF** | Total uptime / # failures | Equipment reliability |
| **SLA Breach Rate** | Breached orders / Total orders | Service level adherence |
| **Defect Rate** | Defect units / Total units | Quality control |
| **High-Risk WO %** | Risk label=1 / Total orders | Model output KPI |
| **Throughput** | Units produced per day | Productivity |
| **Anomaly Rate** | Flagged telemetry / Total readings | Sensor health |

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/inchara08/operational-risk-kpi-dashboard.git
cd operational-risk-kpi-dashboard
pip install -r requirements.txt

# 2. Set up PostgreSQL
brew install postgresql@16 && brew services start postgresql@16
createdb operational_risk

# 3. Run the full pipeline (~4 minutes)
python main.py all

# 4. Launch the dashboard
python main.py dashboard
# → http://localhost:8050

# 5. Or run stages individually
python main.py generate    # Generate 2.1M rows of synthetic manufacturing data
python main.py ingest      # Bulk-load into PostgreSQL via COPY
python main.py validate    # Data quality checks → reports/validation_report.html
python main.py features    # Engineer 26 features → data/processed/features.parquet
python main.py train       # Train ensemble + anomaly detector, generate SHAP plots
python main.py score       # Write risk scores back to PostgreSQL
python main.py kpis        # Compute OEE, MTTR, SLA breach rate → kpi_snapshots
python main.py report      # Standalone HTML + PDF reports
```

---

## Dataset

Synthetic manufacturing domain (inspired by [UCI AI4I 2020 Predictive Maintenance](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) failure distributions):

| Table | Rows | Description |
|---|---|---|
| `work_orders` | 500,000 | Work orders across 5 plants over 3 years |
| `machine_telemetry` | 1,200,000 | Hourly sensor readings (50 machines) |
| `quality_inspections` | 400,000 | Post-production batch inspections |
| `kpi_snapshots` | ~5,500 | Daily KPI aggregates per plant (pipeline output) |

**Key design**: Plant 3 has a 3-month controlled degradation window (failure rate 3×, defect rate 2×) — this is what the models detect and what makes the dashboard visualizations compelling.

Sample data: `data/samples/` (500-row CSVs for exploration without running the full pipeline).

---

## PostgreSQL Schema

5 tables with proper foreign keys, indexes, and materialized views for dashboard performance:

```sql
plants → machines → work_orders
                  → machine_telemetry
         work_orders → quality_inspections
kpi_snapshots (pipeline output, refreshed each run)
```

Power BI users: connect directly to PostgreSQL via the PostgreSQL connector. All KPI queries point to materialized views in `sql/seed_views.sql`.

---

## Project Structure

```
├── main.py                    # Unified CLI (python main.py [generate|ingest|...])
├── config/config.yaml         # All parameters — no magic numbers in code
├── sql/
│   ├── schema.sql             # Full DDL with indexes
│   └── seed_views.sql         # Materialized views for dashboard performance
├── src/
│   ├── ingestion/             # Synthetic generator + PostgreSQL bulk loader
│   ├── validation/            # Schema + business rule checks + HTML reporter
│   ├── features/              # 26-feature engineering pipeline
│   ├── models/                # XGBoost/LightGBM ensemble + Isolation Forest + SHAP
│   ├── kpis/                  # OEE, MTTR, MTBF, SLA, defect rate calculator
│   ├── reporting/             # Standalone HTML report + 4-page PDF
│   └── dashboard/             # 5-page Plotly Dash app
├── tests/                     # 30+ pytest tests (generator, validation, features, models)
├── data/samples/              # 500-row sample CSVs (full data generated locally)
└── results/
    ├── images/                # SHAP beeswarm, confusion matrix, ROC curves
    └── model_metrics.json     # Reproducible training metrics
```

---

## Testing

```bash
pytest tests/ -v
# Runs 30+ tests covering:
# - Generator: row counts, valid statuses, non-negative downtime, degradation window detection
# - Validation: null rate thresholds, out-of-range sensors, temporal consistency, referential integrity
# - Features: priority encoding, schedule pressure, telemetry aggregation
# - Models: anomaly scoring, risk classifier AUC, score probabilities in [0, 1]
```

---

## Design Decisions Worth Noting

- **TimeSeriesSplit** for CV — prevents temporal leakage; training data never "sees" future work orders
- **PostgreSQL COPY** bulk loading — orders of magnitude faster than row-by-row INSERTs
- **Risk scores written back to source DB** — this is what "end-to-end shipped product" means; the model doesn't just output a CSV
- **Controlled degradation injection** in the generator — ensures the model output is meaningful and visually compelling
- **ValidationError halts the pipeline** on critical data quality failures — production-grade behavior, not research code
