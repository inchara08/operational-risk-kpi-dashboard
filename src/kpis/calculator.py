"""
KPI Calculator: computes daily operational KPIs per plant and writes
results to the kpi_snapshots table in PostgreSQL.

KPIs:
  OEE (Overall Equipment Effectiveness) = Availability × Performance × Quality
  MTTR (Mean Time to Repair)
  MTBF (Mean Time Between Failures)
  SLA Breach Rate
  Defect Rate
  High-Risk Work Order Count
  Anomaly Count
  Throughput (units/day)
"""

from __future__ import annotations

import logging

import pandas as pd
import psycopg2
import psycopg2.extras
import yaml

log = logging.getLogger(__name__)


def compute_oee(
    wo_day: pd.DataFrame,
    minutes_per_machine: float = 1440.0,  # 24h × 60min per machine
) -> dict:
    """Compute OEE components for a single day/plant slice."""
    # Total scheduled capacity = machines active that day × minutes per machine
    n_machines = wo_day["machine_id"].nunique()
    scheduled_minutes = max(n_machines, 1) * minutes_per_machine
    total_downtime = wo_day["downtime_minutes"].sum()
    availability = max(0.0, (scheduled_minutes - total_downtime) / scheduled_minutes)

    planned_units = wo_day["planned_units"].sum()
    actual_units = wo_day["actual_units"].fillna(0).sum()
    performance = actual_units / planned_units if planned_units > 0 else 0.0
    performance = min(1.0, max(0.0, performance))

    total_defects = wo_day["defect_count"].sum()
    quality = (actual_units - total_defects) / actual_units if actual_units > 0 else 0.0
    quality = min(1.0, max(0.0, quality))

    oee = availability * performance * quality

    return {
        "oee_score": round(oee, 4),
        "availability_rate": round(availability, 4),
        "performance_rate": round(performance, 4),
        "quality_rate": round(quality, 4),
    }


def compute_mttr_mtbf(wo_day: pd.DataFrame) -> dict:
    failures = wo_day[wo_day["status"] == "failed"]
    n_failures = len(failures)
    total_downtime_h = wo_day["downtime_minutes"].sum() / 60

    mttr = total_downtime_h / n_failures if n_failures > 0 else 0.0

    completed = wo_day[wo_day["status"] == "completed"]
    if len(completed) > 0 and completed["planned_duration_hours"].notna().any():
        total_uptime_h = completed.get("planned_duration_hours", pd.Series([8.0])).sum()
    else:
        total_uptime_h = 24.0

    mtbf = total_uptime_h / n_failures if n_failures > 0 else total_uptime_h

    return {
        "mttr_hours": round(mttr, 2),
        "mtbf_hours": round(min(mtbf, 9999.0), 2),
    }


def run(
    config_path: str = "config/config.yaml",
    use_db: bool = True,
) -> pd.DataFrame:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if use_db:
        from src.db import get_connection
        conn = get_connection(cfg)
        wo = pd.read_sql(
            "SELECT wo.*, m.machine_type FROM work_orders wo "
            "JOIN machines m USING (machine_id)",
            conn,
        )
        tel = pd.read_sql(
            "SELECT plant_id, DATE(recorded_at) AS day, "
            "SUM(anomaly_flag::int) AS anomaly_count "
            "FROM machine_telemetry t "
            "JOIN machines m USING (machine_id) "
            "GROUP BY plant_id, day",
            conn,
        )
    else:
        wo = pd.read_csv("data/raw/work_orders.csv")
        tel = pd.DataFrame(columns=["plant_id", "day", "anomaly_count"])

    wo["scheduled_start"] = pd.to_datetime(wo["scheduled_start"], utc=True, errors="coerce")
    wo["actual_units"] = pd.to_numeric(wo["actual_units"], errors="coerce").fillna(0)
    wo["defect_count"] = pd.to_numeric(wo["defect_count"], errors="coerce").fillna(0)
    wo["downtime_minutes"] = pd.to_numeric(wo["downtime_minutes"], errors="coerce").fillna(0)
    wo["day"] = wo["scheduled_start"].dt.date
    wo["risk_label"] = pd.to_numeric(wo.get("risk_label", 0), errors="coerce").fillna(0)

    if "planned_duration_hours" not in wo.columns:
        wo["scheduled_end"] = pd.to_datetime(wo["scheduled_end"], utc=True, errors="coerce")
        wo["planned_duration_hours"] = (
            (wo["scheduled_end"] - wo["scheduled_start"]).dt.total_seconds() / 3600
        ).clip(lower=0)

    snapshots = []
    for (plant_id, day), group in wo.groupby(["plant_id", "day"]):
        oee_metrics = compute_oee(group)
        time_metrics = compute_mttr_mtbf(group)

        total = len(group)
        sla_breach_count = (
            group["status"].isin(["failed", "delayed"]) &
            (group["downtime_minutes"] > cfg["kpis"]["sla_breach_threshold_hours"] * 60)
        ).sum()
        sla_breach_rate = sla_breach_count / total if total > 0 else 0.0

        actual_units = group["actual_units"].sum()
        defect_rate = group["defect_count"].sum() / actual_units if actual_units > 0 else 0.0

        high_risk_count = int((group["risk_label"] == 1).sum())
        throughput = int(actual_units)

        # Anomaly count from pre-aggregated telemetry
        if isinstance(tel, pd.DataFrame) and len(tel) > 0 and "day" in tel.columns:
            tel_day = tel[(tel["plant_id"] == plant_id) & (tel["day"].astype(str) == str(day))]
            anomaly_count = int(tel_day["anomaly_count"].sum()) if len(tel_day) > 0 else 0
        else:
            anomaly_count = 0

        snapshots.append({
            "snapshot_date": day,
            "plant_id": int(plant_id),
            **oee_metrics,
            **time_metrics,
            "sla_breach_rate": round(sla_breach_rate, 4),
            "defect_rate": round(min(defect_rate, 1.0), 4),
            "high_risk_wo_count": high_risk_count,
            "anomaly_count": anomaly_count,
            "throughput_units": throughput,
        })

    kpi_df = pd.DataFrame(snapshots)
    log.info("Computed %d KPI snapshots across %d plants.", len(kpi_df), kpi_df["plant_id"].nunique())

    if use_db:
        _upsert_snapshots(kpi_df, conn)
        conn.close()

    return kpi_df


def _upsert_snapshots(kpi_df: pd.DataFrame, conn) -> None:
    cols = [
        "snapshot_date", "plant_id", "oee_score", "availability_rate",
        "performance_rate", "quality_rate", "mttr_hours", "mtbf_hours",
        "sla_breach_rate", "defect_rate", "high_risk_wo_count",
        "anomaly_count", "throughput_units",
    ]
    records = [tuple(row[c] for c in cols) for _, row in kpi_df.iterrows()]

    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE kpi_snapshots RESTART IDENTITY")
        psycopg2.extras.execute_values(
            cur,
            f"INSERT INTO kpi_snapshots ({', '.join(cols)}) VALUES %s",
            records,
        )
    conn.commit()
    log.info("Inserted %d KPI snapshot rows.", len(records))
