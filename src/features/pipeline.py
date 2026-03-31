"""
Feature engineering orchestrator.

Reads from PostgreSQL (or Parquet for local dev), assembles all 26 features,
and writes the final feature matrix to data/processed/features.parquet.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import yaml

from src.features.work_order_features import (
    build_static_features,
    build_schedule_pressure_features,
    build_machine_history_features,
    build_plant_context_features,
)
from src.features.telemetry_features import (
    aggregate_telemetry_per_machine,
    join_telemetry_to_work_orders,
)

log = logging.getLogger(__name__)

FEATURE_COLS = [
    # Work order static
    "priority_encoded", "product_type_encoded", "planned_units",
    "day_of_week", "hour_of_day", "is_monday", "is_end_of_quarter",
    # Schedule pressure
    "planned_duration_hours", "lead_time_hours", "queue_depth", "overdue_flag",
    # Machine history
    "machine_age_years", "failure_count_30d", "avg_downtime_30d", "days_since_last_failure",
    # Plant context
    "plant_defect_rate_7d", "plant_failure_rate_7d", "plant_utilization_rate",
    # Telemetry aggregates (prefixed tel_)
    "tel_temp_mean", "tel_temp_std", "tel_temp_max",
    "tel_vib_mean", "tel_vib_max",
    "tel_pres_deviation", "tel_pwr_spike_count", "tel_anomaly_count",
]

TARGET_COL = "risk_label_raw"  # 1 if status in {failed, delayed} else 0


def _load_from_db(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from src.db import get_connection
    conn = get_connection(cfg)
    wo = pd.read_sql("SELECT * FROM work_orders", conn)
    tel = pd.read_sql("SELECT * FROM machine_telemetry", conn)
    machines = pd.read_sql("SELECT machine_id, install_date, machine_type FROM machines", conn)
    conn.close()
    return wo, tel, machines


def _load_from_csv(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wo = pd.read_csv(data_dir / "work_orders.csv")
    tel = pd.read_csv(data_dir / "machine_telemetry.csv")
    machines = pd.read_csv(data_dir / "machines.csv")[["machine_id", "install_date", "machine_type"]]
    return wo, tel, machines


def run(
    config_path: str = "config/config.yaml",
    data_dir: str = "data/raw",
    out_dir: str = "data/processed",
    use_db: bool = True,
) -> pd.DataFrame:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    log.info("Loading data...")
    if use_db:
        wo, tel, machines = _load_from_db(cfg)
    else:
        wo, tel, machines = _load_from_csv(Path(data_dir))

    # Merge machine metadata into work orders
    wo = wo.merge(machines, on="machine_id", how="left")

    log.info("Building static features...")
    wo = build_static_features(wo)

    log.info("Building schedule pressure features...")
    wo = build_schedule_pressure_features(wo)

    log.info("Building machine history features...")
    wo = build_machine_history_features(wo)

    log.info("Building plant context features...")
    wo = build_plant_context_features(wo)

    log.info("Aggregating telemetry (this may take a few minutes)...")
    lookback_h = cfg["features"]["telemetry_lookback_hours"]
    tel_agg = aggregate_telemetry_per_machine(tel)
    wo = join_telemetry_to_work_orders(wo, tel_agg, lookback_hours=lookback_h)

    # Build binary target: 1 if work order resulted in failure or delayed > 2h
    wo[TARGET_COL] = (wo["status"].isin(["failed", "delayed"])).astype(int)

    # Final feature matrix
    available_features = [c for c in FEATURE_COLS if c in wo.columns]
    missing = set(FEATURE_COLS) - set(available_features)
    if missing:
        log.warning("Missing features (will be filled with 0): %s", missing)
        for col in missing:
            wo[col] = 0.0

    out_cols = ["work_order_id"] + FEATURE_COLS + [TARGET_COL]
    features_df = wo[[c for c in out_cols if c in wo.columns]].copy()
    features_df[FEATURE_COLS] = features_df[FEATURE_COLS].fillna(0)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / "features.parquet"
    features_df.to_parquet(out_path, index=False)
    log.info("Feature matrix saved: %s (%d rows × %d cols)", out_path, len(features_df), len(FEATURE_COLS))

    return features_df
