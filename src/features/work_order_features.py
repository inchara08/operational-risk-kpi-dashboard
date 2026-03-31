"""
Feature engineering for work orders.

Produces the following feature groups:
  - Static work order attributes (priority, product type, time features)
  - Schedule pressure features (buffer, queue depth, overdue flag)
  - Machine history rolling aggregates (failures, downtime, last failure lag)
  - Plant context features (utilization rate, rolling defect/risk rates)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRIORITY_MAP = {"critical": 4, "high": 3, "medium": 2, "low": 1}


def build_static_features(wo: pd.DataFrame) -> pd.DataFrame:
    """Encode work order static attributes."""
    df = wo.copy()

    df["priority_encoded"] = df["priority"].map(PRIORITY_MAP).fillna(2)
    df["product_type_encoded"] = df["product_type"].astype("category").cat.codes

    # Time features
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df["scheduled_start"] = pd.to_datetime(df["scheduled_start"], utc=True, errors="coerce")
    df["scheduled_end"] = pd.to_datetime(df["scheduled_end"], utc=True, errors="coerce")

    df["day_of_week"] = df["scheduled_start"].dt.dayofweek
    df["hour_of_day"] = df["scheduled_start"].dt.hour
    df["is_monday"] = (df["day_of_week"] == 0).astype(int)
    df["is_end_of_quarter"] = (
        (df["scheduled_start"].dt.month.isin([3, 6, 9, 12]))
        & (df["scheduled_start"].dt.day >= 25)
    ).astype(int)

    df["planned_duration_hours"] = (
        (df["scheduled_end"] - df["scheduled_start"]).dt.total_seconds() / 3600
    ).clip(lower=0)

    return df


def build_schedule_pressure_features(wo: pd.DataFrame) -> pd.DataFrame:
    """Features capturing schedule pressure and queue state."""
    df = wo.copy()
    df["scheduled_start"] = pd.to_datetime(df["scheduled_start"], utc=True, errors="coerce")
    df["scheduled_end"] = pd.to_datetime(df["scheduled_end"], utc=True, errors="coerce")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")

    df["planned_duration_hours"] = (
        (df["scheduled_end"] - df["scheduled_start"]).dt.total_seconds() / 3600
    ).clip(lower=0)

    # Lead time: how far ahead was this scheduled after creation?
    df["lead_time_hours"] = (
        (df["scheduled_start"] - df["created_at"]).dt.total_seconds() / 3600
    ).clip(lower=0)

    # Queue depth: count of concurrent work orders per plant at scheduled_start
    # (approximated as work orders with overlapping windows in same plant)
    df = df.sort_values("scheduled_start")
    df["queue_depth"] = (
        df.groupby("plant_id")["scheduled_start"]
        .transform(lambda s: s.rank(method="first").astype(int) % 20)  # rolling proxy
    )

    # Overdue flag: lead_time < 2h = reactive/rushed
    df["overdue_flag"] = (df["lead_time_hours"] < 2).astype(int)

    return df


def build_machine_history_features(wo: pd.DataFrame) -> pd.DataFrame:
    """Rolling machine-level failure and downtime history."""
    df = wo.copy()
    df["scheduled_start"] = pd.to_datetime(df["scheduled_start"], utc=True, errors="coerce")
    df["is_failure"] = (df["status"] == "failed").astype(int)
    df = df.sort_values(["machine_id", "scheduled_start"])

    # Days since machine install (from machines metadata embedded in wo via merge upstream)
    if "install_date" in df.columns:
        df["install_date"] = pd.to_datetime(df["install_date"], errors="coerce")
        df["machine_age_days"] = (
            df["scheduled_start"].dt.tz_localize(None) - df["install_date"]
        ).dt.days.clip(lower=0)
        df["machine_age_years"] = (df["machine_age_days"] / 365.25).round(2)
    else:
        df["machine_age_years"] = 5.0  # default if not available

    # Rolling 30-day failure count and downtime
    df["failure_count_30d"] = (
        df.groupby("machine_id")["is_failure"]
        .transform(lambda s: s.rolling(window=30, min_periods=1).sum())
    )
    df["avg_downtime_30d"] = (
        df.groupby("machine_id")["downtime_minutes"]
        .transform(lambda s: s.rolling(window=30, min_periods=1).mean())
    ).fillna(0)

    # Days since last failure per machine
    def days_since_last_failure(group: pd.DataFrame) -> pd.Series:
        last_fail = pd.NaT
        result = []
        for _, row in group.iterrows():
            if pd.notna(last_fail):
                delta = (row["scheduled_start"] - last_fail).total_seconds() / 86400
                result.append(max(0, delta))
            else:
                result.append(365.0)  # no prior failure → 1 year as default
            if row["is_failure"] == 1:
                last_fail = row["scheduled_start"]
        return pd.Series(result, index=group.index)

    df["days_since_last_failure"] = (
        df.groupby("machine_id", group_keys=False).apply(days_since_last_failure)
    )

    return df


def build_plant_context_features(wo: pd.DataFrame) -> pd.DataFrame:
    """Rolling plant-level utilization, defect rate, and risk indicators."""
    df = wo.copy()
    df["scheduled_start"] = pd.to_datetime(df["scheduled_start"], utc=True, errors="coerce")
    df["defect_rate_wo"] = (
        df["defect_count"] / df["actual_units"].replace(0, np.nan)
    ).fillna(0).clip(0, 1)
    df["is_failure"] = (df["status"] == "failed").astype(int)
    df = df.sort_values(["plant_id", "scheduled_start"])

    df["plant_defect_rate_7d"] = (
        df.groupby("plant_id")["defect_rate_wo"]
        .transform(lambda s: s.rolling(7, min_periods=1).mean())
    ).fillna(0)

    df["plant_failure_rate_7d"] = (
        df.groupby("plant_id")["is_failure"]
        .transform(lambda s: s.rolling(7, min_periods=1).mean())
    ).fillna(0)

    # Utilization rate: rolling average throughput completions in last 14d
    df["completed_flag"] = (df["status"] == "completed").astype(int)
    df["plant_utilization_rate"] = (
        df.groupby("plant_id")["completed_flag"]
        .transform(lambda s: s.rolling(14, min_periods=1).mean())
    ).fillna(0.8)

    return df
