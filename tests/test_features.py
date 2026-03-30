"""Tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.features.work_order_features import (
    build_static_features,
    build_schedule_pressure_features,
    PRIORITY_MAP,
)
from src.features.telemetry_features import aggregate_telemetry_per_machine


def _sample_wo(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame({
        "work_order_id": range(1, n + 1),
        "machine_id": [1] * n,
        "plant_id": [1] * n,
        "created_at": pd.date_range("2022-01-01", periods=n, freq="6h", tz="UTC"),
        "scheduled_start": pd.date_range("2022-01-01 01:00", periods=n, freq="6h", tz="UTC"),
        "scheduled_end": pd.date_range("2022-01-01 09:00", periods=n, freq="6h", tz="UTC"),
        "actual_start": pd.date_range("2022-01-01 01:05", periods=n, freq="6h", tz="UTC"),
        "actual_end": pd.date_range("2022-01-01 09:10", periods=n, freq="6h", tz="UTC"),
        "status": ["completed"] * n,
        "priority": ["medium"] * n,
        "product_type": ["Widget-A"] * (n // 2) + ["Widget-B"] * (n - n // 2),
        "planned_units": [100] * n,
        "actual_units": [98] * n,
        "defect_count": [2] * n,
        "downtime_minutes": [5] * n,
    })


def _sample_tel(n: int = 50) -> pd.DataFrame:
    return pd.DataFrame({
        "machine_id": [1] * n,
        "recorded_at": pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC"),
        "temperature_c": np.random.default_rng(42).normal(65, 5, n),
        "vibration_hz": np.random.default_rng(42).normal(12, 2, n),
        "pressure_bar": np.random.default_rng(42).normal(8, 1, n),
        "power_kw": np.random.default_rng(42).normal(45, 5, n),
        "rpm": np.random.default_rng(42).normal(3000, 100, n),
        "anomaly_flag": [False] * n,
        "anomaly_score": [0.0] * n,
    })


def test_static_features_priority_encoding():
    wo = _sample_wo()
    result = build_static_features(wo)
    assert "priority_encoded" in result.columns
    assert set(result["priority_encoded"].unique()).issubset(set(PRIORITY_MAP.values()))


def test_static_features_planned_duration_positive():
    wo = _sample_wo()
    result = build_static_features(wo)
    assert "planned_duration_hours" in result.columns
    assert (result["planned_duration_hours"] >= 0).all()


def test_static_features_time_columns():
    wo = _sample_wo()
    result = build_static_features(wo)
    for col in ["day_of_week", "hour_of_day", "is_monday", "is_end_of_quarter"]:
        assert col in result.columns


def test_schedule_pressure_overdue_flag():
    wo = _sample_wo()
    # Set lead_time < 2h by putting created_at very close to scheduled_start
    wo["created_at"] = wo["scheduled_start"] - pd.Timedelta(hours=1)
    result = build_schedule_pressure_features(wo)
    assert "overdue_flag" in result.columns
    assert (result["overdue_flag"] == 1).all()


def test_telemetry_aggregation_columns():
    tel = _sample_tel()
    agg = aggregate_telemetry_per_machine(tel)
    expected_cols = {"machine_id", "hour", "temp_mean", "vib_mean", "pres_mean", "rpm_cv"}
    assert expected_cols.issubset(set(agg.columns))


def test_telemetry_aggregation_no_nulls():
    tel = _sample_tel()
    agg = aggregate_telemetry_per_machine(tel)
    assert agg.isnull().sum().sum() == 0


def test_product_type_encoding():
    wo = _sample_wo()
    result = build_static_features(wo)
    # All rows should have a valid integer encoding
    assert result["product_type_encoded"].dtype in [int, "int64", "int32"]
