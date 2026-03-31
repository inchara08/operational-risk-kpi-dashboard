"""Tests for the data validation layer."""

import pandas as pd

from src.validation.business_rules import (
    check_numeric_sanity,
    check_referential_integrity,
    check_temporal_consistency,
)
from src.validation.schema_validator import (
    validate_telemetry,
    validate_work_orders,
)

MINIMAL_CFG = {
    "validation": {
        "max_null_rate": 0.02,
        "temperature_range": [15.0, 120.0],
        "vibration_range": [0.0, 50.0],
        "pressure_range": [0.5, 20.0],
        "power_range": [0.0, 500.0],
        "rpm_range": [0.0, 5000.0],
    }
}


def _good_wo(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({
        "work_order_id": range(1, n + 1),
        "machine_id": [1] * n,
        "plant_id": [1] * n,
        "created_at": ["2022-01-01T00:00:00+00:00"] * n,
        "scheduled_start": ["2022-01-01T01:00:00+00:00"] * n,
        "scheduled_end": ["2022-01-01T09:00:00+00:00"] * n,
        "actual_start": ["2022-01-01T01:05:00+00:00"] * n,
        "actual_end": ["2022-01-01T09:10:00+00:00"] * n,
        "status": ["completed"] * n,
        "priority": ["medium"] * n,
        "product_type": ["Widget-A"] * n,
        "planned_units": [100] * n,
        "actual_units": [98] * n,
        "defect_count": [2] * n,
        "downtime_minutes": [5] * n,
    })


def _good_tel(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({
        "machine_id": [1] * n,
        "recorded_at": pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC"),
        "temperature_c": [65.0] * n,
        "vibration_hz": [12.0] * n,
        "pressure_bar": [8.0] * n,
        "power_kw": [45.0] * n,
        "rpm": [3000.0] * n,
    })


# ─── Schema validator ─────────────────────────────────────────

def test_valid_work_orders_pass():
    results = validate_work_orders(_good_wo(), MINIMAL_CFG)
    criticals = [r for r in results if r.level == "CRITICAL"]
    assert len(criticals) == 0


def test_null_rate_trigger():
    wo = _good_wo()
    wo["status"] = None
    results = validate_work_orders(wo, MINIMAL_CFG)
    criticals = [r for r in results if r.level == "CRITICAL" and "status" in r.check]
    assert len(criticals) > 0


def test_negative_planned_units_critical():
    wo = _good_wo()
    wo["planned_units"] = -1
    results = validate_work_orders(wo, MINIMAL_CFG)
    criticals = [r for r in results if r.level == "CRITICAL" and "planned_units" in r.check]
    assert len(criticals) > 0


def test_telemetry_out_of_range():
    tel = _good_tel()
    tel.loc[0, "temperature_c"] = 999.0  # way out of range
    results = validate_telemetry(tel, MINIMAL_CFG)
    # Should flag as CRITICAL (>1% out of range)
    range_results = [r for r in results if "range:temperature_c" in r.check and r.level in ("CRITICAL", "WARNING")]
    assert len(range_results) > 0


def test_valid_telemetry_passes():
    results = validate_telemetry(_good_tel(), MINIMAL_CFG)
    criticals = [r for r in results if r.level == "CRITICAL"]
    assert len(criticals) == 0


# ─── Business rules ───────────────────────────────────────────

def test_actual_end_before_start_critical():
    wo = _good_wo(5)
    wo["actual_start"] = "2022-01-01T09:00:00+00:00"
    wo["actual_end"] = "2022-01-01T01:00:00+00:00"  # end before start!
    results = check_temporal_consistency(wo)
    criticals = [r for r in results if r.level == "CRITICAL"]
    assert len(criticals) > 0


def test_negative_downtime_critical():
    wo = _good_wo(5)
    wo["downtime_minutes"] = -10
    results = check_numeric_sanity(wo)
    criticals = [r for r in results if r.level == "CRITICAL"]
    assert len(criticals) > 0


def test_referential_integrity_orphaned_inspection():
    wo = _good_wo(5)
    insp = pd.DataFrame({"work_order_id": [9999]})  # doesn't exist in wo
    results = check_referential_integrity(wo, insp)
    criticals = [r for r in results if r.level == "CRITICAL"]
    assert len(criticals) > 0


def test_referential_integrity_valid():
    wo = _good_wo(5)
    insp = pd.DataFrame({"work_order_id": [1, 2, 3]})
    results = check_referential_integrity(wo, insp)
    criticals = [r for r in results if r.level == "CRITICAL"]
    assert len(criticals) == 0
