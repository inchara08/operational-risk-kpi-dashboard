"""
Schema and range validation for ingested operational data.

Checks:
  - Null rates per column (CRITICAL if exceeds threshold)
  - Numeric range violations (WARNING or CRITICAL)
  - Categorical domain violations (WARNING)

Raises ValidationError if any CRITICAL check fails.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when a CRITICAL validation rule fails."""


@dataclass
class ValidationResult:
    level: str          # CRITICAL | WARNING | INFO
    table: str
    check: str
    detail: str
    value: Any = None
    threshold: Any = None

    @property
    def passed(self) -> bool:
        return self.level == "INFO"


def validate_work_orders(df: pd.DataFrame, cfg: dict) -> list[ValidationResult]:
    results = []
    val_cfg = cfg["validation"]
    max_null = val_cfg["max_null_rate"]

    required_cols = [
        "work_order_id", "machine_id", "plant_id", "created_at",
        "scheduled_start", "scheduled_end", "status", "priority",
        "product_type", "planned_units",
    ]
    for col in required_cols:
        null_rate = df[col].isna().mean() if col in df.columns else 1.0
        level = "CRITICAL" if null_rate > max_null else "INFO"
        results.append(ValidationResult(
            level=level, table="work_orders", check=f"null_rate:{col}",
            detail=f"Null rate {null_rate:.2%}",
            value=null_rate, threshold=max_null,
        ))

    # planned_units must be positive
    if "planned_units" in df.columns:
        bad = (df["planned_units"] <= 0).sum()
        level = "CRITICAL" if bad > 0 else "INFO"
        results.append(ValidationResult(
            level=level, table="work_orders", check="planned_units_positive",
            detail=f"{bad} rows with planned_units <= 0", value=bad,
        ))

    # defect_count <= actual_units
    if {"defect_count", "actual_units"}.issubset(df.columns):
        mask = df["actual_units"].notna() & df["defect_count"].notna()
        bad = (df.loc[mask, "defect_count"] > df.loc[mask, "actual_units"]).sum()
        level = "WARNING" if bad > 0 else "INFO"
        results.append(ValidationResult(
            level=level, table="work_orders", check="defect_count_lte_actual_units",
            detail=f"{bad} rows where defect_count > actual_units", value=bad,
        ))

    # status domain
    valid_statuses = {"completed", "failed", "delayed", "in_progress", "cancelled"}
    if "status" in df.columns:
        bad_vals = df.loc[~df["status"].isin(valid_statuses), "status"].unique()
        level = "WARNING" if len(bad_vals) > 0 else "INFO"
        results.append(ValidationResult(
            level=level, table="work_orders", check="status_domain",
            detail=f"Invalid statuses: {list(bad_vals)}", value=len(bad_vals),
        ))

    return results


def validate_telemetry(df: pd.DataFrame, cfg: dict) -> list[ValidationResult]:
    results = []
    val_cfg = cfg["validation"]
    max_null = val_cfg["max_null_rate"]

    range_checks = {
        "temperature_c":  val_cfg["temperature_range"],
        "vibration_hz":   val_cfg["vibration_range"],
        "pressure_bar":   val_cfg["pressure_range"],
        "power_kw":       val_cfg["power_range"],
        "rpm":            val_cfg["rpm_range"],
    }

    for col, (lo, hi) in range_checks.items():
        if col not in df.columns:
            continue
        null_rate = df[col].isna().mean()
        if null_rate > max_null:
            results.append(ValidationResult(
                level="WARNING", table="machine_telemetry", check=f"null_rate:{col}",
                detail=f"Null rate {null_rate:.2%}", value=null_rate, threshold=max_null,
            ))
        out_of_range = df[col].dropna().pipe(lambda s: ((s < lo) | (s > hi)).sum())
        pct = out_of_range / len(df) if len(df) > 0 else 0
        level = "CRITICAL" if pct > 0.01 else ("WARNING" if pct > 0.001 else "INFO")
        results.append(ValidationResult(
            level=level, table="machine_telemetry", check=f"range:{col}",
            detail=f"{out_of_range} rows outside [{lo}, {hi}]",
            value=out_of_range, threshold=f"[{lo}, {hi}]",
        ))

    return results


def validate_inspections(df: pd.DataFrame, cfg: dict) -> list[ValidationResult]:
    results = []

    # units_passed + units_failed == units_inspected
    if {"units_passed", "units_failed", "units_inspected"}.issubset(df.columns):
        mask = df[["units_passed", "units_failed", "units_inspected"]].notna().all(axis=1)
        bad = (
            df.loc[mask, "units_passed"] + df.loc[mask, "units_failed"]
            != df.loc[mask, "units_inspected"]
        ).sum()
        level = "WARNING" if bad > 0 else "INFO"
        results.append(ValidationResult(
            level=level, table="quality_inspections", check="unit_count_consistency",
            detail=f"{bad} rows where passed + failed != inspected", value=bad,
        ))

    # severity domain
    valid_severities = {"minor", "major", "critical"}
    if "severity" in df.columns:
        bad_count = (~df["severity"].isin(valid_severities)).sum()
        level = "WARNING" if bad_count > 0 else "INFO"
        results.append(ValidationResult(
            level=level, table="quality_inspections", check="severity_domain",
            detail=f"{bad_count} invalid severity values", value=bad_count,
        ))

    return results


def run_all(
    work_orders: pd.DataFrame,
    telemetry: pd.DataFrame,
    inspections: pd.DataFrame,
    cfg: dict,
) -> list[ValidationResult]:
    """Run all validators. Raises ValidationError if any CRITICAL check fails."""
    results = (
        validate_work_orders(work_orders, cfg)
        + validate_telemetry(telemetry, cfg)
        + validate_inspections(inspections, cfg)
    )

    criticals = [r for r in results if r.level == "CRITICAL"]
    warnings = [r for r in results if r.level == "WARNING"]

    log.info(
        "Validation complete: %d checks | %d CRITICAL | %d WARNING",
        len(results), len(criticals), len(warnings),
    )
    for r in criticals:
        log.error("CRITICAL [%s.%s]: %s", r.table, r.check, r.detail)
    for r in warnings:
        log.warning("WARNING [%s.%s]: %s", r.table, r.check, r.detail)

    if criticals:
        raise ValidationError(
            f"{len(criticals)} critical validation failure(s). "
            "Fix data quality issues before proceeding. "
            "See validation_report.html for details."
        )

    return results
