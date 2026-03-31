"""
Domain-specific business rule validation.

Rules checked:
  - actual_end cannot precede actual_start
  - scheduled_end must be after scheduled_start
  - created_at must precede scheduled_start
  - downtime_minutes must be non-negative
  - Inspections must reference existing work_order_ids
"""

from __future__ import annotations

import logging

import pandas as pd

from src.validation.schema_validator import ValidationError, ValidationResult

log = logging.getLogger(__name__)


def check_temporal_consistency(wo: pd.DataFrame) -> list[ValidationResult]:
    results = []

    ts_cols = ["created_at", "scheduled_start", "scheduled_end", "actual_start", "actual_end"]
    for col in ts_cols:
        if col in wo.columns:
            wo[col] = pd.to_datetime(wo[col], utc=True, errors="coerce")

    # scheduled_end > scheduled_start
    mask = wo["scheduled_start"].notna() & wo["scheduled_end"].notna()
    bad = (wo.loc[mask, "scheduled_end"] <= wo.loc[mask, "scheduled_start"]).sum()
    level = "CRITICAL" if bad > 0 else "INFO"
    results.append(ValidationResult(
        level=level, table="work_orders", check="scheduled_end_after_start",
        detail=f"{bad} rows where scheduled_end <= scheduled_start", value=bad,
    ))

    # created_at <= scheduled_start
    mask = wo["created_at"].notna() & wo["scheduled_start"].notna()
    bad = (wo.loc[mask, "created_at"] > wo.loc[mask, "scheduled_start"]).sum()
    level = "WARNING" if bad > 0 else "INFO"
    results.append(ValidationResult(
        level=level, table="work_orders", check="created_before_scheduled_start",
        detail=f"{bad} rows where created_at > scheduled_start", value=bad,
    ))

    # actual_end > actual_start (when both present)
    mask = wo["actual_start"].notna() & wo["actual_end"].notna()
    bad = (wo.loc[mask, "actual_end"] < wo.loc[mask, "actual_start"]).sum()
    level = "CRITICAL" if bad > 0 else "INFO"
    results.append(ValidationResult(
        level=level, table="work_orders", check="actual_end_after_start",
        detail=f"{bad} rows where actual_end < actual_start", value=bad,
    ))

    return results


def check_numeric_sanity(wo: pd.DataFrame) -> list[ValidationResult]:
    results = []

    # downtime_minutes >= 0
    if "downtime_minutes" in wo.columns:
        bad = (wo["downtime_minutes"] < 0).sum()
        level = "CRITICAL" if bad > 0 else "INFO"
        results.append(ValidationResult(
            level=level, table="work_orders", check="downtime_non_negative",
            detail=f"{bad} rows with negative downtime_minutes", value=bad,
        ))

    # actual_units >= 0
    if "actual_units" in wo.columns:
        bad = (wo["actual_units"].dropna() < 0).sum()
        level = "CRITICAL" if bad > 0 else "INFO"
        results.append(ValidationResult(
            level=level, table="work_orders", check="actual_units_non_negative",
            detail=f"{bad} rows with negative actual_units", value=bad,
        ))

    return results


def check_referential_integrity(
    wo: pd.DataFrame,
    inspections: pd.DataFrame,
) -> list[ValidationResult]:
    results = []

    if "work_order_id" in wo.columns and "work_order_id" in inspections.columns:
        valid_ids = set(wo["work_order_id"].dropna().astype(int))
        insp_ids = set(inspections["work_order_id"].dropna().astype(int))
        orphaned = insp_ids - valid_ids
        level = "CRITICAL" if len(orphaned) > 0 else "INFO"
        results.append(ValidationResult(
            level=level, table="quality_inspections", check="referential_integrity",
            detail=f"{len(orphaned)} inspections reference non-existent work_order_ids",
            value=len(orphaned),
        ))

    return results


def run_all(
    work_orders: pd.DataFrame,
    inspections: pd.DataFrame,
) -> list[ValidationResult]:
    results = (
        check_temporal_consistency(work_orders)
        + check_numeric_sanity(work_orders)
        + check_referential_integrity(work_orders, inspections)
    )

    criticals = [r for r in results if r.level == "CRITICAL"]
    if criticals:
        for r in criticals:
            log.error("CRITICAL [%s.%s]: %s", r.table, r.check, r.detail)
        raise ValidationError(
            f"{len(criticals)} critical business rule failure(s). "
            "Pipeline halted. See validation report for details."
        )

    return results
