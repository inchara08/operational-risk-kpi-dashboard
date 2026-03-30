"""Tests for the synthetic data generator."""

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.ingestion.generator import (
    generate_machines,
    generate_plants,
    generate_work_orders,
    _is_in_degradation_window,
)
from datetime import datetime, timezone, timedelta


MINIMAL_CFG = {
    "generator": {
        "seed": 42,
        "n_plants": 5,
        "n_machines_per_plant": 10,
        "n_work_orders": 200,
        "n_telemetry_hours": 100,
        "n_inspections": 100,
        "date_start": "2021-01-01",
        "date_end": "2023-12-31",
        "degradation": {
            "plant_index": 2,
            "start_month": 15,
            "end_month": 18,
            "failure_rate_multiplier": 3.0,
            "defect_rate_multiplier": 2.0,
        },
    },
    "kpis": {"sla_breach_threshold_hours": 2.0},
    "validation": {},
}


def test_generate_plants_count():
    plants = generate_plants()
    assert len(plants) == 5


def test_generate_plants_unique_codes():
    plants = generate_plants()
    codes = [p["plant_code"] for p in plants]
    assert len(set(codes)) == len(codes)


def test_generate_machines_count():
    rng = np.random.default_rng(42)
    plants = generate_plants()
    machines = generate_machines(plants, rng)
    assert len(machines) == 50  # 5 plants × 10 machines


def test_generate_machines_plant_fk():
    rng = np.random.default_rng(42)
    plants = generate_plants()
    machines = generate_machines(plants, rng)
    valid_plant_ids = {p["plant_id"] for p in plants}
    for m in machines:
        assert m["plant_id"] in valid_plant_ids


def test_work_orders_no_negative_downtime():
    rng = np.random.default_rng(42)
    plants = generate_plants()
    machines = generate_machines(plants, rng)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "work_orders.csv"
        generate_work_orders(plants, machines, MINIMAL_CFG, rng, out_path)
        with open(out_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                assert int(row["downtime_minutes"]) >= 0, "downtime_minutes must be non-negative"


def test_work_orders_valid_status():
    rng = np.random.default_rng(42)
    plants = generate_plants()
    machines = generate_machines(plants, rng)
    valid_statuses = {"completed", "failed", "delayed", "cancelled"}
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "work_orders.csv"
        generate_work_orders(plants, machines, MINIMAL_CFG, rng, out_path)
        with open(out_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                assert row["status"] in valid_statuses


def test_work_orders_row_count():
    rng = np.random.default_rng(42)
    plants = generate_plants()
    machines = generate_machines(plants, rng)
    n = MINIMAL_CFG["generator"]["n_work_orders"]
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "work_orders.csv"
        count = generate_work_orders(plants, machines, MINIMAL_CFG, rng, out_path)
    assert count == n


def test_degradation_window_detection():
    date_start = datetime(2021, 1, 1, tzinfo=timezone.utc)
    degrad_cfg = {"start_month": 15, "end_month": 18}

    # Month 0 — not degraded
    ts_before = date_start + timedelta(days=30)
    assert not _is_in_degradation_window(ts_before, date_start, degrad_cfg)

    # Month 16 — in degradation window
    ts_during = date_start + timedelta(days=16 * 30)
    assert _is_in_degradation_window(ts_during, date_start, degrad_cfg)

    # Month 25 — after window
    ts_after = date_start + timedelta(days=25 * 30)
    assert not _is_in_degradation_window(ts_after, date_start, degrad_cfg)
