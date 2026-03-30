"""
Synthetic data generator for manufacturing operational data.

Produces 3 interrelated tables (~2.1M rows total):
  - work_orders       (500k rows, 3 years)
  - machine_telemetry (1.2M rows, hourly sensor readings)
  - quality_inspections (400k rows, post-production batches)

Key design: Plant 3 has a controlled degradation window (months 15-18)
where failure rate is 3x and defect rate is 2x baseline. This makes the
risk model's output visually dramatic and validates the anomaly detector.

Statistical basis: distributions seeded from UCI AI4I 2020 Predictive
Maintenance dataset failure rates and sensor value ranges.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator

import numpy as np
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
PLANT_NAMES = [
    ("PLT-001", "North Assembly Plant", "Northeast", 12000),
    ("PLT-002", "South Fabrication Hub", "Southeast", 9500),
    ("PLT-003", "Midwest Press Works", "Midwest", 11000),
    ("PLT-004", "West Coast Packaging", "West", 8000),
    ("PLT-005", "Central CNC Center", "Central", 10500),
]

MACHINE_TYPES = ["CNC", "Press", "Assembly", "Conveyor", "Packaging"]

PRODUCT_TYPES = [
    "Widget-A", "Widget-B", "Component-X", "Component-Y",
    "Assembly-Z", "Panel-1", "Panel-2", "Module-Alpha",
]

FAILURE_MODES = [
    "mechanical_wear", "electrical_fault", "overheating",
    "misalignment", "material_defect", "operator_error", None,
]

DEFECT_TYPES = [
    "dimensional", "surface_finish", "assembly_error",
    "material_flaw", "functional_failure",
]

PRIORITIES = ["critical", "high", "medium", "low"]
PRIORITY_WEIGHTS = [0.05, 0.20, 0.50, 0.25]

STATUSES = ["completed", "failed", "delayed", "cancelled"]


# ──────────────────────────────────────────────────────────────
# Plant & machine metadata
# ──────────────────────────────────────────────────────────────

def generate_plants() -> list[dict]:
    plants = []
    for i, (code, name, region, capacity) in enumerate(PLANT_NAMES, start=1):
        plants.append({
            "plant_id": i,
            "plant_code": code,
            "plant_name": name,
            "region": region,
            "capacity_units": capacity,
        })
    return plants


def generate_machines(plants: list[dict], rng: np.random.Generator) -> list[dict]:
    machines = []
    machine_id = 1
    for plant in plants:
        for j in range(10):
            machine_type = MACHINE_TYPES[j % len(MACHINE_TYPES)]
            install_year = rng.integers(2010, 2021)
            install_date = datetime(int(install_year), int(rng.integers(1, 13)), 1)
            machines.append({
                "machine_id": machine_id,
                "machine_code": f"MCH-{plant['plant_id']:02d}-{j + 1:02d}",
                "plant_id": plant["plant_id"],
                "machine_type": machine_type,
                "install_date": install_date.date().isoformat(),
                "expected_life_years": 15,
            })
            machine_id += 1
    return machines


# ──────────────────────────────────────────────────────────────
# Work order generation
# ──────────────────────────────────────────────────────────────

def _is_in_degradation_window(
    ts: datetime,
    date_start: datetime,
    degrad_cfg: dict,
) -> bool:
    months_elapsed = (ts.year - date_start.year) * 12 + (ts.month - date_start.month)
    return degrad_cfg["start_month"] <= months_elapsed <= degrad_cfg["end_month"]


def generate_work_orders(
    plants: list[dict],
    machines: list[dict],
    cfg: dict,
    rng: np.random.Generator,
    out_path: Path,
) -> int:
    """Stream work orders to CSV. Returns row count."""
    gen_cfg = cfg["generator"]
    degrad_cfg = gen_cfg["degradation"]
    degraded_plant_id = plants[degrad_cfg["plant_index"]]["plant_id"]

    date_start = datetime.fromisoformat(gen_cfg["date_start"]).replace(tzinfo=timezone.utc)
    date_end = datetime.fromisoformat(gen_cfg["date_end"]).replace(tzinfo=timezone.utc)
    total_seconds = int((date_end - date_start).total_seconds())

    n = gen_cfg["n_work_orders"]
    machine_by_plant: dict[int, list[dict]] = {}
    for m in machines:
        machine_by_plant.setdefault(m["plant_id"], []).append(m)

    fieldnames = [
        "work_order_id", "machine_id", "plant_id", "created_at",
        "scheduled_start", "actual_start", "scheduled_end", "actual_end",
        "status", "priority", "product_type", "planned_units", "actual_units",
        "defect_count", "downtime_minutes", "operator_id", "failure_mode",
        "risk_score", "risk_label",
    ]

    rows_written = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(n):
            # Random plant (weighted toward higher-capacity plants)
            plant = plants[rng.integers(0, len(plants))]
            plant_machines = machine_by_plant[plant["plant_id"]]
            machine = plant_machines[rng.integers(0, len(plant_machines))]

            # Timestamp — random within date range, with weekday/hour seasonality
            offset_s = rng.integers(0, total_seconds)
            created_at = date_start + timedelta(seconds=int(offset_s))

            # Inject seasonality: higher failure on Mondays (weekday=0) and end-of-quarter
            is_monday = created_at.weekday() == 0
            is_end_of_quarter = created_at.month in (3, 6, 9, 12) and created_at.day >= 25

            # Base failure probability
            is_degraded = (
                plant["plant_id"] == degraded_plant_id
                and _is_in_degradation_window(created_at, date_start, degrad_cfg)
            )
            base_fail_prob = 0.18
            if is_degraded:
                base_fail_prob *= degrad_cfg["failure_rate_multiplier"]
            if is_monday:
                base_fail_prob *= 1.3
            if is_end_of_quarter:
                base_fail_prob *= 1.2
            base_fail_prob = min(base_fail_prob, 0.85)

            priority = rng.choice(PRIORITIES, p=PRIORITY_WEIGHTS)
            product_type = rng.choice(PRODUCT_TYPES)
            planned_units = int(rng.integers(50, 1001))

            # Planned duration: 1–24 hours, priority-weighted
            priority_duration_scale = {"critical": 0.5, "high": 0.7, "medium": 1.0, "low": 1.5}
            planned_hours = float(rng.uniform(1, 24) * priority_duration_scale[priority])
            scheduled_start = created_at + timedelta(hours=float(rng.uniform(0.5, 4)))
            scheduled_end = scheduled_start + timedelta(hours=planned_hours)

            # Actual outcomes
            failed = rng.random() < base_fail_prob
            delayed = (not failed) and rng.random() < 0.15
            cancelled = (not failed) and (not delayed) and rng.random() < 0.02

            if cancelled:
                status = "cancelled"
                actual_start = None
                actual_end = None
                actual_units = 0
                downtime_minutes = 0
                defect_count = 0
                failure_mode = None
            elif failed:
                status = "failed"
                delay_h = float(rng.uniform(0, 2))
                actual_start = scheduled_start + timedelta(hours=delay_h)
                fail_point = float(rng.uniform(0.1, 0.9))
                actual_end = actual_start + timedelta(hours=planned_hours * fail_point)
                actual_units = int(planned_units * fail_point * float(rng.uniform(0.5, 0.9)))
                downtime_minutes = int(rng.integers(30, 480))
                base_defect = 0.25
                if is_degraded:
                    base_defect *= degrad_cfg["defect_rate_multiplier"]
                defect_count = int(actual_units * float(rng.uniform(base_defect * 0.5, base_defect * 1.5)))
                failure_mode = rng.choice([fm for fm in FAILURE_MODES if fm is not None])
            elif delayed:
                status = "delayed"
                delay_h = float(rng.uniform(0.5, 6))
                actual_start = scheduled_start + timedelta(hours=delay_h)
                actual_end = actual_start + timedelta(hours=planned_hours * float(rng.uniform(1.0, 1.5)))
                actual_units = int(planned_units * float(rng.uniform(0.85, 1.0)))
                downtime_minutes = int(rng.integers(0, 60))
                defect_count = int(actual_units * float(rng.uniform(0.01, 0.06)))
                failure_mode = None
            else:
                status = "completed"
                actual_start = scheduled_start + timedelta(minutes=float(rng.uniform(0, 15)))
                actual_end = actual_start + timedelta(hours=planned_hours * float(rng.uniform(0.9, 1.05)))
                actual_units = int(planned_units * float(rng.uniform(0.95, 1.0)))
                downtime_minutes = int(rng.integers(0, 20))
                defect_count = int(actual_units * float(rng.uniform(0.001, 0.02)))
                failure_mode = None

            writer.writerow({
                "work_order_id": i + 1,
                "machine_id": machine["machine_id"],
                "plant_id": plant["plant_id"],
                "created_at": created_at.isoformat(),
                "scheduled_start": scheduled_start.isoformat(),
                "actual_start": actual_start.isoformat() if actual_start else "",
                "scheduled_end": scheduled_end.isoformat(),
                "actual_end": actual_end.isoformat() if actual_end else "",
                "status": status,
                "priority": priority,
                "product_type": product_type,
                "planned_units": planned_units,
                "actual_units": actual_units if actual_units is not None else "",
                "defect_count": defect_count,
                "downtime_minutes": downtime_minutes,
                "operator_id": int(rng.integers(1, 201)),
                "failure_mode": failure_mode if failure_mode else "",
                "risk_score": "",   # populated by scoring stage
                "risk_label": "",   # populated by scoring stage
            })
            rows_written += 1

    return rows_written


# ──────────────────────────────────────────────────────────────
# Telemetry generation
# ──────────────────────────────────────────────────────────────

def generate_telemetry(
    plants: list[dict],
    machines: list[dict],
    cfg: dict,
    rng: np.random.Generator,
    out_path: Path,
) -> int:
    """Stream hourly telemetry to CSV. Returns row count."""
    gen_cfg = cfg["generator"]
    degrad_cfg = gen_cfg["degradation"]
    degraded_plant_id = plants[degrad_cfg["plant_index"]]["plant_id"]

    date_start = datetime.fromisoformat(gen_cfg["date_start"]).replace(tzinfo=timezone.utc)
    date_end = datetime.fromisoformat(gen_cfg["date_end"]).replace(tzinfo=timezone.utc)

    fieldnames = [
        "telemetry_id", "machine_id", "recorded_at",
        "temperature_c", "vibration_hz", "pressure_bar",
        "power_kw", "rpm",
        "anomaly_flag", "anomaly_score",
    ]

    # Baseline sensor profiles per machine type
    sensor_profiles = {
        "CNC":       {"temp": (65, 8),  "vib": (12, 3),  "pres": (8, 1),   "pwr": (45, 8),  "rpm": (3000, 200)},
        "Press":     {"temp": (55, 6),  "vib": (18, 4),  "pres": (15, 2),  "pwr": (80, 12), "rpm": (1200, 100)},
        "Assembly":  {"temp": (40, 5),  "vib": (5, 1.5), "pres": (3, 0.5), "pwr": (20, 4),  "rpm": (600, 80)},
        "Conveyor":  {"temp": (45, 6),  "vib": (8, 2),   "pres": (2, 0.3), "pwr": (15, 3),  "rpm": (900, 60)},
        "Packaging": {"temp": (38, 4),  "vib": (4, 1),   "pres": (2, 0.3), "pwr": (12, 2),  "rpm": (400, 50)},
    }

    machine_map = {m["machine_id"]: m for m in machines}
    rows_written = 0
    telemetry_id = 1

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for machine in machines:
            profile = sensor_profiles[machine["machine_type"]]
            plant_id = machine["plant_id"]
            is_degraded_machine = (plant_id == degraded_plant_id)

            # Iterate hour by hour
            current = date_start
            while current < date_end:
                in_degrad = (
                    is_degraded_machine
                    and _is_in_degradation_window(current, date_start, degrad_cfg)
                )

                # Sensor drift during degradation window
                drift = 1.4 if in_degrad else 1.0
                noise = 2.0 if in_degrad else 1.0

                temp = float(rng.normal(profile["temp"][0] * drift, profile["temp"][1] * noise))
                vib = float(rng.normal(profile["vib"][0] * drift, profile["vib"][1] * noise))
                pres = float(rng.normal(profile["pres"][0], profile["pres"][1]))
                pwr = float(rng.normal(profile["pwr"][0] * drift, profile["pwr"][1] * noise))
                rpm = float(rng.normal(profile["rpm"][0], profile["rpm"][1]))

                # Clamp to physical limits
                temp = max(15.0, min(120.0, round(temp, 2)))
                vib = max(0.0, min(50.0, round(abs(vib), 4)))
                pres = max(0.5, min(20.0, round(abs(pres), 3)))
                pwr = max(0.0, min(500.0, round(abs(pwr), 3)))
                rpm = max(0.0, min(5000.0, round(abs(rpm), 1)))

                writer.writerow({
                    "telemetry_id": telemetry_id,
                    "machine_id": machine["machine_id"],
                    "recorded_at": current.isoformat(),
                    "temperature_c": temp,
                    "vibration_hz": vib,
                    "pressure_bar": pres,
                    "power_kw": pwr,
                    "rpm": rpm,
                    "anomaly_flag": "",    # populated by anomaly scoring stage
                    "anomaly_score": "",
                })
                telemetry_id += 1
                rows_written += 1
                current += timedelta(hours=1)

    return rows_written


# ──────────────────────────────────────────────────────────────
# Quality inspection generation
# ──────────────────────────────────────────────────────────────

def generate_inspections(
    work_orders_path: Path,
    cfg: dict,
    rng: np.random.Generator,
    out_path: Path,
) -> int:
    """Generate quality inspections from completed/delayed work orders."""
    gen_cfg = cfg["generator"]
    sla_threshold_h = cfg["kpis"]["sla_breach_threshold_hours"]

    fieldnames = [
        "inspection_id", "work_order_id", "inspected_at",
        "inspector_id", "units_inspected", "units_passed", "units_failed",
        "defect_type", "severity", "sla_breach",
    ]

    eligible_statuses = {"completed", "delayed", "failed"}
    rows_written = 0
    inspection_id = 1

    with open(work_orders_path) as wo_f, open(out_path, "w", newline="") as out_f:
        reader = csv.DictReader(wo_f)
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for wo in reader:
            if wo["status"] not in eligible_statuses:
                continue
            if not wo["actual_end"]:
                continue

            actual_units = int(wo["actual_units"]) if wo["actual_units"] else 0
            if actual_units == 0:
                continue

            # 1 inspection per work order (occasionally 2 for critical priority)
            n_inspections = 2 if (wo["priority"] == "critical" and rng.random() < 0.4) else 1

            for _ in range(n_inspections):
                actual_end = datetime.fromisoformat(wo["actual_end"])
                insp_delay_h = float(rng.uniform(0.5, 4))
                inspected_at = actual_end + timedelta(hours=insp_delay_h)

                units_inspected = min(actual_units, int(rng.integers(max(1, actual_units // 4), actual_units + 1)))
                defect_rate = int(wo["defect_count"]) / actual_units if actual_units > 0 else 0.01
                units_failed = int(units_inspected * defect_rate * float(rng.uniform(0.8, 1.2)))
                units_failed = min(units_failed, units_inspected)
                units_passed = units_inspected - units_failed

                defect_type = rng.choice(DEFECT_TYPES) if units_failed > 0 else ""

                # Severity
                if units_failed / units_inspected > 0.15:
                    severity = "critical"
                elif units_failed / units_inspected > 0.05:
                    severity = "major"
                else:
                    severity = "minor"

                # SLA breach: actual_end vs scheduled_end
                scheduled_end = datetime.fromisoformat(wo["scheduled_end"])
                breach_hours = (actual_end - scheduled_end).total_seconds() / 3600
                sla_breach = breach_hours > sla_threshold_h

                writer.writerow({
                    "inspection_id": inspection_id,
                    "work_order_id": wo["work_order_id"],
                    "inspected_at": inspected_at.isoformat(),
                    "inspector_id": int(rng.integers(1, 51)),
                    "units_inspected": units_inspected,
                    "units_passed": units_passed,
                    "units_failed": units_failed,
                    "defect_type": defect_type,
                    "severity": severity if units_failed > 0 else "minor",
                    "sla_breach": sla_breach,
                })
                inspection_id += 1
                rows_written += 1

    return rows_written


# ──────────────────────────────────────────────────────────────
# Top-level runner
# ──────────────────────────────────────────────────────────────

def run(config_path: str = "config/config.yaml", out_dir: str = "data/raw") -> dict:
    """Generate all synthetic datasets. Returns dict of {table: row_count}."""
    import time

    cfg = load_config(config_path)
    rng = np.random.default_rng(cfg["generator"]["seed"])
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Generating plant and machine metadata...")
    plants = generate_plants()
    machines = generate_machines(plants, rng)

    # Write dimension CSVs
    _write_csv(out_path / "plants.csv", plants)
    _write_csv(out_path / "machines.csv", machines)
    print(f"  plants: {len(plants)} rows | machines: {len(machines)} rows")

    print(f"Generating {cfg['generator']['n_work_orders']:,} work orders...")
    t0 = time.time()
    wo_path = out_path / "work_orders.csv"
    wo_count = generate_work_orders(plants, machines, cfg, rng, wo_path)
    print(f"  work_orders: {wo_count:,} rows ({time.time() - t0:.1f}s)")

    print(f"Generating telemetry (50 machines × 3 years hourly)...")
    t0 = time.time()
    tel_path = out_path / "machine_telemetry.csv"
    tel_count = generate_telemetry(plants, machines, cfg, rng, tel_path)
    print(f"  machine_telemetry: {tel_count:,} rows ({time.time() - t0:.1f}s)")

    print("Generating quality inspections...")
    t0 = time.time()
    insp_path = out_path / "quality_inspections.csv"
    insp_count = generate_inspections(wo_path, cfg, rng, insp_path)
    print(f"  quality_inspections: {insp_count:,} rows ({time.time() - t0:.1f}s)")

    total = wo_count + tel_count + insp_count
    print(f"\nTotal rows generated: {total:,}")
    return {
        "plants": len(plants),
        "machines": len(machines),
        "work_orders": wo_count,
        "machine_telemetry": tel_count,
        "quality_inspections": insp_count,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
