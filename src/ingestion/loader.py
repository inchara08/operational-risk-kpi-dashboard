"""
PostgreSQL bulk loader using COPY FROM STDIN.

Uses psycopg2's copy_expert() for maximum throughput — far faster than
row-by-row INSERTs or SQLAlchemy's to_sql(). Idempotent: truncates target
tables before loading so re-runs are safe.
"""

from __future__ import annotations

import io
import os
import logging
from pathlib import Path

import psycopg2
import psycopg2.extras
import yaml

log = logging.getLogger(__name__)


def get_connection(cfg: dict):
    db = cfg["database"]
    return psycopg2.connect(
        host=db["host"],
        port=db["port"],
        dbname=db["dbname"],
        user=db["user"],
        password=os.environ.get("PGPASSWORD", ""),
    )


def _copy_csv(conn, table: str, csv_path: Path, columns: list[str]) -> int:
    """Bulk-load a CSV into a table via COPY. Returns rows loaded."""
    cols = ", ".join(columns)
    sql = (
        f"COPY {table} ({cols}) FROM STDIN WITH ("
        "FORMAT CSV, HEADER TRUE, NULL '', DELIMITER ',')"
    )
    with open(csv_path, "r") as f:
        with conn.cursor() as cur:
            cur.copy_expert(sql, f)
            return cur.rowcount


def run(config_path: str = "config/config.yaml", data_dir: str = "data/raw") -> dict:
    """Load all CSVs into PostgreSQL. Returns {table: row_count}."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_path = Path(data_dir)
    conn = get_connection(cfg)
    conn.autocommit = False
    counts = {}

    try:
        with conn.cursor() as cur:
            # Apply schema
            schema_sql = (Path(__file__).parent.parent.parent / "sql" / "schema.sql").read_text()
            cur.execute(schema_sql)

        # Load in dependency order
        load_order = [
            (
                "plants",
                data_path / "plants.csv",
                ["plant_code", "plant_name", "region", "capacity_units"],
            ),
            (
                "machines",
                data_path / "machines.csv",
                ["machine_code", "plant_id", "machine_type", "install_date", "expected_life_years"],
            ),
            (
                "work_orders",
                data_path / "work_orders.csv",
                [
                    "machine_id", "plant_id", "created_at", "scheduled_start",
                    "actual_start", "scheduled_end", "actual_end", "status",
                    "priority", "product_type", "planned_units", "actual_units",
                    "defect_count", "downtime_minutes", "operator_id", "failure_mode",
                ],
            ),
            (
                "machine_telemetry",
                data_path / "machine_telemetry.csv",
                [
                    "machine_id", "recorded_at", "temperature_c", "vibration_hz",
                    "pressure_bar", "power_kw", "rpm",
                ],
            ),
            (
                "quality_inspections",
                data_path / "quality_inspections.csv",
                [
                    "work_order_id", "inspected_at", "inspector_id",
                    "units_inspected", "units_passed", "units_failed",
                    "defect_type", "severity", "sla_breach",
                ],
            ),
        ]

        for table, csv_path, columns in load_order:
            if not csv_path.exists():
                log.warning("CSV not found, skipping: %s", csv_path)
                continue

            log.info("Loading %s...", table)
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE")

            n = _copy_csv(conn, table, csv_path, columns)
            counts[table] = n
            log.info("  %s: %s rows loaded", table, f"{n:,}")

        conn.commit()
        log.info("All tables loaded successfully.")

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return counts
