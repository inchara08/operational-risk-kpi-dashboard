"""
Operational Risk & KPI Analytics Dashboard — Pipeline CLI

Usage:
    python main.py generate     # Generate synthetic raw data
    python main.py ingest       # Load raw CSVs → PostgreSQL
    python main.py validate     # Run data quality checks
    python main.py features     # Engineer features → Parquet
    python main.py train        # Train risk + anomaly models
    python main.py score        # Score work orders + telemetry in DB
    python main.py kpis         # Compute KPI snapshots
    python main.py report       # Generate HTML + PDF reports
    python main.py dashboard    # Launch Plotly Dash on :8050
    python main.py all          # Run full pipeline end-to-end
"""

import logging
import os
import sys
from pathlib import Path

import click
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

CONFIG_PATH = "config/config.yaml"


def _load_cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@click.group()
def cli():
    """Operational Risk & KPI Analytics Dashboard pipeline."""


@cli.command()
@click.option("--config", default=CONFIG_PATH, help="Path to config.yaml")
@click.option("--out-dir", default="data/raw", help="Output directory for raw CSVs")
def generate(config, out_dir):
    """Generate synthetic manufacturing datasets (~2.1M rows)."""
    from src.ingestion.generator import run
    log.info("=== STAGE: generate ===")
    counts = run(config_path=config, out_dir=out_dir)
    for table, n in counts.items():
        log.info("  %s: %s rows", table, f"{n:,}")
    log.info("Generation complete.")


@cli.command()
@click.option("--config", default=CONFIG_PATH)
@click.option("--data-dir", default="data/raw")
def ingest(config, data_dir):
    """Bulk-load raw CSVs into PostgreSQL via COPY."""
    from src.ingestion.loader import run
    log.info("=== STAGE: ingest ===")
    counts = run(config_path=config, data_dir=data_dir)
    for table, n in counts.items():
        log.info("  %s: %s rows loaded", table, f"{n:,}")
    log.info("Ingest complete.")


@cli.command()
@click.option("--config", default=CONFIG_PATH)
@click.option("--data-dir", default="data/raw", help="CSV dir (used if --no-db)")
@click.option("--out-dir", default="reports", help="Report output directory")
@click.option("--no-db", is_flag=True, default=False, help="Read from CSVs instead of DB")
def validate(config, data_dir, out_dir, no_db):
    """Run data quality and business rule validation."""
    import pandas as pd
    from src.validation import schema_validator, business_rules, reporter

    log.info("=== STAGE: validate ===")
    cfg = _load_cfg()

    if not no_db:
        from src.db import get_connection
        conn = get_connection(cfg)
        wo = pd.read_sql("SELECT * FROM work_orders LIMIT 200000", conn)
        tel = pd.read_sql("SELECT * FROM machine_telemetry LIMIT 200000", conn)
        insp = pd.read_sql("SELECT * FROM quality_inspections LIMIT 200000", conn)
        conn.close()
    else:
        wo = pd.read_csv(Path(data_dir) / "work_orders.csv")
        tel = pd.read_csv(Path(data_dir) / "machine_telemetry.csv")
        insp = pd.read_csv(Path(data_dir) / "quality_inspections.csv")

    results = schema_validator.run_all(wo, tel, insp, cfg)
    results += business_rules.run_all(wo, insp)
    reporter.generate(results, Path(out_dir) / "validation_report.html")
    log.info("Validation complete. Report: %s/validation_report.html", out_dir)


@cli.command()
@click.option("--config", default=CONFIG_PATH)
@click.option("--data-dir", default="data/raw")
@click.option("--out-dir", default="data/processed")
@click.option("--no-db", is_flag=True, default=False)
def features(config, data_dir, out_dir, no_db):
    """Engineer 26 features and save to Parquet."""
    from src.features.pipeline import run
    log.info("=== STAGE: features ===")
    df = run(config_path=config, data_dir=data_dir, out_dir=out_dir, use_db=not no_db)
    log.info("Features complete: %d rows, %d columns", len(df), len(df.columns))


@cli.command()
@click.option("--config", default=CONFIG_PATH)
@click.option("--processed-dir", default="data/processed")
@click.option("--data-dir", default="data/raw")
@click.option("--no-db", is_flag=True, default=False)
def train(config, processed_dir, data_dir, no_db):
    """Train XGBoost/LightGBM ensemble + Isolation Forest anomaly detector."""
    from src.models import risk_classifier, anomaly_detector, evaluator

    log.info("=== STAGE: train ===")

    # Train anomaly detector on telemetry
    log.info("Training anomaly detector...")
    anomaly_detector.run_training(config_path=config, data_dir=data_dir, use_db=not no_db)

    # Train risk classifier
    log.info("Training risk classifier...")
    metrics = risk_classifier.run_training(config_path=config, processed_dir=processed_dir)

    log.info("Generating evaluation plots (SHAP, confusion matrix, ROC)...")
    evaluator.run_all(config_path=config, processed_dir=processed_dir)

    log.info(
        "Training complete: AUC-ROC=%.4f | AUC-PR=%.4f | F1=%.4f",
        metrics["test_auc_roc"], metrics["test_auc_pr"], metrics["test_f1"],
    )


@cli.command()
@click.option("--config", default=CONFIG_PATH)
@click.option("--processed-dir", default="data/processed")
@click.option("--data-dir", default="data/raw")
@click.option("--no-db", is_flag=True, default=False)
def score(config, processed_dir, data_dir, no_db):
    """Score all work orders and write risk_score/risk_label back to PostgreSQL."""
    import pandas as pd
    from src.models import risk_classifier, anomaly_detector

    log.info("=== STAGE: score ===")
    cfg = _load_cfg()

    # Score risk classifier
    features_path = Path(processed_dir) / "features.parquet"
    features_df = pd.read_parquet(features_path)
    scores_df = risk_classifier.score(features_df, cfg)

    if not no_db:
        import psycopg2.extras
        from src.db import get_connection
        conn = get_connection(cfg)
        records = list(scores_df[["risk_score", "risk_label", "work_order_id"]].itertuples(index=False, name=None))
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                "UPDATE work_orders SET risk_score=%s, risk_label=%s WHERE work_order_id=%s",
                records,
                page_size=5000,
            )
        conn.commit()
        log.info("Risk scores written to PostgreSQL: %d rows updated.", len(records))

        # Score anomaly detector and write back
        log.info("Scoring anomaly detector on telemetry...")
        tel = pd.read_sql("SELECT * FROM machine_telemetry", conn)
        model, scaler = anomaly_detector.load_model()
        tel_scored = anomaly_detector.score(tel, model, scaler, cfg)
        anom_records = list(
            tel_scored[["anomaly_score", "anomaly_flag", "telemetry_id"]]
            .itertuples(index=False, name=None)
        )
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                "UPDATE machine_telemetry SET anomaly_score=%s, anomaly_flag=%s WHERE telemetry_id=%s",
                anom_records,
                page_size=10000,
            )
        conn.commit()
        log.info("Anomaly scores written: %d rows.", len(anom_records))
        conn.close()
    else:
        scores_df.to_csv(Path(data_dir) / "risk_scores.csv", index=False)
        log.info("Risk scores saved to CSV (--no-db mode).")


@cli.command()
@click.option("--config", default=CONFIG_PATH)
@click.option("--no-db", is_flag=True, default=False)
def kpis(config, no_db):
    """Compute and store daily KPI snapshots per plant."""
    from src.kpis.calculator import run
    log.info("=== STAGE: kpis ===")
    df = run(config_path=config, use_db=not no_db)
    log.info("KPI stage complete: %d snapshot rows.", len(df))


@cli.command()
@click.option("--config", default=CONFIG_PATH)
@click.option("--out-dir", default="reports")
@click.option("--no-db", is_flag=True, default=False)
def report(config, out_dir, no_db):
    """Generate standalone HTML and PDF reports."""
    from src.reporting.html_report import generate as gen_html
    from src.reporting.pdf_report import generate as gen_pdf
    log.info("=== STAGE: report ===")
    cfg = _load_cfg()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    gen_html(cfg, out_dir=out_dir, use_db=not no_db)
    gen_pdf(cfg, out_dir=out_dir, use_db=not no_db)
    log.info("Reports written to %s/", out_dir)


@cli.command()
@click.option("--config", default=CONFIG_PATH)
def dashboard(config):
    """Launch interactive Plotly Dash dashboard on http://localhost:8050."""
    from src.dashboard.app import create_app
    log.info("=== STAGE: dashboard ===")
    cfg = _load_cfg()
    app = create_app(cfg)
    app.run(
        host=cfg["dashboard"]["host"],
        port=cfg["dashboard"]["port"],
        debug=cfg["dashboard"]["debug"],
    )


@cli.command(name="all")
@click.option("--config", default=CONFIG_PATH)
@click.option("--no-db", is_flag=True, default=False, help="Use CSVs instead of PostgreSQL")
@click.pass_context
def run_all(ctx, config, no_db):
    """Run the complete pipeline end-to-end."""
    log.info("=== FULL PIPELINE START ===")
    ctx.invoke(generate, config=config, out_dir="data/raw")
    if not no_db:
        ctx.invoke(ingest, config=config, data_dir="data/raw")
    ctx.invoke(validate, config=config, data_dir="data/raw", out_dir="reports", no_db=no_db)
    ctx.invoke(features, config=config, data_dir="data/raw", out_dir="data/processed", no_db=no_db)
    ctx.invoke(train, config=config, processed_dir="data/processed", data_dir="data/raw", no_db=no_db)
    ctx.invoke(score, config=config, processed_dir="data/processed", data_dir="data/raw", no_db=no_db)
    ctx.invoke(kpis, config=config, no_db=no_db)
    ctx.invoke(report, config=config, out_dir="reports", no_db=no_db)
    log.info("=== FULL PIPELINE COMPLETE ===")
    log.info("Run `python main.py dashboard` to launch the interactive dashboard.")


if __name__ == "__main__":
    cli()
