-- =============================================================
-- Operational Risk & KPI Analytics — PostgreSQL Schema
-- =============================================================

-- ─── Dimension: Plants ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS plants (
    plant_id        SERIAL PRIMARY KEY,
    plant_code      VARCHAR(10)  NOT NULL UNIQUE,
    plant_name      VARCHAR(100) NOT NULL,
    region          VARCHAR(50),
    capacity_units  INTEGER      NOT NULL
);

-- ─── Dimension: Machines ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS machines (
    machine_id           SERIAL PRIMARY KEY,
    machine_code         VARCHAR(20) NOT NULL UNIQUE,
    plant_id             INTEGER     NOT NULL REFERENCES plants(plant_id),
    machine_type         VARCHAR(50) NOT NULL,   -- CNC, Press, Assembly, Conveyor, Packaging
    install_date         DATE        NOT NULL,
    expected_life_years  INTEGER     NOT NULL DEFAULT 15
);

-- ─── Fact: Work Orders ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS work_orders (
    work_order_id       BIGSERIAL PRIMARY KEY,
    machine_id          INTEGER      NOT NULL REFERENCES machines(machine_id),
    plant_id            INTEGER      NOT NULL REFERENCES plants(plant_id),
    created_at          TIMESTAMPTZ  NOT NULL,
    scheduled_start     TIMESTAMPTZ  NOT NULL,
    actual_start        TIMESTAMPTZ,
    scheduled_end       TIMESTAMPTZ  NOT NULL,
    actual_end          TIMESTAMPTZ,
    status              VARCHAR(20)  NOT NULL CHECK (status IN ('completed','failed','delayed','in_progress','cancelled')),
    priority            VARCHAR(10)  NOT NULL CHECK (priority IN ('critical','high','medium','low')),
    product_type        VARCHAR(50)  NOT NULL,
    planned_units       INTEGER      NOT NULL,
    actual_units        INTEGER,
    defect_count        INTEGER      NOT NULL DEFAULT 0,
    downtime_minutes    INTEGER      NOT NULL DEFAULT 0,
    operator_id         INTEGER,
    failure_mode        VARCHAR(50),             -- NULL if no failure
    -- Model outputs (populated by scoring stage)
    risk_score          NUMERIC(6,4),
    risk_label          SMALLINT                 -- 0=low, 1=high
);

-- ─── Fact: Machine Telemetry (hourly sensor readings) ────────
CREATE TABLE IF NOT EXISTS machine_telemetry (
    telemetry_id    BIGSERIAL   PRIMARY KEY,
    machine_id      INTEGER     NOT NULL REFERENCES machines(machine_id),
    recorded_at     TIMESTAMPTZ NOT NULL,
    temperature_c   NUMERIC(6,2),
    vibration_hz    NUMERIC(8,4),
    pressure_bar    NUMERIC(6,3),
    power_kw        NUMERIC(7,3),
    rpm             NUMERIC(7,1),
    -- Model outputs (populated by anomaly scoring stage)
    anomaly_flag    BOOLEAN     DEFAULT FALSE,
    anomaly_score   NUMERIC(7,4)
);

-- ─── Fact: Quality Inspections ───────────────────────────────
CREATE TABLE IF NOT EXISTS quality_inspections (
    inspection_id   BIGSERIAL   PRIMARY KEY,
    work_order_id   BIGINT      NOT NULL REFERENCES work_orders(work_order_id),
    inspected_at    TIMESTAMPTZ NOT NULL,
    inspector_id    INTEGER,
    units_inspected INTEGER     NOT NULL,
    units_passed    INTEGER     NOT NULL,
    units_failed    INTEGER     NOT NULL,
    defect_type     VARCHAR(50),
    severity        VARCHAR(10) CHECK (severity IN ('minor','major','critical')),
    sla_breach      BOOLEAN     NOT NULL DEFAULT FALSE
);

-- ─── KPI Snapshots (pipeline output, read by dashboard) ──────
CREATE TABLE IF NOT EXISTS kpi_snapshots (
    snapshot_id         BIGSERIAL   PRIMARY KEY,
    snapshot_date       DATE        NOT NULL,
    plant_id            INTEGER     NOT NULL REFERENCES plants(plant_id),
    oee_score           NUMERIC(6,4),
    availability_rate   NUMERIC(6,4),
    performance_rate    NUMERIC(6,4),
    quality_rate        NUMERIC(6,4),
    mttr_hours          NUMERIC(8,2),
    mtbf_hours          NUMERIC(8,2),
    sla_breach_rate     NUMERIC(6,4),
    defect_rate         NUMERIC(6,4),
    high_risk_wo_count  INTEGER,
    anomaly_count       INTEGER,
    throughput_units    INTEGER,
    UNIQUE (snapshot_date, plant_id)
);

-- ─── Indexes ─────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_wo_created       ON work_orders(created_at);
CREATE INDEX IF NOT EXISTS idx_wo_plant_status  ON work_orders(plant_id, status);
CREATE INDEX IF NOT EXISTS idx_wo_machine       ON work_orders(machine_id);
CREATE INDEX IF NOT EXISTS idx_wo_risk_label    ON work_orders(risk_label) WHERE risk_label = 1;

CREATE INDEX IF NOT EXISTS idx_telemetry_machine_time ON machine_telemetry(machine_id, recorded_at);
CREATE INDEX IF NOT EXISTS idx_telemetry_anomaly      ON machine_telemetry(recorded_at) WHERE anomaly_flag = TRUE;

CREATE INDEX IF NOT EXISTS idx_qi_work_order    ON quality_inspections(work_order_id);
CREATE INDEX IF NOT EXISTS idx_qi_sla_breach    ON quality_inspections(inspected_at) WHERE sla_breach = TRUE;

CREATE INDEX IF NOT EXISTS idx_kpi_date_plant   ON kpi_snapshots(snapshot_date, plant_id);
