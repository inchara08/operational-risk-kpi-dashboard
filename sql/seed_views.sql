-- =============================================================
-- Materialized Views for Dashboard & Power BI Performance
-- Run after pipeline scoring + KPI stages are complete
-- =============================================================

-- Daily OEE per plant (primary dashboard timeseries)
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_daily_oee AS
SELECT
    snapshot_date                           AS day,
    p.plant_code,
    p.plant_name,
    k.oee_score,
    k.availability_rate,
    k.performance_rate,
    k.quality_rate,
    k.mttr_hours,
    k.mtbf_hours,
    k.sla_breach_rate,
    k.defect_rate,
    k.high_risk_wo_count,
    k.anomaly_count,
    k.throughput_units
FROM kpi_snapshots k
JOIN plants p USING (plant_id)
WITH DATA;

CREATE UNIQUE INDEX IF NOT EXISTS uidx_mv_daily_oee ON mv_daily_oee(day, plant_code);

-- Top high-risk open work orders (refreshed by pipeline)
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_top_risk_work_orders AS
SELECT
    wo.work_order_id,
    wo.risk_score,
    wo.risk_label,
    wo.priority,
    wo.product_type,
    wo.status,
    wo.scheduled_start,
    wo.scheduled_end,
    wo.planned_units,
    wo.defect_count,
    wo.downtime_minutes,
    p.plant_name,
    m.machine_code,
    m.machine_type
FROM work_orders wo
JOIN plants   p USING (plant_id)
JOIN machines m USING (machine_id)
WHERE wo.risk_label = 1
ORDER BY wo.risk_score DESC
WITH DATA;

-- Plant-level summary (for executive KPI cards)
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_plant_summary AS
SELECT
    p.plant_id,
    p.plant_code,
    p.plant_name,
    AVG(k.oee_score)        AS avg_oee,
    AVG(k.mttr_hours)       AS avg_mttr_hours,
    AVG(k.sla_breach_rate)  AS avg_sla_breach_rate,
    AVG(k.defect_rate)      AS avg_defect_rate,
    SUM(k.high_risk_wo_count)  AS total_high_risk_wos,
    SUM(k.anomaly_count)       AS total_anomalies,
    SUM(k.throughput_units)    AS total_throughput
FROM kpi_snapshots k
JOIN plants p USING (plant_id)
GROUP BY p.plant_id, p.plant_code, p.plant_name
WITH DATA;

-- Helper: refresh all views in one call
CREATE OR REPLACE FUNCTION refresh_all_views()
RETURNS void LANGUAGE plpgsql AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_daily_oee;
    REFRESH MATERIALIZED VIEW mv_top_risk_work_orders;
    REFRESH MATERIALIZED VIEW mv_plant_summary;
END;
$$;
