"""
Aggregate telemetry sensor readings per machine per work order window.

For each work order, we look back 24 hours of telemetry for the assigned
machine and compute statistical aggregates. These become the "sensor health"
features in the risk classifier.
"""

from __future__ import annotations

import pandas as pd


def aggregate_telemetry_per_machine(telemetry: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-aggregate telemetry to hourly machine-level stats.
    Returns a DataFrame indexed by (machine_id, hour).
    """
    tel = telemetry.copy()
    tel["recorded_at"] = pd.to_datetime(tel["recorded_at"], utc=True, errors="coerce")
    tel["hour"] = tel["recorded_at"].dt.floor("h")

    agg = tel.groupby(["machine_id", "hour"]).agg(
        temp_mean=("temperature_c", "mean"),
        temp_std=("temperature_c", "std"),
        temp_max=("temperature_c", "max"),
        vib_mean=("vibration_hz", "mean"),
        vib_std=("vibration_hz", "std"),
        vib_max=("vibration_hz", "max"),
        pres_mean=("pressure_bar", "mean"),
        pres_deviation=("pressure_bar", "std"),
        pwr_mean=("power_kw", "mean"),
        pwr_spike_count=("power_kw", lambda x: (x > x.mean() + 2 * x.std()).sum()),
        rpm_mean=("rpm", "mean"),
        rpm_cv=("rpm", lambda x: x.std() / x.mean() if x.mean() > 0 else 0),
        anomaly_count=("anomaly_flag", lambda x: pd.to_numeric(x, errors="coerce").fillna(0).sum()),
    ).reset_index()

    return agg.fillna(0)


def join_telemetry_to_work_orders(
    wo: pd.DataFrame,
    tel_agg: pd.DataFrame,
    lookback_hours: int = 24,
) -> pd.DataFrame:
    """
    For each work order, compute mean of telemetry aggregates in the
    [scheduled_start - lookback_hours, scheduled_start] window.

    Returns work_orders with new telemetry feature columns appended.
    """
    wo = wo.copy()
    wo["scheduled_start"] = pd.to_datetime(wo["scheduled_start"], utc=True, errors="coerce")

    tel_feature_cols = [
        "temp_mean", "temp_std", "temp_max",
        "vib_mean", "vib_std", "vib_max",
        "pres_mean", "pres_deviation",
        "pwr_mean", "pwr_spike_count",
        "rpm_mean", "rpm_cv",
        "anomaly_count",
    ]

    # Initialize feature columns
    for col in tel_feature_cols:
        wo[f"tel_{col}"] = 0.0

    tel_agg["hour"] = pd.to_datetime(tel_agg["hour"], utc=True, errors="coerce")

    for machine_id, group in wo.groupby("machine_id"):
        machine_tel = tel_agg[tel_agg["machine_id"] == machine_id].set_index("hour")

        for idx, row in group.iterrows():
            window_end = row["scheduled_start"]
            window_start = window_end - pd.Timedelta(hours=lookback_hours)
            mask = (machine_tel.index >= window_start) & (machine_tel.index < window_end)
            window_data = machine_tel.loc[mask, tel_feature_cols]

            if len(window_data) > 0:
                means = window_data.mean()
                for col in tel_feature_cols:
                    wo.at[idx, f"tel_{col}"] = means[col]

    return wo
