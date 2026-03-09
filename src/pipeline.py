from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .bottlenecks import build_bottleneck_artifacts
from .data_prep import prepare_businessmap_dataset
from .forecast import (
    build_forecast_dashboard,
    build_forecast_inputs,
    build_forecast_scenarios,
)


def run_baseline_pipeline(
    workbook_path: str | Path,
    rate_window_days: int = 60,
    dashboard_horizon_days: int = 5,
    scenario_horizons: Iterable[int] = (5, 10, 20),
    min_group_size: int = 5,
) -> dict[str, pd.DataFrame | pd.Timestamp]:
    prepared = prepare_businessmap_dataset(workbook_path)
    bm = prepared["bm"]
    reference_date = prepared["reference_date"]

    forecast_inputs = build_forecast_inputs(
        bm,
        reference_date=reference_date,
        rate_window_days=rate_window_days,
    )
    forecast_base = forecast_inputs["forecast_base"]
    forecast_reference_date = forecast_inputs["forecast_reference_date"]

    forecast_dashboard = build_forecast_dashboard(
        forecast_base,
        horizon_days=dashboard_horizon_days,
        rate_window_days=rate_window_days,
        reference_date=forecast_reference_date,
    )
    forecast_dashboard_export = forecast_dashboard[
        [
            "Owner",
            "arrival_rate_per_day",
            "completion_rate_per_day",
            "current_wip",
            "expected_arrivals",
            "expected_completions",
            "forecast_wip",
            "status",
            "status_reason",
        ]
    ].copy()
    forecast_scenarios = build_forecast_scenarios(
        forecast_base,
        horizons=scenario_horizons,
        rate_window_days=rate_window_days,
        reference_date=forecast_reference_date,
    )

    bottlenecks = build_bottleneck_artifacts(
        bm,
        forecast_dashboard=forecast_dashboard,
        forecast_reference_date=forecast_reference_date,
        min_group_size=min_group_size,
    )

    return {
        "bm": bm,
        "links": prepared["links"],
        "subtasks": prepared["subtasks"],
        "reference_date": reference_date,
        "forecast_reference_date": forecast_reference_date,
        "forecast_base": forecast_base,
        "arrivals": forecast_inputs["arrivals"],
        "completions": forecast_inputs["completions"],
        "arrivals_daily": forecast_inputs["arrivals_daily"],
        "completions_daily": forecast_inputs["completions_daily"],
        "forecast_dashboard": forecast_dashboard,
        "forecast_dashboard_export": forecast_dashboard_export,
        "forecast_scenarios": forecast_scenarios,
        "bottleneck_thresholds": bottlenecks["bottleneck_thresholds"],
        "task_alerts": bottlenecks["task_alerts"],
        "owner_bottlenecks": bottlenecks["owner_bottlenecks"],
        "column_bottlenecks": bottlenecks["column_bottlenecks"],
        "type_bottlenecks": bottlenecks["type_bottlenecks"],
        "open_tasks": bottlenecks["open_tasks"],
    }
