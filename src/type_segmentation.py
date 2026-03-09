from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .bottlenecks import build_bottleneck_artifacts
from .forecast import (
    build_forecast_dashboard,
    build_forecast_inputs,
    build_forecast_scenarios,
)

DEFAULT_ANALYSIS_TYPES = (
    "PROYECTO EXTERNO",
    "PROYECTO INTERNO",
    "PRODUCTO",
)


def _resolve_type_values(
    df: pd.DataFrame,
    type_values: Iterable[str] | None = None,
) -> list[str]:
    requested = list(type_values or DEFAULT_ANALYSIS_TYPES)
    available = set(df["Type Name"].dropna().unique())
    return [type_name for type_name in requested if type_name in available]


def combine_type_tables(
    segmented_artifacts: dict[str, object],
    artifact_name: str,
    type_column: str = "analysis_type",
) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for type_name, artifacts in segmented_artifacts["type_artifacts"].items():
        table = artifacts.get(artifact_name)
        if table is None or not isinstance(table, pd.DataFrame):
            continue
        typed_table = table.copy()
        typed_table.insert(0, type_column, type_name)
        tables.append(typed_table)

    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def run_type_segmented_analysis(
    df: pd.DataFrame,
    reference_date: pd.Timestamp,
    rate_window_days: int = 60,
    dashboard_horizon_days: int = 5,
    scenario_horizons: Iterable[int] = (5, 10, 20),
    min_group_size: int = 5,
    type_values: Iterable[str] | None = None,
) -> dict[str, object]:
    analysis_types = _resolve_type_values(df, type_values=type_values)
    type_artifacts: dict[str, dict[str, pd.DataFrame | pd.Timestamp]] = {}
    summary_rows: list[dict[str, object]] = []
    forecast_rows: list[dict[str, object]] = []
    bottleneck_rows: list[dict[str, object]] = []

    for type_name in analysis_types:
        subset = df.loc[df["Type Name"] == type_name].copy()
        if subset.empty:
            continue

        forecast_inputs = build_forecast_inputs(
            subset,
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
            subset,
            forecast_dashboard=forecast_dashboard,
            forecast_reference_date=forecast_reference_date,
            min_group_size=min_group_size,
        )

        type_artifacts[type_name] = {
            "bm": subset,
            "forecast_base": forecast_base,
            "forecast_dashboard": forecast_dashboard,
            "forecast_dashboard_export": forecast_dashboard_export,
            "forecast_scenarios": forecast_scenarios,
            "bottleneck_thresholds": bottlenecks["bottleneck_thresholds"],
            "task_alerts": bottlenecks["task_alerts"],
            "owner_bottlenecks": bottlenecks["owner_bottlenecks"],
            "column_bottlenecks": bottlenecks["column_bottlenecks"],
            "type_bottlenecks": bottlenecks["type_bottlenecks"],
            "open_tasks": bottlenecks["open_tasks"],
            "forecast_reference_date": forecast_reference_date,
        }

        open_subset = subset.loc[subset["Actual End Date"].isna()]
        closed_duration = subset.loc[
            subset["is_closed"] & subset["duration_days"].notna(),
            "duration_days",
        ]

        summary_rows.append(
            {
                "type_name": type_name,
                "n_rows": int(len(subset)),
                "n_closed": int(subset["is_closed"].sum()),
                "n_open": int((~subset["is_closed"]).sum()),
                "owners": int(subset["Owner"].dropna().nunique()),
                "median_closed_duration_days": round(float(closed_duration.median()), 2)
                if len(closed_duration)
                else None,
                "p90_closed_duration_days": round(float(closed_duration.quantile(0.90)), 2)
                if len(closed_duration)
                else None,
                "median_open_age_days": round(float(open_subset["age_days"].median()), 2)
                if len(open_subset)
                else None,
                "max_open_age_days": round(float(open_subset["age_days"].max()), 2)
                if len(open_subset)
                else None,
            }
        )

        forecast_rows.append(
            {
                "type_name": type_name,
                "owners_in_forecast": int(forecast_dashboard.shape[0]),
                "current_wip": round(float(forecast_dashboard["current_wip"].sum()), 2),
                "forecast_wip": round(float(forecast_dashboard["forecast_wip"].sum()), 2),
                "no_throughput_owners": int((forecast_dashboard["status"] == "no_throughput").sum()),
                "overload_owners": int((forecast_dashboard["status"] == "overload").sum()),
                "risk_owners": int((forecast_dashboard["status"] == "risk").sum()),
                "healthy_owners": int((forecast_dashboard["status"] == "healthy").sum()),
            }
        )

        owner_bottlenecks = bottlenecks["owner_bottlenecks"]
        column_bottlenecks = bottlenecks["column_bottlenecks"]
        top_column = (
            column_bottlenecks.iloc[0]["column_name"]
            if not column_bottlenecks.empty
            else None
        )

        bottleneck_rows.append(
            {
                "type_name": type_name,
                "open_tasks": int(bottlenecks["task_alerts"].shape[0]),
                "bottleneck_tasks": int((bottlenecks["task_alerts"]["alert_level"] == "bottleneck").sum()),
                "risk_tasks": int((bottlenecks["task_alerts"]["alert_level"] == "risk").sum()),
                "healthy_tasks": int((bottlenecks["task_alerts"]["alert_level"] == "healthy").sum()),
                "bottleneck_owners": int((owner_bottlenecks["bottleneck_status"] == "bottleneck").sum()),
                "risk_owners": int((owner_bottlenecks["bottleneck_status"] == "risk").sum()),
                "healthy_owners": int((owner_bottlenecks["bottleneck_status"] == "healthy").sum()),
                "top_problem_column": top_column,
            }
        )

    type_summary = pd.DataFrame(summary_rows)
    if not type_summary.empty:
        type_summary = type_summary.sort_values("n_open", ascending=False)

    type_forecast_overview = pd.DataFrame(forecast_rows)
    if not type_forecast_overview.empty:
        type_forecast_overview = type_forecast_overview.sort_values(
            ["forecast_wip", "current_wip"],
            ascending=[False, False],
        )

    type_bottleneck_overview = pd.DataFrame(bottleneck_rows)
    if not type_bottleneck_overview.empty:
        type_bottleneck_overview = type_bottleneck_overview.sort_values(
            ["bottleneck_tasks", "risk_tasks", "open_tasks"],
            ascending=[False, False, False],
        )

    return {
        "analysis_types": analysis_types,
        "type_summary": type_summary.reset_index(drop=True),
        "type_forecast_overview": type_forecast_overview.reset_index(drop=True),
        "type_bottleneck_overview": type_bottleneck_overview.reset_index(drop=True),
        "type_artifacts": type_artifacts,
    }
