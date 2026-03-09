from __future__ import annotations

import numpy as np
import pandas as pd


def build_duration_benchmarks(
    df: pd.DataFrame,
    min_group_size: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    closed = df.loc[
        df["is_closed"] & df["duration_days"].notna() & (df["duration_days"] >= 0)
    ].copy()

    owner_type = (
        closed.groupby(["Owner", "Type Name"])["duration_days"]
        .agg(
            owner_type_count="count",
            owner_type_median="median",
            owner_type_p90=lambda series: series.quantile(0.90),
        )
        .reset_index()
    )
    owner_type = owner_type.loc[owner_type["owner_type_count"] >= min_group_size].copy()

    type_only = (
        closed.groupby("Type Name")["duration_days"]
        .agg(
            type_count="count",
            type_median="median",
            type_p90=lambda series: series.quantile(0.90),
        )
        .reset_index()
    )

    global_median = float(closed["duration_days"].median())
    global_p90 = float(closed["duration_days"].quantile(0.90))
    return owner_type, type_only, global_median, global_p90


def build_alert_reason(row: pd.Series) -> str:
    reasons: list[str] = []

    if row["flag_old_open"]:
        reasons.append("old_open")
    if row["flag_stagnant"]:
        reasons.append("stagnant")
    if row["flag_old_vs_history"]:
        reasons.append("older_than_history")
    if row["flag_dependency_risk"]:
        reasons.append("dependency_risk")
    if row["flag_complexity_risk"]:
        reasons.append("complexity_risk")
    if row["flag_blocked_current"]:
        reasons.append("currently_blocked")
    elif row["flag_blocked_history"]:
        reasons.append("historical_block")
    if row["flag_owner_pressure"]:
        reasons.append(f"owner_{row['owner_forecast_status']}")

    return "; ".join(reasons) if reasons else "healthy"


def build_open_tasks(
    df: pd.DataFrame,
    forecast_dashboard: pd.DataFrame,
    forecast_reference_date: pd.Timestamp,
    min_group_size: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    owner_type_bench, type_bench, global_duration_median, global_duration_p90 = (
        build_duration_benchmarks(df, min_group_size=min_group_size)
    )

    open_tasks = df.loc[df["Actual End Date"].isna()].copy()
    open_tasks["Owner"] = open_tasks["Owner"].fillna("UNASSIGNED")
    open_tasks["Type Name"] = open_tasks["Type Name"].fillna("UNKNOWN")
    open_tasks["Column Name"] = open_tasks["Column Name"].fillna("UNKNOWN")
    if "task_status" not in open_tasks.columns:
        open_tasks["task_status"] = "open"

    if "Last Moved" in open_tasks.columns:
        last_moved = pd.to_datetime(open_tasks["Last Moved"], errors="coerce").dt.floor("D")
    else:
        last_moved = pd.Series(pd.NaT, index=open_tasks.index)
    open_tasks["days_since_last_moved"] = (
        pd.Timestamp(forecast_reference_date).floor("D") - last_moved
    ).dt.total_seconds() / 86400

    if "Blocked State" in open_tasks.columns:
        open_tasks["blocked_state"] = open_tasks["Blocked State"].fillna("no")
    else:
        open_tasks["blocked_state"] = "no"

    if "Block Time (hours)" in open_tasks.columns:
        open_tasks["block_time_hours"] = pd.to_numeric(
            open_tasks["Block Time (hours)"],
            errors="coerce",
        ).fillna(0)
    else:
        open_tasks["block_time_hours"] = 0.0

    open_tasks = open_tasks.merge(owner_type_bench, on=["Owner", "Type Name"], how="left")
    open_tasks = open_tasks.merge(type_bench, on="Type Name", how="left")
    open_tasks = open_tasks.merge(
        forecast_dashboard[
            ["Owner", "forecast_wip", "status", "status_reason"]
        ].rename(
            columns={
                "forecast_wip": "owner_forecast_wip",
                "status": "owner_forecast_status",
                "status_reason": "owner_forecast_status_reason",
            }
        ),
        on="Owner",
        how="left",
    )

    type_open_age_thresholds = (
        open_tasks.groupby("Type Name")["age_days"]
        .agg(
            type_open_age_p75=lambda series: series.quantile(0.75),
            type_open_age_p90=lambda series: series.quantile(0.90),
        )
        .reset_index()
    )
    open_tasks = open_tasks.merge(type_open_age_thresholds, on="Type Name", how="left")

    open_tasks["benchmark_source"] = np.select(
        [open_tasks["owner_type_median"].notna(), open_tasks["type_median"].notna()],
        ["owner_type", "type"],
        default="global",
    )
    open_tasks["benchmark_median_days"] = (
        open_tasks["owner_type_median"]
        .fillna(open_tasks["type_median"])
        .fillna(global_duration_median)
        .clip(lower=1)
    )
    open_tasks["benchmark_p90_days"] = (
        open_tasks["owner_type_p90"]
        .fillna(open_tasks["type_p90"])
        .fillna(global_duration_p90)
        .clip(lower=1)
    )
    open_tasks["age_vs_benchmark"] = (
        open_tasks["age_days"] / open_tasks["benchmark_median_days"]
    )
    open_tasks["owner_forecast_status"] = open_tasks["owner_forecast_status"].fillna(
        "healthy"
    )
    open_tasks["owner_forecast_status_reason"] = open_tasks[
        "owner_forecast_status_reason"
    ].fillna("capacity aligned with horizon")
    open_tasks["owner_forecast_wip"] = pd.to_numeric(
        open_tasks["owner_forecast_wip"],
        errors="coerce",
    ).fillna(0).round(2)

    open_age_p75 = float(open_tasks["age_days"].quantile(0.75))
    open_age_p90 = float(open_tasks["age_days"].quantile(0.90))
    stagnation_base = open_tasks["days_since_last_moved"].dropna()
    stagnation_p75 = stagnation_base.quantile(0.75) if len(stagnation_base) else np.nan
    stagnation_p90 = stagnation_base.quantile(0.90) if len(stagnation_base) else np.nan
    history_absolute_threshold_days = 60
    complexity_threshold = max(
        3,
        int(np.ceil(open_tasks["subtasks_count"].fillna(0).quantile(0.75))),
    )

    open_tasks["effective_open_age_p75"] = open_tasks["type_open_age_p75"].fillna(
        open_age_p75
    )
    open_tasks["effective_open_age_p90"] = open_tasks["type_open_age_p90"].fillna(
        open_age_p90
    )

    open_tasks["flag_old_open"] = (
        open_tasks["age_days"] >= open_tasks["effective_open_age_p75"]
    )
    open_tasks["flag_very_old_open"] = (
        open_tasks["age_days"] >= open_tasks["effective_open_age_p90"]
    )
    open_tasks["flag_stagnant"] = (
        pd.notna(open_tasks["days_since_last_moved"])
        & (open_tasks["days_since_last_moved"] >= stagnation_p75)
    )
    open_tasks["flag_very_stagnant"] = (
        pd.notna(open_tasks["days_since_last_moved"])
        & (open_tasks["days_since_last_moved"] >= stagnation_p90)
    )
    open_tasks["flag_old_vs_history"] = (
        (
            (open_tasks["age_days"] >= open_tasks["benchmark_p90_days"])
            & (open_tasks["age_days"] >= history_absolute_threshold_days)
        )
        | (
            (open_tasks["age_vs_benchmark"] >= 3)
            & (open_tasks["age_days"] >= history_absolute_threshold_days)
        )
    )
    open_tasks["flag_dependency_risk"] = open_tasks["has_any_links"].eq(1) & (
        open_tasks["flag_old_open"] | open_tasks["flag_stagnant"]
    )
    open_tasks["flag_complexity_risk"] = open_tasks["subtasks_count"].ge(
        complexity_threshold
    ) & (open_tasks["flag_old_open"] | open_tasks["flag_stagnant"])
    open_tasks["flag_blocked_current"] = open_tasks["blocked_state"].eq("yes")
    open_tasks["flag_blocked_history"] = open_tasks["block_time_hours"].gt(0)
    open_tasks["flag_owner_pressure"] = open_tasks["owner_forecast_status"].isin(
        ["risk", "overload", "no_throughput"]
    )

    open_tasks["alert_score"] = (
        open_tasks["flag_old_open"].astype(int)
        + 2 * open_tasks["flag_stagnant"].astype(int)
        + open_tasks["flag_old_vs_history"].astype(int)
        + 2 * open_tasks["flag_dependency_risk"].astype(int)
        + open_tasks["flag_complexity_risk"].astype(int)
        + 2 * open_tasks["flag_blocked_current"].astype(int)
        + open_tasks["flag_blocked_history"].astype(int)
        + open_tasks["flag_owner_pressure"].astype(int)
    )

    open_tasks["alert_level"] = np.select(
        [
            open_tasks["flag_blocked_current"]
            & (open_tasks["flag_stagnant"] | open_tasks["flag_old_vs_history"]),
            open_tasks["flag_dependency_risk"] & open_tasks["flag_very_old_open"],
            open_tasks["flag_very_stagnant"] & open_tasks["flag_owner_pressure"],
            open_tasks["alert_score"] >= 5,
            open_tasks["alert_score"] >= 2,
        ],
        ["bottleneck", "bottleneck", "bottleneck", "bottleneck", "risk"],
        default="healthy",
    )
    open_tasks["alert_reason"] = open_tasks.apply(build_alert_reason, axis=1)

    bottleneck_thresholds = pd.DataFrame(
        {
            "metric": [
                "open_age_p75",
                "open_age_p90",
                "stagnation_p75",
                "stagnation_p90",
                "history_absolute_threshold_days",
                "complexity_threshold",
            ],
            "value": [
                round(open_age_p75, 2),
                round(open_age_p90, 2),
                round(stagnation_p75, 2) if pd.notna(stagnation_p75) else np.nan,
                round(stagnation_p90, 2) if pd.notna(stagnation_p90) else np.nan,
                history_absolute_threshold_days,
                complexity_threshold,
            ],
        }
    )

    return open_tasks, bottleneck_thresholds, type_open_age_thresholds


def build_task_alerts(open_tasks: pd.DataFrame) -> pd.DataFrame:
    task_alerts = open_tasks[
        [
            "Card ID",
            "Owner",
            "Column Name",
            "task_status",
            "Type Name",
            "age_days",
            "days_since_last_moved",
            "subtasks_count",
            "has_any_links",
            "owner_forecast_wip",
            "owner_forecast_status",
            "owner_forecast_status_reason",
            "benchmark_source",
            "benchmark_median_days",
            "benchmark_p90_days",
            "age_vs_benchmark",
            "blocked_state",
            "block_time_hours",
            "alert_score",
            "alert_level",
            "alert_reason",
        ]
    ].copy()

    numeric_columns = [
        "age_days",
        "days_since_last_moved",
        "subtasks_count",
        "owner_forecast_wip",
        "benchmark_median_days",
        "benchmark_p90_days",
        "age_vs_benchmark",
        "block_time_hours",
        "alert_score",
    ]
    for column in numeric_columns:
        task_alerts[column] = pd.to_numeric(task_alerts[column], errors="coerce").round(2)

    alert_order = {"bottleneck": 0, "risk": 1, "healthy": 2}
    task_alerts["alert_rank"] = task_alerts["alert_level"].map(alert_order)
    task_alerts = task_alerts.sort_values(
        ["alert_rank", "alert_score", "age_days"],
        ascending=[True, False, False],
    ).drop(columns="alert_rank")

    return task_alerts.reset_index(drop=True)


def build_owner_bottlenecks(
    open_tasks: pd.DataFrame,
    forecast_dashboard: pd.DataFrame,
) -> pd.DataFrame:
    owner_top_column = (
        open_tasks.groupby(["Owner", "Column Name"])
        .size()
        .rename("tasks_in_column")
        .reset_index()
        .sort_values(["Owner", "tasks_in_column", "Column Name"], ascending=[True, False, True])
        .drop_duplicates(subset="Owner")
        .rename(
            columns={
                "Column Name": "dominant_column",
                "tasks_in_column": "dominant_column_tasks",
            }
        )
    )

    owner_bottlenecks = (
        open_tasks.groupby("Owner")
        .agg(
            open_tasks=("Card ID", "count"),
            risk_tasks=("alert_level", lambda series: (series == "risk").sum()),
            bottleneck_tasks=("alert_level", lambda series: (series == "bottleneck").sum()),
            old_open_tasks=("flag_old_open", "sum"),
            stagnant_tasks=("flag_stagnant", "sum"),
            dependency_risk_tasks=("flag_dependency_risk", "sum"),
            complexity_risk_tasks=("flag_complexity_risk", "sum"),
            currently_blocked_tasks=("flag_blocked_current", "sum"),
            historical_block_tasks=("flag_blocked_history", "sum"),
            median_open_age_days=("age_days", "median"),
            max_open_age_days=("age_days", "max"),
        )
        .reset_index()
    )

    owner_bottlenecks = owner_bottlenecks.merge(owner_top_column, on="Owner", how="left")
    owner_bottlenecks = owner_bottlenecks.merge(
        forecast_dashboard[
            ["Owner", "forecast_wip", "status", "status_reason"]
        ].rename(
            columns={
                "status": "forecast_status",
                "status_reason": "forecast_status_reason",
            }
        ),
        on="Owner",
        how="left",
    )

    owner_bottlenecks["forecast_status"] = owner_bottlenecks["forecast_status"].fillna(
        "healthy"
    )
    owner_bottlenecks["forecast_status_reason"] = owner_bottlenecks[
        "forecast_status_reason"
    ].fillna("capacity aligned with horizon")
    owner_bottlenecks["forecast_wip"] = pd.to_numeric(
        owner_bottlenecks["forecast_wip"], errors="coerce"
    ).fillna(0)
    owner_bottlenecks["bottleneck_score"] = (
        2 * owner_bottlenecks["bottleneck_tasks"]
        + owner_bottlenecks["risk_tasks"]
        + owner_bottlenecks["dependency_risk_tasks"]
        + owner_bottlenecks["currently_blocked_tasks"]
        + owner_bottlenecks["forecast_status"].isin(["overload", "no_throughput"]).astype(int)
        + owner_bottlenecks["forecast_status"].eq("risk").astype(int)
    )
    owner_bottlenecks["bottleneck_status"] = np.select(
        [
            owner_bottlenecks["forecast_status"].isin(["overload", "no_throughput"])
            | owner_bottlenecks["bottleneck_tasks"].gt(0),
            owner_bottlenecks["forecast_status"].eq("risk")
            | owner_bottlenecks["risk_tasks"].gt(0),
        ],
        ["bottleneck", "risk"],
        default="healthy",
    )

    numeric_columns = [
        "open_tasks",
        "risk_tasks",
        "bottleneck_tasks",
        "old_open_tasks",
        "stagnant_tasks",
        "dependency_risk_tasks",
        "complexity_risk_tasks",
        "currently_blocked_tasks",
        "historical_block_tasks",
        "median_open_age_days",
        "max_open_age_days",
        "dominant_column_tasks",
        "forecast_wip",
        "bottleneck_score",
    ]
    for column in numeric_columns:
        owner_bottlenecks[column] = pd.to_numeric(
            owner_bottlenecks[column], errors="coerce"
        ).round(2)

    status_order = {"bottleneck": 0, "risk": 1, "healthy": 2}
    owner_bottlenecks["status_rank"] = owner_bottlenecks["bottleneck_status"].map(
        status_order
    )
    owner_bottlenecks = owner_bottlenecks.sort_values(
        ["status_rank", "bottleneck_score", "open_tasks"],
        ascending=[True, False, False],
    ).drop(columns="status_rank")

    return owner_bottlenecks.reset_index(drop=True)


def build_column_bottlenecks(open_tasks: pd.DataFrame) -> pd.DataFrame:
    column_bottlenecks = (
        open_tasks.groupby("Column Name")
        .agg(
            open_tasks=("Card ID", "count"),
            risk_tasks=("alert_level", lambda series: (series == "risk").sum()),
            bottleneck_tasks=("alert_level", lambda series: (series == "bottleneck").sum()),
            owners_involved=("Owner", "nunique"),
            median_age_days=("age_days", "median"),
            max_age_days=("age_days", "max"),
            mean_days_since_last_moved=("days_since_last_moved", "mean"),
            dependency_risk_tasks=("flag_dependency_risk", "sum"),
            currently_blocked_tasks=("flag_blocked_current", "sum"),
            historical_block_tasks=("flag_blocked_history", "sum"),
        )
        .reset_index()
        .rename(columns={"Column Name": "column_name"})
    )

    column_bottlenecks["bottleneck_status"] = np.select(
        [
            column_bottlenecks["bottleneck_tasks"].gt(0),
            column_bottlenecks["risk_tasks"].gt(0),
        ],
        ["bottleneck", "risk"],
        default="healthy",
    )

    numeric_columns = [
        "open_tasks",
        "risk_tasks",
        "bottleneck_tasks",
        "owners_involved",
        "median_age_days",
        "max_age_days",
        "mean_days_since_last_moved",
        "dependency_risk_tasks",
        "currently_blocked_tasks",
        "historical_block_tasks",
    ]
    for column in numeric_columns:
        column_bottlenecks[column] = pd.to_numeric(
            column_bottlenecks[column], errors="coerce"
        ).round(2)

    status_order = {"bottleneck": 0, "risk": 1, "healthy": 2}
    column_bottlenecks["status_rank"] = column_bottlenecks["bottleneck_status"].map(
        status_order
    )
    column_bottlenecks = column_bottlenecks.sort_values(
        ["status_rank", "bottleneck_tasks", "open_tasks"],
        ascending=[True, False, False],
    ).drop(columns="status_rank")

    return column_bottlenecks.reset_index(drop=True)


def build_type_bottlenecks(
    open_tasks: pd.DataFrame,
    type_open_age_thresholds: pd.DataFrame,
) -> pd.DataFrame:
    type_bottlenecks = (
        open_tasks.groupby("Type Name")
        .agg(
            open_tasks=("Card ID", "count"),
            risk_tasks=("alert_level", lambda series: (series == "risk").sum()),
            bottleneck_tasks=("alert_level", lambda series: (series == "bottleneck").sum()),
            median_age_days=("age_days", "median"),
            max_age_days=("age_days", "max"),
            median_days_since_last_moved=("days_since_last_moved", "median"),
            mean_age_vs_benchmark=("age_vs_benchmark", "mean"),
            dependency_risk_tasks=("flag_dependency_risk", "sum"),
            complexity_risk_tasks=("flag_complexity_risk", "sum"),
            currently_blocked_tasks=("flag_blocked_current", "sum"),
        )
        .reset_index()
        .rename(columns={"Type Name": "type_name"})
    )

    type_bottlenecks = type_bottlenecks.merge(
        type_open_age_thresholds.rename(columns={"Type Name": "type_name"}),
        on="type_name",
        how="left",
    )

    type_bottlenecks["bottleneck_status"] = np.select(
        [
            type_bottlenecks["bottleneck_tasks"].gt(0),
            type_bottlenecks["risk_tasks"].gt(0),
        ],
        ["bottleneck", "risk"],
        default="healthy",
    )

    numeric_columns = [
        "open_tasks",
        "risk_tasks",
        "bottleneck_tasks",
        "median_age_days",
        "max_age_days",
        "median_days_since_last_moved",
        "mean_age_vs_benchmark",
        "dependency_risk_tasks",
        "complexity_risk_tasks",
        "currently_blocked_tasks",
        "type_open_age_p75",
        "type_open_age_p90",
    ]
    for column in numeric_columns:
        type_bottlenecks[column] = pd.to_numeric(
            type_bottlenecks[column], errors="coerce"
        ).round(2)

    status_order = {"bottleneck": 0, "risk": 1, "healthy": 2}
    type_bottlenecks["status_rank"] = type_bottlenecks["bottleneck_status"].map(
        status_order
    )
    type_bottlenecks = type_bottlenecks.sort_values(
        ["status_rank", "bottleneck_tasks", "open_tasks"],
        ascending=[True, False, False],
    ).drop(columns="status_rank")

    return type_bottlenecks.reset_index(drop=True)


def build_bottleneck_artifacts(
    df: pd.DataFrame,
    forecast_dashboard: pd.DataFrame,
    forecast_reference_date: pd.Timestamp,
    min_group_size: int = 5,
) -> dict[str, pd.DataFrame]:
    open_tasks, bottleneck_thresholds, type_open_age_thresholds = build_open_tasks(
        df,
        forecast_dashboard=forecast_dashboard,
        forecast_reference_date=forecast_reference_date,
        min_group_size=min_group_size,
    )

    task_alerts = build_task_alerts(open_tasks)
    owner_bottlenecks = build_owner_bottlenecks(open_tasks, forecast_dashboard)
    column_bottlenecks = build_column_bottlenecks(open_tasks)
    type_bottlenecks = build_type_bottlenecks(open_tasks, type_open_age_thresholds)

    return {
        "open_tasks": open_tasks,
        "bottleneck_thresholds": bottleneck_thresholds,
        "task_alerts": task_alerts,
        "owner_bottlenecks": owner_bottlenecks,
        "column_bottlenecks": column_bottlenecks,
        "type_bottlenecks": type_bottlenecks,
    }
