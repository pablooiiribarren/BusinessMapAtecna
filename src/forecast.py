from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def build_forecast_inputs(
    df: pd.DataFrame,
    reference_date: pd.Timestamp,
    rate_window_days: int = 60,
) -> dict[str, object]:
    bm = df.copy()
    bm["Created Date"] = pd.to_datetime(bm["Created At"], errors="coerce").dt.floor("D")
    bm["Closed Date"] = pd.to_datetime(bm["Actual End Date"], errors="coerce").dt.floor("D")

    forecast_reference_date = pd.Timestamp(reference_date).floor("D")
    window_start = forecast_reference_date - pd.Timedelta(days=rate_window_days - 1)
    window_dates = pd.date_range(window_start, forecast_reference_date, freq="D")
    owner_index = pd.Index(sorted(bm["Owner"].dropna().unique()), name="Owner")

    arrivals = (
        bm.loc[
            bm["Created Date"].between(window_start, forecast_reference_date)
            & bm["Owner"].notna()
        ]
        .groupby(["Owner", "Created Date"])
        .size()
        .rename("arrivals")
        .reset_index()
    )

    completions = (
        bm.loc[
            bm["Closed Date"].between(window_start, forecast_reference_date)
            & bm["Owner"].notna()
        ]
        .groupby(["Owner", "Closed Date"])
        .size()
        .rename("completions")
        .reset_index()
    )

    arrivals_daily = (
        arrivals.pivot(index="Created Date", columns="Owner", values="arrivals")
        .reindex(index=window_dates, columns=owner_index, fill_value=0)
        .fillna(0)
    )
    completions_daily = (
        completions.pivot(index="Closed Date", columns="Owner", values="completions")
        .reindex(index=window_dates, columns=owner_index, fill_value=0)
        .fillna(0)
    )

    arrival_rate = arrivals_daily.mean(axis=0).rename("arrival_rate_per_day")
    completion_rate = completions_daily.mean(axis=0).rename("completion_rate_per_day")
    wip_now = (
        bm.loc[bm["Actual End Date"].isna() & bm["Owner"].notna()]
        .groupby("Owner")["Card ID"]
        .count()
        .reindex(owner_index, fill_value=0)
        .rename("current_wip")
    )

    forecast_base = pd.concat([arrival_rate, completion_rate, wip_now], axis=1).fillna(0)

    return {
        "bm": bm,
        "forecast_base": forecast_base,
        "forecast_reference_date": forecast_reference_date,
        "window_start": window_start,
        "window_dates": window_dates,
        "arrivals": arrivals,
        "completions": completions,
        "arrivals_daily": arrivals_daily,
        "completions_daily": completions_daily,
        "rate_window_days": rate_window_days,
    }


def forecast_wip(df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    forecast = df.copy()
    forecast["expected_arrivals"] = forecast["arrival_rate_per_day"] * horizon_days
    forecast["expected_completions"] = forecast["completion_rate_per_day"] * horizon_days
    forecast["forecast_wip"] = (
        forecast["current_wip"]
        + forecast["expected_arrivals"]
        - forecast["expected_completions"]
    ).clip(lower=0)
    return forecast


def simulate_new_tasks(
    forecast_df: pd.DataFrame,
    owner: str,
    new_tasks: int,
    horizon_days: int,
) -> pd.DataFrame:
    simulated = forecast_df.copy()
    if owner not in simulated.index:
        raise KeyError(f"Owner '{owner}' no existe en forecast_df.index")

    simulated.loc[owner, "current_wip"] = simulated.loc[owner, "current_wip"] + new_tasks
    return forecast_wip(simulated, horizon_days)


def compute_backlog_days(wip: float, completion_rate: float) -> float:
    if completion_rate > 0:
        return wip / completion_rate
    return np.nan


def classify_forecast_status(row: pd.Series, horizon_days: int) -> pd.Series:
    completion = row["completion_rate_per_day"]
    arrival = row["arrival_rate_per_day"]
    net_flow = row["net_flow_per_day"]
    current_wip = row["current_wip"]
    forecast_backlog_days = row["forecast_backlog_days"]

    if completion <= 0:
        if current_wip > 0 or arrival > 0:
            return pd.Series(
                {
                    "status": "no_throughput",
                    "status_reason": "open work but no recent completions",
                }
            )
        return pd.Series({"status": "healthy", "status_reason": "no recent load"})

    reasons: list[str] = []
    if net_flow > 0:
        reasons.append("arrivals > completions")
    if pd.notna(forecast_backlog_days) and forecast_backlog_days > horizon_days:
        reasons.append("projected backlog exceeds horizon")

    if len(reasons) == 2:
        status = "overload"
    elif len(reasons) == 1:
        status = "risk"
    else:
        status = "healthy"
        reasons.append("capacity aligned with horizon")

    return pd.Series({"status": status, "status_reason": "; ".join(reasons)})


def build_forecast_dashboard(
    forecast_df: pd.DataFrame,
    horizon_days: int,
    rate_window_days: int,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    dashboard = forecast_wip(forecast_df, horizon_days).copy()
    dashboard["window_days"] = rate_window_days
    dashboard["horizon_days"] = horizon_days
    dashboard["snapshot_date"] = pd.Timestamp(reference_date).normalize()
    dashboard["net_flow_per_day"] = (
        dashboard["arrival_rate_per_day"] - dashboard["completion_rate_per_day"]
    )
    dashboard["current_backlog_days"] = dashboard.apply(
        lambda row: compute_backlog_days(
            row["current_wip"],
            row["completion_rate_per_day"],
        ),
        axis=1,
    )
    dashboard["forecast_backlog_days"] = dashboard.apply(
        lambda row: compute_backlog_days(
            row["forecast_wip"],
            row["completion_rate_per_day"],
        ),
        axis=1,
    )
    dashboard["throughput_ratio"] = dashboard.apply(
        lambda row: (
            row["arrival_rate_per_day"] / row["completion_rate_per_day"]
            if row["completion_rate_per_day"] > 0
            else np.nan
        ),
        axis=1,
    )

    status_info = dashboard.apply(
        lambda row: classify_forecast_status(row, horizon_days),
        axis=1,
    )
    dashboard = pd.concat([dashboard, status_info], axis=1)
    dashboard = dashboard.reset_index().rename(columns={"index": "Owner"})

    ordered_columns = [
        "Owner",
        "snapshot_date",
        "window_days",
        "horizon_days",
        "arrival_rate_per_day",
        "completion_rate_per_day",
        "net_flow_per_day",
        "current_wip",
        "expected_arrivals",
        "expected_completions",
        "forecast_wip",
        "current_backlog_days",
        "forecast_backlog_days",
        "throughput_ratio",
        "status",
        "status_reason",
    ]
    dashboard = dashboard[ordered_columns].copy()

    numeric_columns = [
        "arrival_rate_per_day",
        "completion_rate_per_day",
        "net_flow_per_day",
        "current_wip",
        "expected_arrivals",
        "expected_completions",
        "forecast_wip",
        "current_backlog_days",
        "forecast_backlog_days",
        "throughput_ratio",
    ]
    for column in numeric_columns:
        dashboard[column] = pd.to_numeric(dashboard[column], errors="coerce").round(2)

    status_order = {"no_throughput": 0, "overload": 1, "risk": 2, "healthy": 3}
    dashboard["status_rank"] = dashboard["status"].map(status_order)
    dashboard = dashboard.sort_values(
        ["status_rank", "forecast_wip", "current_wip"],
        ascending=[True, False, False],
    ).drop(columns="status_rank")

    return dashboard.reset_index(drop=True)


def build_forecast_scenarios(
    forecast_df: pd.DataFrame,
    horizons: Iterable[int],
    rate_window_days: int,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    scenario_tables: list[pd.DataFrame] = []

    for horizon in horizons:
        scenario_table = build_forecast_dashboard(
            forecast_df,
            horizon_days=horizon,
            rate_window_days=rate_window_days,
            reference_date=reference_date,
        )[["Owner", "forecast_wip", "status"]].rename(
            columns={
                "forecast_wip": f"forecast_wip_{horizon}d",
                "status": f"status_{horizon}d",
            }
        )
        scenario_tables.append(scenario_table)

    if not scenario_tables:
        return pd.DataFrame(columns=["Owner"])

    forecast_scenarios = scenario_tables[0]
    for scenario_table in scenario_tables[1:]:
        forecast_scenarios = forecast_scenarios.merge(
            scenario_table,
            on="Owner",
            how="left",
        )

    return forecast_scenarios
