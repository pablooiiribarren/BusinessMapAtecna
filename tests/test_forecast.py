from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecast import bootstrap_forecast_wip, build_forecast_inputs, forecast_wip


def _make_df(
    owners_config: list[tuple[str, list[int], list[int]]],
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build a minimal BusinessMap-style DataFrame from per-day count arrays."""
    rows = []
    card_id = 0
    for owner, arr_counts, comp_counts in owners_config:
        for i, date in enumerate(dates):
            for _ in range(arr_counts[i]):
                rows.append(
                    {
                        "Owner": owner,
                        "Created At": date,
                        "Actual End Date": pd.NaT,
                        "Card ID": card_id,
                    }
                )
                card_id += 1
            for _ in range(comp_counts[i]):
                rows.append(
                    {
                        "Owner": owner,
                        "Created At": pd.Timestamp("2020-01-01"),
                        "Actual End Date": date,
                        "Card ID": card_id,
                    }
                )
                card_id += 1
    return pd.DataFrame(rows)


def test_sqrt_scaling_in_confidence_interval():
    """CI width must follow sqrt(N) scaling, not linear N scaling."""
    rng_data = np.random.default_rng(42)
    n_days = 60
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    reference_date = dates[-1]
    horizon_days = 20

    owners_config = [
        ("Alice", rng_data.poisson(3, n_days).tolist(), rng_data.poisson(2, n_days).tolist()),
        ("Bob", rng_data.poisson(5, n_days).tolist(), rng_data.poisson(4, n_days).tolist()),
    ]
    df = _make_df(owners_config, dates)

    fi = build_forecast_inputs(df, reference_date=reference_date)
    fb = fi["forecast_base"]
    forecast = forecast_wip(fb, horizon_days=horizon_days)

    for owner in ["Alice", "Bob"]:
        arr_std = fb.loc[owner, "arrival_rate_std_per_day"]
        comp_std = fb.loc[owner, "completion_rate_std_per_day"]
        expected_half_width = np.sqrt(arr_std**2 + comp_std**2) * np.sqrt(horizon_days)

        low = forecast.loc[owner, "forecast_wip_low"]
        high = forecast.loc[owner, "forecast_wip_high"]

        assert low >= 0, "forecast_wip_low must be non-negative"
        assert pd.notna(high), "forecast_wip_high must not be NaN"

        if low > 0:
            actual_width = high - low
            np.testing.assert_allclose(
                actual_width,
                2 * expected_half_width,
                rtol=0.05,
                err_msg=f"CI width for {owner} does not match sqrt(N) formula",
            )


def test_ewm_weights_recent_data():
    """EWM rate must be lower than flat mean when recent activity dropped to zero."""
    n_days = 60
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    reference_date = dates[-1]

    # First 45 days: 5 arrivals/day; last 15 days: 0
    arr_counts = [5] * 45 + [0] * 15
    comp_counts = [0] * n_days

    df = _make_df([("A", arr_counts, comp_counts)], dates)

    fi = build_forecast_inputs(df, reference_date=reference_date, halflife_days=14.0)
    ewm_rate = fi["forecast_base"].loc["A", "arrival_rate_per_day"]

    simple_mean = sum(arr_counts) / n_days  # 3.75

    assert ewm_rate < simple_mean, (
        f"EWM rate ({ewm_rate:.3f}) should be < flat mean ({simple_mean:.3f}) "
        "when recent activity dropped"
    )
    assert ewm_rate < 2.5, (
        f"EWM rate ({ewm_rate:.3f}) should be well below midpoint 2.5 "
        "given 15 days of zero activity with halflife=14"
    )


def test_bootstrap_deterministic_with_seed():
    """bootstrap_forecast_wip must return identical results for the same random_state."""
    n_days = 60
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    reference_date = dates[-1]

    arr_counts = [3] * n_days
    comp_counts = [2] * n_days
    df = _make_df([("X", arr_counts, comp_counts)], dates)

    fi = build_forecast_inputs(df, reference_date=reference_date)

    result1 = bootstrap_forecast_wip(fi, horizon_days=10, n_simulations=200, random_state=42)
    result2 = bootstrap_forecast_wip(fi, horizon_days=10, n_simulations=200, random_state=42)

    pd.testing.assert_frame_equal(result1, result2)
