from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.reliability import (
    DurationBenchmarks,
    HISTORICAL_LIMITED_THRESHOLD,
    MIN_GROUP_SIZE,
    build_duration_benchmarks,
    compute_owner_reliability,
)


def _make_closed_df(rows: list[dict]) -> pd.DataFrame:
    """rows: list of {Owner, Type Name, duration_days}. Fills is_closed=True."""
    df = pd.DataFrame(rows)
    df["is_closed"] = True
    return df


def test_min_duration_filter_excludes_short_tasks():
    rows = (
        [{"Owner": "Alice", "Type Name": "T1", "duration_days": 0.5}] * 5
        + [{"Owner": "Alice", "Type Name": "T1", "duration_days": 2.0}] * 5
    )
    df = _make_closed_df(rows)

    bench = build_duration_benchmarks(df, min_duration_days=1.0, min_group_size=5)

    assert bench.n_total == 10
    assert bench.n_effective == 5
    assert bench.excluded_pct == 50.0
    assert bench.global_median == 2.0
    assert len(bench.owner_type) == 1


def test_eligibility_tier_thresholds():
    rows = (
        [{"Owner": "Alice", "Type Name": "T1", "duration_days": 5.0}] * 12
        + [{"Owner": "Bob", "Type Name": "T1", "duration_days": 5.0}] * 7
        + [{"Owner": "Charlie", "Type Name": "T1", "duration_days": 5.0}] * 4
    )
    df = _make_closed_df(rows)

    result = compute_owner_reliability(df)

    assert len(result) == 3

    alice = result.loc[result["Owner"] == "Alice"].iloc[0]
    assert alice["eligibility_tier"] == "reliable"
    assert alice["n_effective"] == 12

    bob = result.loc[result["Owner"] == "Bob"].iloc[0]
    assert bob["eligibility_tier"] == "limited"
    assert bob["n_effective"] == 7

    charlie = result.loc[result["Owner"] == "Charlie"].iloc[0]
    assert charlie["eligibility_tier"] == "insufficient"
    assert charlie["n_effective"] == 0
    assert charlie["n_total"] == 4
    assert np.isnan(charlie["deviation_weighted"])


def test_weighted_deviation_simple_case():
    rows = (
        [{"Owner": "Alice", "Type Name": "T1", "duration_days": 10.0}] * 5
        + [{"Owner": "Bob", "Type Name": "T1", "duration_days": 6.0}] * 5
        + [{"Owner": "Alice", "Type Name": "T2", "duration_days": 30.0}] * 5
        + [{"Owner": "Bob", "Type Name": "T2", "duration_days": 10.0}] * 5
    )
    df = _make_closed_df(rows)

    result = compute_owner_reliability(df, min_group_size=5)

    alice = result.loc[result["Owner"] == "Alice"].iloc[0]
    bob = result.loc[result["Owner"] == "Bob"].iloc[0]

    assert np.isclose(alice["deviation_weighted"], 0.375)
    assert np.isclose(bob["deviation_weighted"], -0.375)
    assert alice["eligibility_tier"] == "reliable"
    assert bob["eligibility_tier"] == "reliable"
    assert alice["n_effective"] == 10
    assert bob["n_effective"] == 10
