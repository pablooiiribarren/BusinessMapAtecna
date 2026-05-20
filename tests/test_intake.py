from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.intake import (
    IntakeRecommendation,
    TypeEstimation,
    build_intake_recommendation,
    classify_dispersion,
)
from src.reliability import MIN_GROUP_SIZE, HISTORICAL_LIMITED_THRESHOLD


def _make_closed_df(rows: list[dict]) -> pd.DataFrame:
    """rows: list of {Owner, Type Name, duration_days}. Fills is_closed=True."""
    df = pd.DataFrame(rows)
    df["is_closed"] = True
    return df


def _make_synthetic_forecast_base(owners: list[str]) -> pd.DataFrame:
    """
    Forecast base con arrival < completion para que todos los owners sean "healthy".
    arrival=0.5, completion=0.6 → net_flow < 0, backlog_days < any horizon.
    """
    idx = pd.Index(owners, name="Owner")
    return pd.DataFrame(
        {
            "arrival_rate_per_day": 0.5,
            "arrival_rate_std_per_day": 0.1,
            "completion_rate_per_day": 0.6,
            "completion_rate_std_per_day": 0.1,
            "current_wip": 2,
        },
        index=idx,
    )


def test_basic_recommendation_structure():
    """End-to-end con dataset sintético: estructura, tiers y labels correctos."""
    rows = (
        [{"Owner": "Alice", "Type Name": "T1", "duration_days": 8.0}] * 12
        + [{"Owner": "Bob", "Type Name": "T1", "duration_days": 12.0}] * 6
        + [{"Owner": "Bob", "Type Name": "T2", "duration_days": 30.0}] * 5
        + [{"Owner": "Charlie", "Type Name": "T1", "duration_days": 10.0}] * 2
    )
    df = _make_closed_df(rows)
    forecast_base = _make_synthetic_forecast_base(["Alice", "Bob", "Charlie"])
    reference_date = pd.Timestamp("2024-01-01")
    start_date = reference_date + pd.Timedelta(days=7)

    rec = build_intake_recommendation(
        df=df,
        forecast_base=forecast_base,
        type_name="T1",
        start_date=start_date,
        reference_date=reference_date,
    )

    # Estimación del tipo
    assert rec.type_estimation.type_name == "T1"
    assert rec.type_estimation.n_type == 12 + 6 + 2  # todos los T1 efectivos

    # candidates_with_history: solo Alice (12) y Bob (6); Charlie (2) queda fuera
    assert len(rec.candidates_with_history) == 2
    hist = rec.candidates_with_history
    alice_row = hist.loc[hist["Owner"] == "Alice"].iloc[0]
    bob_row = hist.loc[hist["Owner"] == "Bob"].iloc[0]

    assert alice_row["n_in_type"] == 12
    assert alice_row["tier_in_type"] == "reliable"
    assert bob_row["n_in_type"] == 6
    assert bob_row["tier_in_type"] == "limited"

    # candidates_without_history: Charlie con n_total=2 → label "Sin histórico"
    nohist = rec.candidates_without_history
    charlie_rows = nohist.loc[nohist["Owner"] == "Charlie"]
    assert len(charlie_rows) == 1
    assert "Sin histórico" in charlie_rows.iloc[0]["label"]

    # horizon correcto
    assert rec.metadata["horizon_days"] == 7


def test_dispersion_classification():
    """classify_dispersion produce ratio, level y warning correctos."""
    # ratio = 1.2 → baja, sin warning
    ratio, level, warning = classify_dispersion(10.0, 12.0)
    assert np.isclose(ratio, 1.2)
    assert level == "baja"
    assert warning is None

    # ratio = 2.0 → media, sin warning
    ratio, level, warning = classify_dispersion(10.0, 20.0)
    assert np.isclose(ratio, 2.0)
    assert level == "media"
    assert warning is None

    # ratio = 3.0 → alta, con warning
    ratio, level, warning = classify_dispersion(10.0, 30.0)
    assert np.isclose(ratio, 3.0)
    assert level == "alta"
    assert warning is not None
    assert "30" in warning

    # median = 0 → NaN, baja, sin warning
    ratio, level, warning = classify_dispersion(0.0, 20.0)
    assert np.isnan(ratio)
    assert level == "baja"
    assert warning is None


def test_horizon_calculation():
    """horizon_days = max(1, (start_date - reference_date).days)."""
    rows = [{"Owner": "Alice", "Type Name": "T1", "duration_days": 8.0}] * 12
    df = _make_closed_df(rows)
    forecast_base = _make_synthetic_forecast_base(["Alice"])
    reference_date = pd.Timestamp("2024-06-01")

    # start_date == reference_date → horizon = 1
    rec = build_intake_recommendation(df, forecast_base, "T1", reference_date, reference_date)
    assert rec.metadata["horizon_days"] == 1

    # start_date en el pasado → horizon = 1
    rec = build_intake_recommendation(
        df, forecast_base, "T1",
        reference_date - pd.Timedelta(days=5),
        reference_date,
    )
    assert rec.metadata["horizon_days"] == 1

    # start_date + 14 días → horizon = 14
    rec = build_intake_recommendation(
        df, forecast_base, "T1",
        reference_date + pd.Timedelta(days=14),
        reference_date,
    )
    assert rec.metadata["horizon_days"] == 14


def test_invalid_type_raises():
    """ValueError si type_name no existe, con mensaje que incluye los tipos disponibles."""
    rows = [{"Owner": "Alice", "Type Name": "T1", "duration_days": 8.0}] * 5
    df = _make_closed_df(rows)
    forecast_base = _make_synthetic_forecast_base(["Alice"])
    reference_date = pd.Timestamp("2024-01-01")
    start_date = reference_date + pd.Timedelta(days=7)

    with pytest.raises(ValueError) as exc_info:
        build_intake_recommendation(
            df, forecast_base, "TIPO_INEXISTENTE", start_date, reference_date
        )

    message = str(exc_info.value)
    assert "TIPO_INEXISTENTE" in message
    assert "T1" in message  # tipo disponible incluido en el mensaje
