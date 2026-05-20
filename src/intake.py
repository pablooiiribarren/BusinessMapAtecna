"""
Recomendación de intake para proyectos entrantes.

Dado un tipo de proyecto y una fecha de inicio prevista, produce una estimación
de duración del tipo (mediana, p90, dispersión) y dos rankings de candidatos.

Decisiones de diseño:
1. Ranking "con histórico": status healthy/risk, ordenado por forecast_wip_at_start
   ascendente; desvío como columna informativa, sin score compuesto.
2. Horizonte de capacidad: max(1, (start_date - reference_date).days).
3. Dispersión cualitativa: ratio p90/median; <1.5 baja, 1.5–2.5 media, >2.5 alta.
   Solo el nivel "alta" genera un mensaje de warning.
4. Doble etiqueta en "sin histórico": distingue n_total ≥ MIN_GROUP_SIZE
   (experiencia global) de n_total < MIN_GROUP_SIZE (sin historia).
5. No se filtran owners agrupadores ("Otros 2", "Consultor 1"…); sus desvíos
   extremos son información válida para el PM.
6. API: una función orquestadora build_intake_recommendation devuelve un
   IntakeRecommendation con estimación, dos rankings y metadata.

Ejemplo mínimo de uso::

    from src.pipeline import load_data, compute_reference_date
    from src.forecast import build_forecast_inputs
    from src.intake import build_intake_recommendation

    df = load_data("data/export.xlsx")
    reference_date = compute_reference_date(df)
    fi = build_forecast_inputs(df, reference_date=reference_date)

    rec = build_intake_recommendation(
        df=df,
        forecast_base=fi["forecast_base"],
        type_name="EXTERNO",
        start_date=pd.Timestamp("2024-09-01"),
        reference_date=reference_date,
    )
    print(rec.type_estimation)
    print(rec.candidates_with_history)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.reliability import (
    MIN_DURATION_DAYS,
    MIN_GROUP_SIZE,
    HISTORICAL_LIMITED_THRESHOLD,
    assign_tier,
    build_duration_benchmarks,
    compute_owner_reliability,
)
from src.forecast import build_forecast_dashboard

DISPERSION_LOW_THRESHOLD: float = 1.5
DISPERSION_HIGH_THRESHOLD: float = 2.5
ELIGIBLE_FORECAST_STATUSES: tuple[str, ...] = ("healthy", "risk")


@dataclass(frozen=True)
class TypeEstimation:
    type_name: str
    n_type: int
    type_median: float
    type_p90: float
    dispersion_ratio: float
    dispersion_level: str
    dispersion_warning: str | None


@dataclass(frozen=True)
class IntakeRecommendation:
    type_estimation: TypeEstimation
    candidates_with_history: pd.DataFrame
    candidates_without_history: pd.DataFrame
    metadata: dict


def classify_dispersion(type_median: float, type_p90: float) -> tuple[float, str, str | None]:
    """
    Returns (dispersion_ratio, dispersion_level, dispersion_warning_or_None).

    Si type_median == 0 o NaN → ratio=NaN, level="baja", warning=None.
    """
    if not pd.notna(type_median) or type_median == 0:
        return (np.nan, "baja", None)

    ratio = type_p90 / type_median

    if ratio < DISPERSION_LOW_THRESHOLD:
        return (ratio, "baja", None)
    if ratio <= DISPERSION_HIGH_THRESHOLD:
        return (ratio, "media", None)
    warning = f"Dispersión alta: 1 de cada 10 supera {type_p90:.0f} días"
    return (ratio, "alta", warning)


def build_intake_recommendation(
    df: pd.DataFrame,
    forecast_base: pd.DataFrame,
    type_name: str,
    start_date: pd.Timestamp,
    reference_date: pd.Timestamp,
    rate_window_days: int = 60,
    min_duration_days: float = MIN_DURATION_DAYS,
    min_group_size: int = MIN_GROUP_SIZE,
) -> IntakeRecommendation:
    """
    Genera la recomendación de intake para un proyecto entrante.

    Parameters
    ----------
    df : DataFrame con el dataset completo de BusinessMap (output de pipeline).
        Debe contener al menos las columnas: Owner, Type Name, is_closed,
        duration_days.
    forecast_base : DataFrame con tasas y WIP por owner (output["forecast_base"]
        de build_forecast_inputs). Necesario para recomputar el dashboard al
        horizonte específico del intake.
    type_name : Tipo de proyecto del intake (ej. "EXTERNO", "INTERNO", "PRODUCTO").
    start_date : Fecha de inicio prevista del proyecto.
    reference_date : Fecha de referencia del dataset (típicamente compute_reference_date(df)).
    rate_window_days, min_duration_days, min_group_size : pasan a las dependencias.

    Returns
    -------
    IntakeRecommendation con estimación del tipo, dos rankings de candidatos y metadata.

    Raises
    ------
    ValueError : si type_name no existe en df["Type Name"].
    """
    # 1. Validación de entrada
    available_types = sorted(df["Type Name"].dropna().unique().tolist())
    if type_name not in available_types:
        raise ValueError(
            f"Type '{type_name}' no existe en el dataset. "
            f"Tipos disponibles: {available_types}"
        )

    # 2. Horizonte de capacidad
    start_date = pd.Timestamp(start_date).floor("D")
    reference_date = pd.Timestamp(reference_date).floor("D")
    horizon_days = max(1, (start_date - reference_date).days)

    # 3. Benchmarks de duración
    bench = build_duration_benchmarks(df, min_duration_days, min_group_size)

    # 4. Estimación del tipo
    type_row = bench.type_only.loc[bench.type_only["Type Name"] == type_name]
    if type_row.empty:
        type_estimation = TypeEstimation(
            type_name=type_name,
            n_type=0,
            type_median=np.nan,
            type_p90=np.nan,
            dispersion_ratio=np.nan,
            dispersion_level="baja",
            dispersion_warning=None,
        )
    else:
        r = type_row.iloc[0]
        n_type = int(r["type_count"])
        type_median = float(r["type_median"])
        type_p90 = float(r["type_p90"])
        dispersion_ratio, dispersion_level, dispersion_warning = classify_dispersion(
            type_median, type_p90
        )
        type_estimation = TypeEstimation(
            type_name=type_name,
            n_type=n_type,
            type_median=type_median,
            type_p90=type_p90,
            dispersion_ratio=dispersion_ratio,
            dispersion_level=dispersion_level,
            dispersion_warning=dispersion_warning,
        )

    # 5. Forecast al horizonte del intake
    dashboard = build_forecast_dashboard(
        forecast_base,
        horizon_days=horizon_days,
        rate_window_days=rate_window_days,
        reference_date=reference_date,
    )
    dashboard = dashboard.rename(
        columns={"forecast_wip": "forecast_wip_at_start", "status": "forecast_status"}
    )

    # 6. Fiabilidad global por owner (para doble etiqueta en sin histórico)
    reliability = compute_owner_reliability(df, min_duration_days, min_group_size)

    # 7. Owners con histórico en este tipo
    type_owners = bench.owner_type.loc[bench.owner_type["Type Name"] == type_name].copy()
    type_owners = type_owners.rename(columns={"owner_type_count": "n_in_type"})

    type_median_val = type_estimation.type_median
    if pd.notna(type_median_val) and type_median_val != 0:
        type_owners["deviation_in_type"] = (
            type_owners["owner_type_median"] / type_median_val - 1
        )
    else:
        type_owners["deviation_in_type"] = np.nan

    type_owners["tier_in_type"] = type_owners["n_in_type"].apply(assign_tier)

    # 8. Construir candidates_with_history
    cands_with = type_owners.merge(
        dashboard[["Owner", "forecast_wip_at_start", "forecast_status"]],
        on="Owner",
        how="inner",
    )
    cands_with = cands_with.loc[
        cands_with["forecast_status"].isin(ELIGIBLE_FORECAST_STATUSES)
    ]
    cands_with = cands_with.sort_values(
        ["forecast_wip_at_start", "deviation_in_type"],
        ascending=[True, True],
    )
    cands_with = cands_with[
        ["Owner", "forecast_wip_at_start", "forecast_status", "n_in_type", "deviation_in_type", "tier_in_type"]
    ].reset_index(drop=True)

    # 9. Construir candidates_without_history
    eligible_dashboard = dashboard.loc[
        dashboard["forecast_status"].isin(ELIGIBLE_FORECAST_STATUSES)
    ]
    owners_with_history = set(cands_with["Owner"])
    cands_without = eligible_dashboard.loc[
        ~eligible_dashboard["Owner"].isin(owners_with_history)
        & (eligible_dashboard["Owner"] != "UNASSIGNED")
    ].copy()

    cands_without = cands_without.merge(
        reliability[["Owner", "n_total"]], on="Owner", how="left"
    )
    cands_without["n_total"] = cands_without["n_total"].fillna(0).astype(int)

    def _make_label(row: pd.Series) -> str:
        if row["n_total"] >= MIN_GROUP_SIZE:
            return f"Sin histórico de {type_name} ({row['n_total']} en otros tipos)"
        return "Sin histórico"

    cands_without["label"] = cands_without.apply(_make_label, axis=1)
    cands_without = cands_without.sort_values("forecast_wip_at_start", ascending=True)
    cands_without = cands_without[
        ["Owner", "forecast_wip_at_start", "forecast_status", "n_total", "label"]
    ].reset_index(drop=True)

    # 10. Metadata
    metadata = {
        "type_name": type_name,
        "start_date": start_date,
        "reference_date": reference_date,
        "horizon_days": horizon_days,
        "rate_window_days": rate_window_days,
        "min_duration_days": min_duration_days,
        "min_group_size": min_group_size,
        "n_total_dataset": bench.n_total,
        "n_effective_dataset": bench.n_effective,
        "excluded_pct_dataset": bench.excluded_pct,
    }

    # 11. Devolver
    return IntakeRecommendation(
        type_estimation=type_estimation,
        candidates_with_history=cands_with,
        candidates_without_history=cands_without,
        metadata=metadata,
    )
