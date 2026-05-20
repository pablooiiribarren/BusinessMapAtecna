"""
Benchmarks de duración histórica y métrica de fiabilidad por owner.

Clientes:
  - forecast (Tab 1): consume `compute_owner_reliability` para mostrar desvío
    histórico ponderado por tipo en el dashboard de predicción de carga.
  - intake (Tab futura): consume `build_duration_benchmarks` para rankings de
    candidatos (con/sin histórico) por tipo de proyecto.

El filtro `min_duration_days` excluye tareas de duración no significativa
(uso administrativo retroactivo del Kanban, ~32% del histórico).

El desvío en Tab 1 se pondera por tipo para evitar comparar peras con manzanas:
un owner que solo hace tareas largas no debería verse penalizado frente a otro
que solo hace tareas cortas.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_DURATION_DAYS: float = 1.0
MIN_GROUP_SIZE: int = 5
HISTORICAL_LIMITED_THRESHOLD: int = 10


@dataclass(frozen=True)
class DurationBenchmarks:
    owner_type: pd.DataFrame
    type_only: pd.DataFrame
    global_median: float
    global_p90: float
    n_total: int
    n_effective: int
    n_excluded: int
    excluded_pct: float


def build_duration_benchmarks(
    df: pd.DataFrame,
    min_duration_days: float = MIN_DURATION_DAYS,
    min_group_size: int = MIN_GROUP_SIZE,
) -> DurationBenchmarks:
    """
    Benchmarks de duración histórica filtrando tareas con duración no significativa.

    Solo tareas cerradas con `duration_days >= min_duration_days` entran al cálculo.
    Tareas con Owner o Type Name nulos no contribuyen a sus respectivos grupos
    pero sí cuentan en `n_total` / `n_effective` globales.
    """
    _owner_type_empty = pd.DataFrame(
        columns=["Owner", "Type Name", "owner_type_count", "owner_type_median", "owner_type_p90"]
    )
    _type_only_empty = pd.DataFrame(
        columns=["Type Name", "type_count", "type_median", "type_p90"]
    )

    base = df.loc[
        df["is_closed"] & df["duration_days"].notna() & (df["duration_days"] >= 0)
    ]
    n_total = len(base)

    effective = base.loc[base["duration_days"] >= min_duration_days]
    n_effective = len(effective)
    n_excluded = n_total - n_effective
    excluded_pct = (n_excluded / n_total * 100) if n_total > 0 else 0.0

    if n_effective == 0:
        return DurationBenchmarks(
            owner_type=_owner_type_empty,
            type_only=_type_only_empty,
            global_median=float("nan"),
            global_p90=float("nan"),
            n_total=n_total,
            n_effective=0,
            n_excluded=n_excluded,
            excluded_pct=excluded_pct,
        )

    owner_type = (
        effective.groupby(["Owner", "Type Name"])["duration_days"]
        .agg(
            owner_type_count="count",
            owner_type_median="median",
            owner_type_p90=lambda s: s.quantile(0.90),
        )
        .reset_index()
    )
    owner_type = owner_type.loc[owner_type["owner_type_count"] >= min_group_size].copy()

    type_only = (
        effective.groupby("Type Name")["duration_days"]
        .agg(
            type_count="count",
            type_median="median",
            type_p90=lambda s: s.quantile(0.90),
        )
        .reset_index()
    )

    global_median = float(effective["duration_days"].median())
    global_p90 = float(effective["duration_days"].quantile(0.90))

    return DurationBenchmarks(
        owner_type=owner_type,
        type_only=type_only,
        global_median=global_median,
        global_p90=global_p90,
        n_total=n_total,
        n_effective=n_effective,
        n_excluded=n_excluded,
        excluded_pct=excluded_pct,
    )


def compute_owner_reliability(
    df: pd.DataFrame,
    min_duration_days: float = MIN_DURATION_DAYS,
    min_group_size: int = MIN_GROUP_SIZE,
) -> pd.DataFrame:
    """
    Desvío histórico ponderado por tipo, agregado a nivel owner.

    Para cada owner, considera los tipos donde tiene N >= min_group_size tras el
    filtro de duración. El desvío en cada tipo se calcula como
    `owner_median_in_type / type_median - 1` y se promedia ponderando por número
    de tareas del owner en ese tipo.

    Returns
    -------
    pd.DataFrame con columnas:
        - Owner
        - n_total: tareas cerradas del owner con duration_days >= min_duration_days
        - n_effective: suma de tareas del owner en tipos donde N >= min_group_size
        - deviation_weighted: float en [-1, +inf]. Ej: 0.20 = 20% más lento que pares.
                              NaN si n_effective == 0.
        - eligibility_tier: "reliable" | "limited" | "insufficient"
    """
    bench = build_duration_benchmarks(df, min_duration_days=min_duration_days, min_group_size=min_group_size)

    effective = df.loc[
        df["is_closed"] & df["duration_days"].notna() & (df["duration_days"] >= min_duration_days)
    ]
    n_total_by_owner = (
        effective.groupby("Owner")
        .size()
        .rename("n_total")
        .reset_index()
    )

    if bench.owner_type.empty or bench.type_only.empty:
        result = n_total_by_owner.copy()
        result["n_effective"] = 0
        result["deviation_weighted"] = float("nan")
        result["eligibility_tier"] = "insufficient"
        return result.sort_values("Owner").reset_index(drop=True)

    merged = bench.owner_type.merge(bench.type_only[["Type Name", "type_median"]], on="Type Name", how="inner")

    merged = merged.loc[merged["type_median"] != 0].copy()
    merged["deviation_type"] = merged["owner_type_median"] / merged["type_median"] - 1

    agg = (
        merged.groupby("Owner")
        .apply(
            lambda g: pd.Series({
                "deviation_weighted": (g["deviation_type"] * g["owner_type_count"]).sum() / g["owner_type_count"].sum(),
                "n_effective": int(g["owner_type_count"].sum()),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    result = n_total_by_owner.merge(agg, on="Owner", how="outer")
    result["n_effective"] = result["n_effective"].fillna(0).astype(int)

    def _tier(n: int) -> str:
        if n >= HISTORICAL_LIMITED_THRESHOLD:
            return "reliable"
        if n >= MIN_GROUP_SIZE:
            return "limited"
        return "insufficient"

    result["eligibility_tier"] = result["n_effective"].apply(_tier)
    return result.sort_values("Owner").reset_index(drop=True)
