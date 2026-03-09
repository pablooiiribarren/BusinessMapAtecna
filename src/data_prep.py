from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

DATE_FIELDS = {
    "created": "Created At",
    "actual_start": "Actual Start Date",
    "actual_end": "Actual End Date",
    "deadline": "Deadline",
    "planned_start": "Planned Start",
    "planned_end": "Planned End",
}

BUSINESSMAP_DATE_COLUMNS = [
    "Deadline",
    "Created At",
    "Actual End Date",
    "Start Date",
    "Actual Start Date",
    "Firts Requested Date",
    "Last Requested Date",
    "End Date",
    "Planned End",
    "Planned Start",
    "Last Date Moved Out Of Backlog (PROYECTOS)",
    "Last Date Moved Out Of A empezar (PROYECTOS)",
    "Last Date Moved Out Of En progreso (PROYECTOS)",
    "Last Date Moved Out Of Testeos internos (PROYECTOS)",
    "Last Date Moved Out Of Pilotos / Bateria de pruebas (PROYECTOS)",
    "Last Date Moved Out Of Pilotos / Batería de pruebas (PROYECTOS)",
    "Last Date Moved Out Of Finalizado (PROYECTOS)",
    "Last Date Moved Out Of Ready to Archive (PROYECTOS)",
    "First End Date",
    "First Start Date",
    "First Date Moved To Backlog (PROYECTOS)",
    "First Date Moved To A empezar (PROYECTOS)",
    "First Date Moved To En progreso (PROYECTOS)",
    "First Date Moved To Testeos internos (PROYECTOS)",
    "First Date Moved To Pilotos / Bateria de pruebas (PROYECTOS)",
    "First Date Moved To Pilotos / Batería de pruebas (PROYECTOS)",
    "First Date Moved To Finalizado (PROYECTOS)",
    "First Date Moved To Ready to Archive (PROYECTOS)",
    "Last Date Moved To Backlog (PROYECTOS)",
    "Last Blocked Date",
    "Last End Date",
    "Last Start Date",
    "Last Date Moved To A empezar (PROYECTOS)",
    "Last Date Moved To En progreso (PROYECTOS)",
    "Last Date Moved To Testeos internos (PROYECTOS)",
    "Last Date Moved To Pilotos / Bateria de pruebas (PROYECTOS)",
    "Last Date Moved To Pilotos / Batería de pruebas (PROYECTOS)",
    "Last Date Moved To Finalizado (PROYECTOS)",
    "Last Date Moved To Ready to Archive (PROYECTOS)",
    "Last Modified",
    "Last Moved",
]

IN_PROGRESS_COLUMNS = {
    "En progreso",
    "Testeos internos",
    "Pilotos / Bateria de pruebas",
    "Pilotos / Batería de pruebas",
}
CLOSED_COLUMNS = {"Ready to Archive", "Finalizado"}


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = (
        cleaned.columns.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    )
    return cleaned


def normalize_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    object_columns = cleaned.select_dtypes(include="object").columns
    if len(object_columns) == 0:
        return cleaned

    cleaned[object_columns] = cleaned[object_columns].apply(
        lambda series: series.map(
            lambda value: " ".join(value.split()) if isinstance(value, str) else value
        )
    )
    cleaned[object_columns] = cleaned[object_columns].replace({"": pd.NA})
    return cleaned


def _clean_standard_sheet(df: pd.DataFrame) -> pd.DataFrame:
    return normalize_object_columns(normalize_column_names(df))


def clean_businessmap_sheet(bm_raw: pd.DataFrame) -> pd.DataFrame:
    headers = bm_raw.iloc[1].astype(str).tolist()
    bm = bm_raw.iloc[2:].copy()
    bm.columns = headers
    bm.reset_index(drop=True, inplace=True)
    bm = bm.dropna(axis=1, how="all")
    bm = normalize_column_names(bm)
    bm = normalize_object_columns(bm)
    return bm


def load_businessmap_workbook(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    workbook_path = Path(path)
    bm_raw = pd.read_excel(workbook_path, sheet_name="Businessmap", header=None)
    bm = clean_businessmap_sheet(bm_raw)
    links = _clean_standard_sheet(pd.read_excel(workbook_path, sheet_name="Links"))
    subtasks = _clean_standard_sheet(pd.read_excel(workbook_path, sheet_name="Subtasks"))
    return bm, links, subtasks


def parse_datetime_columns(
    df: pd.DataFrame,
    date_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    parsed = df.copy()
    active_columns = list(date_columns or BUSINESSMAP_DATE_COLUMNS)
    active_columns = [column for column in active_columns if column in parsed.columns]

    for column in active_columns:
        parsed[column] = pd.to_datetime(parsed[column], errors="coerce")

    return parsed


def add_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["is_closed"] = enriched[DATE_FIELDS["actual_end"]].notna()
    enriched["duration_days"] = (
        enriched[DATE_FIELDS["actual_end"]] - enriched[DATE_FIELDS["actual_start"]]
    ).dt.total_seconds() / 86400
    return enriched


def compute_reference_date(
    df: pd.DataFrame,
    reference_columns: Iterable[str] | None = None,
) -> pd.Timestamp:
    candidate_columns = list(
        reference_columns
        or [
            DATE_FIELDS["created"],
            DATE_FIELDS["actual_end"],
            "Last Modified",
            "Last Moved",
        ]
    )
    candidate_columns = [column for column in candidate_columns if column in df.columns]
    if not candidate_columns:
        raise ValueError("No reference date columns available in dataframe.")
    return df[candidate_columns].max().max()


def add_age_feature(
    df: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    effective_reference_date = (
        reference_date if reference_date is not None else compute_reference_date(df)
    )
    enriched = df.copy()
    enriched["age_days"] = (
        effective_reference_date - enriched[DATE_FIELDS["created"]]
    ).dt.total_seconds() / 86400
    return enriched, effective_reference_date


def normalize_card_identifiers(
    bm: pd.DataFrame,
    links: pd.DataFrame,
    subtasks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bm_normalized = bm.copy()
    links_normalized = links.copy()
    subtasks_normalized = subtasks.copy()

    if "Card ID" in bm_normalized.columns:
        bm_normalized["Card ID"] = pd.to_numeric(
            bm_normalized["Card ID"], errors="coerce"
        ).astype("Int64")

    if "Parent Card ID" in subtasks_normalized.columns:
        subtasks_normalized["Parent Card ID"] = pd.to_numeric(
            subtasks_normalized["Parent Card ID"], errors="coerce"
        ).astype("Int64")

    for column in ("Card ID", "Linked Card ID"):
        if column in links_normalized.columns:
            links_normalized[column] = pd.to_numeric(
                links_normalized[column], errors="coerce"
            ).astype("Int64")

    return bm_normalized, links_normalized, subtasks_normalized


def add_subtask_features(bm: pd.DataFrame, subtasks: pd.DataFrame) -> pd.DataFrame:
    enriched = bm.copy()
    if "Parent Card ID" not in subtasks.columns or "Card ID" not in enriched.columns:
        enriched["subtasks_count"] = 0
        enriched["has_subtasks"] = 0
        return enriched

    subtasks_count = subtasks.groupby("Parent Card ID").size().rename("subtasks_count")
    enriched = enriched.merge(
        subtasks_count,
        left_on="Card ID",
        right_index=True,
        how="left",
    )
    enriched["subtasks_count"] = enriched["subtasks_count"].fillna(0)
    enriched["has_subtasks"] = (enriched["subtasks_count"] > 0).astype(int)
    return enriched


def add_link_features(bm: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    enriched = bm.copy()
    if "Card ID" not in enriched.columns or "Card ID" not in links.columns:
        enriched["outgoing_links_count"] = 0
        enriched["incoming_links_count"] = 0
        enriched["total_links_count"] = 0
        enriched["has_outgoing_links"] = 0
        enriched["has_incoming_links"] = 0
        enriched["has_any_links"] = 0
        return enriched

    outgoing_links = links.groupby("Card ID").size().rename("outgoing_links_count")
    incoming_links = (
        links.groupby("Linked Card ID").size().rename("incoming_links_count")
        if "Linked Card ID" in links.columns
        else pd.Series(dtype="int64", name="incoming_links_count")
    )

    enriched = enriched.merge(
        outgoing_links,
        left_on="Card ID",
        right_index=True,
        how="left",
    )
    enriched = enriched.merge(
        incoming_links,
        left_on="Card ID",
        right_index=True,
        how="left",
    )
    enriched["outgoing_links_count"] = enriched["outgoing_links_count"].fillna(0)
    enriched["incoming_links_count"] = enriched["incoming_links_count"].fillna(0)
    enriched["total_links_count"] = (
        enriched["outgoing_links_count"] + enriched["incoming_links_count"]
    )
    enriched["has_outgoing_links"] = (enriched["outgoing_links_count"] > 0).astype(int)
    enriched["has_incoming_links"] = (enriched["incoming_links_count"] > 0).astype(int)
    enriched["has_any_links"] = (enriched["total_links_count"] > 0).astype(int)
    return enriched


def classify_task_status(column_name: object) -> str:
    if pd.isna(column_name):
        return "open"

    normalized = str(column_name).strip()
    if normalized in CLOSED_COLUMNS:
        return "closed"
    if normalized in IN_PROGRESS_COLUMNS:
        return "in_progress"
    return "open"


def add_task_status(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    if "Column Name" in enriched.columns:
        enriched["task_status"] = enriched["Column Name"].apply(classify_task_status)
    else:
        enriched["task_status"] = "open"
    return enriched


def prepare_businessmap_dataset(path: str | Path) -> dict[str, object]:
    bm, links, subtasks = load_businessmap_workbook(path)
    bm = parse_datetime_columns(bm)
    bm = add_duration_features(bm)
    bm, links, subtasks = normalize_card_identifiers(bm, links, subtasks)
    bm, reference_date = add_age_feature(bm)
    bm = add_subtask_features(bm, subtasks)
    bm = add_link_features(bm, links)
    bm = add_task_status(bm)

    return {
        "bm": bm,
        "links": links,
        "subtasks": subtasks,
        "reference_date": reference_date,
    }
