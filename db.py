from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


def _load_dotenv_from_file() -> None:
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    seen = set()
    candidates = [script_dir, *script_dir.parents, cwd, *cwd.parents]
    for directory in candidates:
        if directory in seen:
            continue
        seen.add(directory)
        env_path = directory / ".env"
        if env_path.exists():
            with env_path.open("r", encoding="utf-8") as env_file:
                for line in env_file:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#") or "=" not in stripped:
                        continue
                    key, value = stripped.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("\"'")
                    if key and key not in os.environ:
                        os.environ[key] = value
            break


_load_dotenv_from_file()

_engine = None


def _get_secret(key: str) -> str | None:
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key)


def get_engine():
    global _engine
    if _engine is not None:
        return _engine

    database_url = _get_secret("DATABASE_URL")
    if not database_url:
        return None

    try:
        from urllib.parse import urlparse, unquote
        from sqlalchemy import create_engine
        from sqlalchemy.engine import URL

        p = urlparse(database_url)
        sa_url = URL.create(
            drivername="postgresql+psycopg2",
            username=p.username,
            password=unquote(p.password or ""),
            host=p.hostname,
            port=p.port,
            database=(p.path or "/postgres").lstrip("/"),
            query={"sslmode": "require"},
        )
        _engine = create_engine(sa_url, pool_pre_ping=True)
        return _engine
    except Exception as exc:
        print(f"[db] No se pudo conectar a PostgreSQL: {exc}")
        return None


def is_available() -> bool:
    engine = get_engine()
    if engine is None:
        return False
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@contextmanager
def get_connection():
    engine = get_engine()
    if engine is None:
        raise RuntimeError("PostgreSQL no está disponible.")
    with engine.connect() as conn:
        yield conn


def init_schema() -> None:
    engine = get_engine()
    if engine is None:
        return
    schema_path = Path(__file__).resolve().parent / "schema.sql"
    if not schema_path.exists():
        return
    with engine.begin() as conn:
        conn.exec_driver_sql(schema_path.read_text(encoding="utf-8"))


def upsert_file(file_info: dict) -> None:
    """Insert or update a record in uploaded_files."""
    from sqlalchemy import text
    cols = ["id", "original_name", "file_path", "file_hash", "uploaded_by", "active", "is_base"]
    record = {k: file_info.get(k) for k in cols}
    with get_connection() as conn:
        conn.execute(text("""
            INSERT INTO uploaded_files (id, original_name, file_path, file_hash, uploaded_by, active, is_base)
            VALUES (:id, :original_name, :file_path, :file_hash, :uploaded_by, :active, :is_base)
            ON CONFLICT (id) DO UPDATE SET
                original_name = EXCLUDED.original_name,
                file_path     = EXCLUDED.file_path,
                file_hash     = EXCLUDED.file_hash,
                uploaded_by   = EXCLUDED.uploaded_by,
                active        = EXCLUDED.active,
                is_base       = EXCLUDED.is_base
        """), record)
        conn.commit()


# ── Carga de DataFrames a BD ────────────────────────────────────────────────

def _sanitize_records(records: list[dict]) -> list[dict]:
    """Replace NaT and float NaN with None so psycopg2 can bind them as NULL."""
    import math
    for rec in records:
        for k, v in rec.items():
            if v is pd.NaT:
                rec[k] = None
            elif isinstance(v, float) and math.isnan(v):
                rec[k] = None
    return records


def _df_to_jsonb_extras(df: pd.DataFrame, known_columns: set) -> pd.Series:
    """Empaqueta las columnas no conocidas en un dict para JSONB."""
    import json
    extra_cols = [c for c in df.columns if c not in known_columns]
    if not extra_cols:
        return pd.Series([None] * len(df), index=df.index)

    def _row_to_json(row):
        out = {}
        for col in extra_cols:
            val = row[col]
            if pd.isna(val):
                continue
            if isinstance(val, pd.Timestamp):
                out[col] = val.isoformat()
            else:
                out[col] = val
        return json.dumps(out, default=str) if out else None

    return df.apply(_row_to_json, axis=1)


def upsert_cards_raw(bm_df: pd.DataFrame, file_source: str) -> int:
    """Inserta/actualiza tarjetas crudas en la tabla cards. PK = card_id."""
    if bm_df.empty:
        return 0
    if get_engine() is None:
        return 0

    known = {
        "Card ID", "Owner", "Type Name", "Column Name", "Board Name", "Lane Name",
        "Workflow Name", "Blocked State", "Block Count", "Block Time (hours)",
        "Cycle Time (hours)", "Total Subtasks Count", "Finished Subtasks Count",
        "Deadline", "Created At", "Start Date", "End Date",
        "Actual Start Date", "Actual End Date",
        "First Start Date", "First End Date",
        "Last Start Date", "Last End Date", "Last Blocked Date",
        "Last Modified", "Last Moved",
    }

    df = bm_df.copy()
    df["extra_fields"] = _df_to_jsonb_extras(df, known)
    df["file_source"] = file_source

    rename_map = {
        "Card ID": "card_id", "Owner": "owner", "Type Name": "type_name",
        "Column Name": "column_name", "Board Name": "board_name", "Lane Name": "lane_name",
        "Workflow Name": "workflow_name", "Blocked State": "blocked_state",
        "Block Count": "block_count", "Block Time (hours)": "block_time_hours",
        "Cycle Time (hours)": "cycle_time_hours",
        "Total Subtasks Count": "total_subtasks_count",
        "Finished Subtasks Count": "finished_subtasks_count",
        "Deadline": "deadline", "Created At": "created_at",
        "Start Date": "start_date", "End Date": "end_date",
        "Actual Start Date": "actual_start_date", "Actual End Date": "actual_end_date",
        "First Start Date": "first_start_date", "First End Date": "first_end_date",
        "Last Start Date": "last_start_date", "Last End Date": "last_end_date",
        "Last Blocked Date": "last_blocked_date",
        "Last Modified": "last_modified", "Last Moved": "last_moved",
    }
    db_cols = list(rename_map.values()) + ["extra_fields", "file_source"]
    df_db = df.rename(columns=rename_map)[db_cols]
    df_db = df_db.where(pd.notna(df_db), None)

    from sqlalchemy import text
    records = _sanitize_records(df_db.to_dict("records"))
    with get_connection() as conn:
        conn.execute(text(
            f"INSERT INTO cards ({', '.join(db_cols)}, ingested_at) "
            f"VALUES ({', '.join(':' + c for c in db_cols)}, NOW()) "
            f"ON CONFLICT (card_id) DO UPDATE SET "
            + ", ".join(f"{c} = EXCLUDED.{c}" for c in db_cols if c != "card_id")
            + ", ingested_at = NOW()"
        ), records)
        conn.commit()
    return len(records)


def insert_links_raw(links_df: pd.DataFrame, file_source: str) -> int:
    """Inserta links. Dedupe por (card_id, linked_card_id, link_type, file_source)."""
    if links_df.empty:
        return 0
    if get_engine() is None:
        return 0

    known = {"Card ID", "Linked Card ID", "Link Type"}
    df = links_df.copy()
    df["extra_fields"] = _df_to_jsonb_extras(df, known)
    df["file_source"] = file_source
    df = df.rename(columns={
        "Card ID": "card_id", "Linked Card ID": "linked_card_id", "Link Type": "link_type"
    })
    df = df[["card_id", "linked_card_id", "link_type", "extra_fields", "file_source"]]
    df = df.where(pd.notna(df), None)

    from sqlalchemy import text
    records = _sanitize_records(df.to_dict("records"))
    with get_connection() as conn:
        conn.execute(text("""
            INSERT INTO card_links (card_id, linked_card_id, link_type, extra_fields, file_source, ingested_at)
            VALUES (:card_id, :linked_card_id, :link_type, :extra_fields, :file_source, NOW())
            ON CONFLICT (card_id, linked_card_id, link_type, file_source) DO NOTHING
        """), records)
        conn.commit()
    return len(records)


def insert_subtasks_raw(subtasks_df: pd.DataFrame, file_source: str) -> int:
    """Inserta subtasks. Borra los anteriores del mismo file_source antes de insertar."""
    if subtasks_df.empty:
        return 0
    if get_engine() is None:
        return 0

    from sqlalchemy import text
    with get_connection() as conn:
        conn.execute(text("DELETE FROM card_subtasks WHERE file_source = :fs"), {"fs": file_source})
        conn.commit()

    known = {"Parent Card ID", "Subtask Owner", "Completion Date", "Age (in days)"}
    df = subtasks_df.copy()
    df["extra_fields"] = _df_to_jsonb_extras(df, known)
    df["file_source"] = file_source
    df = df.rename(columns={
        "Parent Card ID": "parent_card_id",
        "Subtask Owner": "subtask_owner",
        "Completion Date": "completion_date",
        "Age (in days)": "age_days",
    })
    cols = ["parent_card_id", "subtask_owner", "completion_date", "age_days",
            "extra_fields", "file_source"]
    df = df[cols].where(pd.notna(df[cols]), None)

    from sqlalchemy import text
    records = _sanitize_records(df.to_dict("records"))
    with get_connection() as conn:
        conn.execute(text(
            f"INSERT INTO card_subtasks ({', '.join(cols)}, ingested_at) "
            f"VALUES ({', '.join(':' + c for c in cols)}, NOW())"
        ), records)
        conn.commit()
    return len(records)


# ── Lectura de DataFrames desde BD ───────────────────────────────────────────

def load_bm_from_db() -> pd.DataFrame:
    """Devuelve el dataframe de tarjetas con las columnas originales del Excel."""
    engine = get_engine()
    if engine is None:
        return pd.DataFrame()

    df = pd.read_sql("SELECT * FROM cards", engine)
    if df.empty:
        return df

    reverse_rename = {
        "card_id": "Card ID", "owner": "Owner", "type_name": "Type Name",
        "column_name": "Column Name", "board_name": "Board Name", "lane_name": "Lane Name",
        "workflow_name": "Workflow Name", "blocked_state": "Blocked State",
        "block_count": "Block Count", "block_time_hours": "Block Time (hours)",
        "cycle_time_hours": "Cycle Time (hours)",
        "total_subtasks_count": "Total Subtasks Count",
        "finished_subtasks_count": "Finished Subtasks Count",
        "deadline": "Deadline", "created_at": "Created At",
        "start_date": "Start Date", "end_date": "End Date",
        "actual_start_date": "Actual Start Date", "actual_end_date": "Actual End Date",
        "first_start_date": "First Start Date", "first_end_date": "First End Date",
        "last_start_date": "Last Start Date", "last_end_date": "Last End Date",
        "last_blocked_date": "Last Blocked Date",
        "last_modified": "Last Modified", "last_moved": "Last Moved",
    }
    df = df.rename(columns=reverse_rename)

    import json
    if "extra_fields" in df.columns:
        def _expand(val):
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return {}
            if isinstance(val, dict):
                return val
            try:
                return json.loads(val)
            except Exception:
                return {}
        expanded = df["extra_fields"].apply(_expand).apply(pd.Series)
        df = pd.concat([df.drop(columns=["extra_fields"]), expanded], axis=1)

    df = df.drop(columns=[c for c in ("file_source", "ingested_at") if c in df.columns])
    return df


def load_links_from_db() -> pd.DataFrame:
    engine = get_engine()
    if engine is None:
        return pd.DataFrame()
    df = pd.read_sql("SELECT * FROM card_links", engine)
    if df.empty:
        return df
    df = df.rename(columns={
        "card_id": "Card ID", "linked_card_id": "Linked Card ID", "link_type": "Link Type"
    })
    import json
    if "extra_fields" in df.columns:
        def _expand(v):
            if v is None:
                return {}
            if isinstance(v, dict):
                return v
            try:
                return json.loads(v)
            except Exception:
                return {}
        expanded = df["extra_fields"].apply(_expand).apply(pd.Series)
        df = pd.concat([df.drop(columns=["extra_fields"]), expanded], axis=1)
    return df.drop(columns=[c for c in ("id", "file_source", "ingested_at") if c in df.columns])


def load_subtasks_from_db() -> pd.DataFrame:
    engine = get_engine()
    if engine is None:
        return pd.DataFrame()
    df = pd.read_sql("SELECT * FROM card_subtasks", engine)
    if df.empty:
        return df
    df = df.rename(columns={
        "parent_card_id": "Parent Card ID",
        "subtask_owner": "Subtask Owner",
        "completion_date": "Completion Date",
        "age_days": "Age (in days)",
    })
    import json
    if "extra_fields" in df.columns:
        def _expand(v):
            if v is None:
                return {}
            if isinstance(v, dict):
                return v
            try:
                return json.loads(v)
            except Exception:
                return {}
        expanded = df["extra_fields"].apply(_expand).apply(pd.Series)
        df = pd.concat([df.drop(columns=["extra_fields"]), expanded], axis=1)
    return df.drop(columns=[c for c in ("id", "file_source", "ingested_at") if c in df.columns])


def db_has_cards() -> bool:
    """True si la BD tiene al menos una fila en cards."""
    engine = get_engine()
    if engine is None:
        raise RuntimeError("No se pudo conectar a la BD. Comprueba DATABASE_URL.")
    from sqlalchemy import text
    with engine.connect() as conn:
        r = conn.execute(text("SELECT 1 FROM cards LIMIT 1")).fetchone()
    return r is not None
