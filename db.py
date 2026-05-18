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


def get_engine():
    global _engine
    if _engine is not None:
        return _engine

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return None

    try:
        from sqlalchemy import create_engine
        _engine = create_engine(database_url, pool_pre_ping=True)
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
