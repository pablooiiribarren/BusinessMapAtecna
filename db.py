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
