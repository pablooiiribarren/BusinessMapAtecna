from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import bcrypt
import streamlit as st

from db import get_engine, is_available

_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path, override=False)


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except Exception:
        return False


def get_user(username: str) -> Optional[dict]:
    engine = get_engine()
    if engine is None:
        return None
    from sqlalchemy import text
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT * FROM users WHERE username = :u"),
            {"u": username}
        )
        row = result.mappings().fetchone()
        return dict(row) if row else None


def create_user(username: str, password: str, display_name: str) -> bool:
    engine = get_engine()
    if engine is None:
        return False
    from sqlalchemy import text
    with engine.begin() as conn:
        try:
            conn.execute(text("""
                INSERT INTO users (username, password_hash, display_name)
                VALUES (:u, :ph, :dn)
            """), {
                "u": username,
                "ph": _hash_password(password),
                "dn": display_name,
            })
            return True
        except Exception:
            return False


def _update_last_login(username: str) -> None:
    engine = get_engine()
    if engine is None:
        return
    from sqlalchemy import text
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE users SET last_login = :ts WHERE username = :u"),
            {"ts": datetime.now(timezone.utc), "u": username}
        )


def ensure_initial_admin() -> None:
    if not is_available():
        return
    engine = get_engine()
    import pandas as pd
    count = pd.read_sql("SELECT COUNT(*) as n FROM users", engine).iloc[0]["n"]
    if count > 0:
        return

    username = os.getenv("INITIAL_ADMIN_USER", "admin")
    password = os.getenv("INITIAL_ADMIN_PASSWORD", "changeme")
    name = os.getenv("INITIAL_ADMIN_NAME", "Administrador")
    created = create_user(username, password, name)
    if created:
        print(f"[auth] Usuario admin inicial creado: '{username}'")
    else:
        print("[auth] No se pudo crear el usuario admin inicial.")


def show_login_page() -> None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## 📊 BusinessMap Analytics")
        st.markdown("---")

        if not is_available():
            st.error(
                "⚠️ No hay conexión a la base de datos. "
                "Comprueba que PostgreSQL está disponible y que DATABASE_URL está configurada."
            )
            st.stop()

        with st.form("login_form"):
            st.subheader("Acceso")
            username = st.text_input("Usuario", placeholder="tu_usuario")
            password = st.text_input("Contraseña", type="password")
            submitted = st.form_submit_button("Entrar", use_container_width=True)

        if submitted:
            user = get_user(username.strip())
            if user and _verify_password(password, user["password_hash"]):
                st.session_state["authenticated"] = True
                st.session_state["current_user"] = {
                    "username": user["username"],
                    "display_name": user["display_name"],
                }
                _update_last_login(username.strip())
                st.rerun()
            else:
                st.error("Usuario o contraseña incorrectos.")


def logout() -> None:
    st.session_state.pop("authenticated", None)
    st.session_state.pop("current_user", None)
    st.rerun()


def require_auth() -> dict:
    if not st.session_state.get("authenticated"):
        show_login_page()
        st.stop()
    return st.session_state["current_user"]
