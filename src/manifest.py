"""Gestión del manifiesto de datasets de BusinessMap.

El manifiesto (data/manifest.json) es la fuente de verdad sobre qué archivos
forman el dataset. El archivo base siempre está presente y no puede eliminarse.
Los archivos adicionales se suben, activan/desactivan y eliminan desde el dashboard.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

MANIFEST_PATH = Path("data/manifest.json")
UPLOADS_DIR   = Path("data/uploads")
BASE_FILE     = Path("data/raw/DatosBusquedaAvanzada20250208.xlsx")
BASE_ID       = "base"

# Columnas mínimas obligatorias en la hoja Businessmap
REQUIRED_BM_COLUMNS = {
    "Card ID",
    "Owner",
    "Type Name",
    "Column Name",
    "Created At",
    "Actual Start Date",
    "Actual End Date",
}


# ── hash utilities ────────────────────────────────────────────────────────────

def _hash_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()

def _hash_file(path: Path) -> str:
    return _hash_bytes(path.read_bytes()) if path.exists() else ""


# ── manifest I/O ──────────────────────────────────────────────────────────────

def load_manifest() -> dict:
    """Carga el manifiesto del disco. Lo crea si no existe."""
    if not MANIFEST_PATH.exists():
        return _create_manifest()
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        return _sync_missing(manifest)
    except (json.JSONDecodeError, KeyError):
        return _create_manifest()


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _create_manifest() -> dict:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "files": [
            {
                "id":            BASE_ID,
                "original_name": BASE_FILE.name,
                "path":          BASE_FILE.as_posix(),
                "upload_date":   datetime.now(timezone.utc).isoformat(),
                "hash":          _hash_file(BASE_FILE),
                "active":        True,
                "is_base":       True,
                "missing":       not BASE_FILE.exists(),
            }
        ],
        "last_trained_hash": None,
    }
    save_manifest(manifest)
    return manifest


def _sync_missing(manifest: dict) -> dict:
    """Marca como missing archivos que no existen en disco."""
    for f in manifest["files"]:
        f["missing"] = not Path(f["path"]).exists()
    return manifest


# ── queries ───────────────────────────────────────────────────────────────────

def active_paths(manifest: dict) -> tuple[str, ...]:
    """Rutas de archivos activos no faltantes, en orden estable (base primero)."""
    return tuple(
        f["path"]
        for f in manifest["files"]
        if f["active"] and not f.get("missing", False)
    )


def active_hash(manifest: dict) -> str:
    """Hash determinista del conjunto de archivos activos."""
    paths = active_paths(manifest)
    return hashlib.md5("|".join(paths).encode()).hexdigest()


# ── validation ────────────────────────────────────────────────────────────────

def validate_schema(file_bytes: bytes) -> list[str]:
    """Valida que el Excel tenga la estructura correcta de BusinessMap.

    Retorna lista de errores. Lista vacía significa archivo válido.
    """
    import io
    errors: list[str] = []
    try:
        with pd.ExcelFile(io.BytesIO(file_bytes)) as xls:
            missing_sheets = {"Businessmap", "Links", "Subtasks"} - set(xls.sheet_names)
            if missing_sheets:
                errors.append(
                    f"Faltan las hojas: {', '.join(sorted(missing_sheets))}. "
                    "El archivo debe provenir del export de BusinessMap."
                )
                return errors

            # Leer cabeceras con la misma lógica que clean_businessmap_sheet
            bm_raw = pd.read_excel(xls, sheet_name="Businessmap", header=None, nrows=3)
            if len(bm_raw) < 2:
                errors.append("La hoja Businessmap no tiene suficientes filas de cabecera.")
                return errors

            raw_headers = bm_raw.iloc[1].astype(str).tolist()
            headers = {" ".join(h.split()).strip() for h in raw_headers if h.strip() != "nan"}
            missing_cols = REQUIRED_BM_COLUMNS - headers
            if missing_cols:
                errors.append(
                    f"Columnas faltantes en la hoja Businessmap: "
                    f"{', '.join(sorted(missing_cols))}"
                )
    except Exception as exc:
        errors.append(f"Error al leer el archivo: {exc}")
    return errors


# ── mutations ─────────────────────────────────────────────────────────────────

def add_file(
    manifest: dict,
    original_name: str,
    file_bytes: bytes,
) -> tuple[dict, str]:
    """Añade un archivo al manifiesto y lo guarda en data/uploads/.

    Retorna (manifest_actualizado, mensaje_error).
    Si no hay error, mensaje_error es cadena vacía.
    """
    file_hash = _hash_bytes(file_bytes)

    # Comprobar duplicado por hash
    for f in manifest["files"]:
        if f["hash"] == file_hash:
            return manifest, (
                f"Este archivo ya está registrado como «{f['original_name']}». "
                "No se ha añadido de nuevo."
            )

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    file_id   = file_hash[:8]
    safe_name = "".join(c for c in original_name if c.isalnum() or c in "._- ")
    dest      = UPLOADS_DIR / f"{file_id}_{safe_name}"
    dest.write_bytes(file_bytes)

    manifest["files"].append({
        "id":            file_id,
        "original_name": original_name,
        "path":          dest.as_posix(),
        "upload_date":   datetime.now(timezone.utc).isoformat(),
        "hash":          file_hash,
        "active":        True,
        "is_base":       False,
        "missing":       False,
    })
    save_manifest(manifest)
    return manifest, ""


def toggle_active(manifest: dict, file_id: str, active: bool) -> dict:
    """Activa o desactiva un archivo (no aplica al archivo base)."""
    for f in manifest["files"]:
        if f["id"] == file_id and not f.get("is_base"):
            f["active"] = active
    save_manifest(manifest)
    return manifest


def remove_file(manifest: dict, file_id: str) -> dict:
    """Elimina un archivo adicional del manifiesto y del disco."""
    to_remove = [
        f for f in manifest["files"]
        if f["id"] == file_id and not f.get("is_base")
    ]
    for f in to_remove:
        p = Path(f["path"])
        if p.exists():
            p.unlink()
    manifest["files"] = [
        f for f in manifest["files"]
        if not (f["id"] == file_id and not f.get("is_base"))
    ]
    save_manifest(manifest)
    return manifest


def set_trained_hash(manifest: dict, h: str) -> dict:
    manifest["last_trained_hash"] = h
    save_manifest(manifest)
    return manifest
