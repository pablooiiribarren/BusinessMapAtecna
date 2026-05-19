"""
Seed inicial de la BD con los Excel existentes.

Uso:
    python seed_db.py

Carga los archivos en data/raw/ y los inserta en cards, card_links y card_subtasks.
Solo se ejecuta una vez. Si ya hay datos, hace upsert (es idempotente).
"""
import hashlib
from pathlib import Path

import db
from src.data_prep import load_businessmap_workbook


SEED_FILES = [
    "data/raw/DatosBusquedaAvanzada20250208.xlsx",
    "data/raw/DatosBusquedaAvanzada20250323.xlsx",
]


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def seed():
    db.init_schema()

    for path_str in SEED_FILES:
        path = Path(path_str)
        if not path.exists():
            print(f"[seed] saltando {path_str} (no existe)")
            continue

        file_hash = _file_hash(path)
        file_id = file_hash[:8]

        print(f"[seed] cargando {path.name}...")
        bm, links, subtasks = load_businessmap_workbook(path)

        db.upsert_file({
            "id": file_id,
            "original_name": path.name,
            "file_path": str(path.resolve()),
            "file_hash": file_hash,
            "uploaded_at": None,
            "uploaded_by": "seed_script",
            "active": True,
            "is_base": True,
        })

        n_cards = db.upsert_cards_raw(bm, file_id)
        n_links = db.insert_links_raw(links, file_id)
        n_subs  = db.insert_subtasks_raw(subtasks, file_id)
        print(f"[seed]   → cards: {n_cards}, links: {n_links}, subtasks: {n_subs}")

    print("[seed] completado.")


if __name__ == "__main__":
    seed()
