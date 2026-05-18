-- Schema de BusinessMap Atecna Analytics
-- Ejecutar una vez contra la base de datos PostgreSQL

-- Usuarios con acceso al dashboard
CREATE TABLE IF NOT EXISTS users (
    id              SERIAL PRIMARY KEY,
    username        VARCHAR(64) UNIQUE NOT NULL,
    password_hash   VARCHAR(255) NOT NULL,
    display_name    VARCHAR(128) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    last_login      TIMESTAMPTZ
);

-- Archivos Excel subidos (reemplaza manifest.json)
CREATE TABLE IF NOT EXISTS uploaded_files (
    id              VARCHAR(16) PRIMARY KEY,   -- primeros 8 chars del hash del archivo
    original_name   VARCHAR(255) NOT NULL,
    file_path       TEXT NOT NULL,
    file_hash       VARCHAR(64) NOT NULL UNIQUE,
    uploaded_at     TIMESTAMPTZ DEFAULT NOW(),
    uploaded_by     VARCHAR(64),               -- username
    active          BOOLEAN DEFAULT TRUE,
    is_base         BOOLEAN DEFAULT FALSE
);

-- Datos parseados de las tarjetas (evita releer Excel en cada sesión)
CREATE TABLE IF NOT EXISTS cards (
    card_id             VARCHAR(64) PRIMARY KEY,
    owner               VARCHAR(255),
    type_name           VARCHAR(255),
    column_name         VARCHAR(255),
    created_at          DATE,
    actual_start        DATE,
    actual_end          DATE,
    duration_days       FLOAT,
    days_since_moved    FLOAT,
    is_blocked          BOOLEAN DEFAULT FALSE,
    block_count         INTEGER DEFAULT 0,
    subtask_count       INTEGER DEFAULT 0,
    dependency_count    INTEGER DEFAULT 0,
    file_source         VARCHAR(16),            -- FK a uploaded_files.id
    ingested_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cards_owner ON cards(owner);
CREATE INDEX IF NOT EXISTS idx_cards_type ON cards(type_name);
CREATE INDEX IF NOT EXISTS idx_cards_created ON cards(created_at);
CREATE INDEX IF NOT EXISTS idx_cards_file ON cards(file_source);
