-- Schema de BusinessMap Atecna Analytics — v2 (BD como fuente real)

-- ── Usuarios (sin cambios) ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id              SERIAL PRIMARY KEY,
    username        VARCHAR(64) UNIQUE NOT NULL,
    password_hash   VARCHAR(255) NOT NULL,
    display_name    VARCHAR(128) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    last_login      TIMESTAMPTZ
);

-- ── Archivos subidos (sin cambios) ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS uploaded_files (
    id              VARCHAR(16) PRIMARY KEY,
    original_name   VARCHAR(255) NOT NULL,
    file_path       TEXT NOT NULL,
    file_hash       VARCHAR(64) NOT NULL UNIQUE,
    uploaded_at     TIMESTAMPTZ DEFAULT NOW(),
    uploaded_by     VARCHAR(64),
    active          BOOLEAN DEFAULT TRUE,
    is_base         BOOLEAN DEFAULT FALSE
);

-- ── Cards: datos crudos de la hoja Businessmap ──────────────────────────────
CREATE TABLE IF NOT EXISTS cards (
    card_id                 BIGINT PRIMARY KEY,
    owner                   VARCHAR(255),
    type_name               VARCHAR(255),
    column_name             VARCHAR(255),
    board_name              VARCHAR(255),
    lane_name               VARCHAR(255),
    workflow_name           VARCHAR(255),
    blocked_state           VARCHAR(64),
    block_count             INTEGER,
    block_time_hours        FLOAT,
    cycle_time_hours        FLOAT,
    total_subtasks_count    INTEGER,
    finished_subtasks_count INTEGER,
    deadline                TIMESTAMPTZ,
    created_at              TIMESTAMPTZ,
    start_date              TIMESTAMPTZ,
    end_date                TIMESTAMPTZ,
    actual_start_date       TIMESTAMPTZ,
    actual_end_date         TIMESTAMPTZ,
    first_start_date        TIMESTAMPTZ,
    first_end_date          TIMESTAMPTZ,
    last_start_date         TIMESTAMPTZ,
    last_end_date           TIMESTAMPTZ,
    last_blocked_date       TIMESTAMPTZ,
    last_modified           TIMESTAMPTZ,
    last_moved              TIMESTAMPTZ,
    extra_fields            JSONB,
    file_source             VARCHAR(16) REFERENCES uploaded_files(id),
    ingested_at             TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cards_owner       ON cards(owner);
CREATE INDEX IF NOT EXISTS idx_cards_type        ON cards(type_name);
CREATE INDEX IF NOT EXISTS idx_cards_created     ON cards(created_at);
CREATE INDEX IF NOT EXISTS idx_cards_actual_end  ON cards(actual_end_date);
CREATE INDEX IF NOT EXISTS idx_cards_file        ON cards(file_source);

-- ── Card Links: hoja Links ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS card_links (
    id              BIGSERIAL PRIMARY KEY,
    card_id         BIGINT NOT NULL,
    linked_card_id  BIGINT,
    link_type       VARCHAR(64),
    extra_fields    JSONB,
    file_source     VARCHAR(16) REFERENCES uploaded_files(id),
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(card_id, linked_card_id, link_type, file_source)
);

CREATE INDEX IF NOT EXISTS idx_links_card        ON card_links(card_id);
CREATE INDEX IF NOT EXISTS idx_links_linked      ON card_links(linked_card_id);
CREATE INDEX IF NOT EXISTS idx_links_file        ON card_links(file_source);

-- ── Card Subtasks: hoja Subtasks ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS card_subtasks (
    id              BIGSERIAL PRIMARY KEY,
    parent_card_id  BIGINT NOT NULL,
    subtask_owner   VARCHAR(255),
    completion_date TIMESTAMPTZ,
    age_days        INTEGER,
    extra_fields    JSONB,
    file_source     VARCHAR(16) REFERENCES uploaded_files(id),
    ingested_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_subtasks_parent   ON card_subtasks(parent_card_id);
CREATE INDEX IF NOT EXISTS idx_subtasks_file     ON card_subtasks(file_source);
