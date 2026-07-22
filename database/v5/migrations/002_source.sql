BEGIN;

CREATE TABLE IF NOT EXISTS source.dataset (
    dataset_id BIGSERIAL PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE CHECK (slug IN ('student-mat', 'student-por', 'oulad')),
    display_name TEXT NOT NULL,
    source_uri TEXT,
    license_note TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS source.dataset_version (
    dataset_version_id BIGSERIAL PRIMARY KEY,
    dataset_id BIGINT NOT NULL REFERENCES source.dataset(dataset_id),
    version_label TEXT NOT NULL,
    source_sha256 CHAR(64) NOT NULL CHECK (source_sha256 ~ '^[0-9a-f]{64}$'),
    row_count BIGINT NOT NULL CHECK (row_count >= 0),
    data_schema JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'registered' CHECK (status IN ('registered', 'ingesting', 'sealed', 'failed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    sealed_at TIMESTAMPTZ,
    UNIQUE (dataset_id, version_label),
    UNIQUE (dataset_id, source_sha256),
    CHECK ((status = 'sealed') = (sealed_at IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS source.source_file (
    source_file_id BIGSERIAL PRIMARY KEY,
    dataset_version_id BIGINT NOT NULL REFERENCES source.dataset_version(dataset_version_id),
    logical_name TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    sha256 CHAR(64) NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    byte_count BIGINT NOT NULL CHECK (byte_count >= 0),
    row_count BIGINT CHECK (row_count IS NULL OR row_count >= 0),
    media_type TEXT NOT NULL DEFAULT 'text/csv',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (dataset_version_id, logical_name),
    UNIQUE (dataset_version_id, sha256)
);

CREATE TABLE IF NOT EXISTS source.ingestion_run (
    ingestion_run_id BIGSERIAL PRIMARY KEY,
    dataset_version_id BIGINT NOT NULL REFERENCES source.dataset_version(dataset_version_id),
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    rows_read BIGINT NOT NULL DEFAULT 0 CHECK (rows_read >= 0),
    rows_written BIGINT NOT NULL DEFAULT 0 CHECK (rows_written >= 0),
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    error_summary TEXT,
    CHECK ((status = 'running') = (completed_at IS NULL))
);

COMMIT;
