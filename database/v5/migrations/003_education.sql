BEGIN;

CREATE TABLE IF NOT EXISTS education.student (
    student_id BIGSERIAL PRIMARY KEY,
    stable_key TEXT NOT NULL UNIQUE,
    identity_kind TEXT NOT NULL CHECK (identity_kind IN ('source_row', 'oulad_id', 'conservative_quasi_identity')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS education.enrollment (
    enrollment_id BIGSERIAL PRIMARY KEY,
    student_id BIGINT NOT NULL REFERENCES education.student(student_id),
    dataset_version_id BIGINT NOT NULL REFERENCES source.dataset_version(dataset_version_id),
    source_record_key TEXT NOT NULL,
    source_row_number BIGINT CHECK (source_row_number IS NULL OR source_row_number >= 0),
    subject TEXT,
    code_module TEXT,
    code_presentation TEXT,
    attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (dataset_version_id, source_record_key),
    CHECK ((code_module IS NULL) = (code_presentation IS NULL))
);

CREATE TABLE IF NOT EXISTS education.grade_record (
    grade_record_id BIGSERIAL PRIMARY KEY,
    enrollment_id BIGINT NOT NULL REFERENCES education.enrollment(enrollment_id),
    grade_name TEXT NOT NULL,
    grade_value NUMERIC NOT NULL,
    available_stage TEXT NOT NULL CHECK (available_stage IN ('G1', 'G2', 'G3', 'assessment')),
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (enrollment_id, grade_name),
    CHECK (grade_value >= 0)
);

CREATE TABLE IF NOT EXISTS education.activity_summary (
    activity_summary_id BIGSERIAL PRIMARY KEY,
    enrollment_id BIGINT NOT NULL REFERENCES education.enrollment(enrollment_id),
    cutoff_id TEXT NOT NULL,
    storage_path TEXT,
    sha256 CHAR(64) CHECK (sha256 IS NULL OR sha256 ~ '^[0-9a-f]{64}$'),
    summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (enrollment_id, cutoff_id)
);

CREATE TABLE IF NOT EXISTS education.outcome (
    enrollment_id BIGINT PRIMARY KEY REFERENCES education.enrollment(enrollment_id),
    target_contract TEXT NOT NULL,
    class_label TEXT NOT NULL,
    numeric_value NUMERIC,
    source_field TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMIT;
