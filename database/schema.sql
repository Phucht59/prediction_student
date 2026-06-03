CREATE TABLE IF NOT EXISTS students (
    student_id BIGSERIAL PRIMARY KEY,
    dataset_name VARCHAR(64) NOT NULL,
    source_row_index INTEGER,
    raw_profile JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS student_grades (
    grade_id BIGSERIAL PRIMARY KEY,
    student_id BIGINT REFERENCES students(student_id) ON DELETE SET NULL,
    dataset_name VARCHAR(64) NOT NULL,
    source_row_index INTEGER,
    G1 REAL,
    G2 REAL,
    G3 REAL,
    xapi_class VARCHAR(16),
    target_class INTEGER,
    target_class_name VARCHAR(64),
    raw_grade_payload JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS paper_runs (
    paper_run_id BIGSERIAL PRIMARY KEY,
    generated_at TIMESTAMPTZ,
    result_rows INTEGER,
    summary_rows INTEGER,
    postgres_status VARCHAR(64),
    postgres_message TEXT,
    run_payload JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS paper_predictions (
    paper_prediction_id BIGSERIAL PRIMARY KEY,
    run_id BIGINT REFERENCES paper_runs(paper_run_id) ON DELETE SET NULL,
    model_name VARCHAR(128) NOT NULL,
    dataset VARCHAR(64) NOT NULL,
    split VARCHAR(32) NOT NULL,
    row_index INTEGER,
    true_label INTEGER,
    predicted_label INTEGER,
    true_label_name VARCHAR(64),
    predicted_label_name VARCHAR(64),
    probability JSONB,
    seed INTEGER,
    run_label VARCHAR(128),
    G1 REAL,
    G2 REAL,
    G3 REAL,
    xapi_class VARCHAR(16),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
