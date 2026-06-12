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
    confidence REAL,
    original_features JSONB,
    seed INTEGER,
    run_label VARCHAR(128),
    G1 REAL,
    G2 REAL,
    G3 REAL,
    xapi_class VARCHAR(16),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS paper_evaluation_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    run_id BIGINT REFERENCES paper_runs(paper_run_id) ON DELETE SET NULL,
    dataset VARCHAR(64) NOT NULL,
    model_name VARCHAR(128),
    protocol_class VARCHAR(128),
    accuracy REAL,
    precision_macro REAL,
    recall_macro REAL,
    f1_macro REAL,
    rmse REAL,
    r2 REAL,
    precision_recall_payload JSONB,
    metric_payload JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS paper_learning_recommendations (
    recommendation_id BIGSERIAL PRIMARY KEY,
    run_id BIGINT REFERENCES paper_runs(paper_run_id) ON DELETE SET NULL,
    dataset VARCHAR(64) NOT NULL,
    model_name VARCHAR(128),
    row_index INTEGER,
    true_label INTEGER,
    predicted_label INTEGER,
    confidence REAL,
    risk_band VARCHAR(64),
    feature_snapshot JSONB,
    recommended_learning_path JSONB,
    recommendation_payload JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE paper_predictions
ADD COLUMN IF NOT EXISTS confidence REAL;

ALTER TABLE paper_predictions
ADD COLUMN IF NOT EXISTS original_features JSONB;

ALTER TABLE paper_learning_recommendations
ADD COLUMN IF NOT EXISTS row_index INTEGER;

ALTER TABLE paper_learning_recommendations
ADD COLUMN IF NOT EXISTS true_label INTEGER;

ALTER TABLE paper_learning_recommendations
ADD COLUMN IF NOT EXISTS confidence REAL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_students_dataset_row
ON students(dataset_name, source_row_index);

CREATE UNIQUE INDEX IF NOT EXISTS idx_student_grades_dataset_row
ON student_grades(dataset_name, source_row_index);
