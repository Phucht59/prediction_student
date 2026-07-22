BEGIN;

CREATE TABLE IF NOT EXISTS evaluation.prediction_set (
    prediction_set_id BIGSERIAL PRIMARY KEY,
    training_run_id BIGINT NOT NULL REFERENCES experiment.training_run(training_run_id),
    scope TEXT NOT NULL CHECK (scope IN ('inner', 'outer', 'oof', 'validation', 'future', 'transfer')),
    aggregation TEXT NOT NULL CHECK (aggregation IN ('fold', 'seed', 'ensemble', 'pooled_oof')),
    storage_path TEXT,
    sha256 CHAR(64) CHECK (sha256 IS NULL OR sha256 ~ '^[0-9a-f]{64}$'),
    row_count BIGINT NOT NULL CHECK (row_count >= 0),
    probability_schema JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'registered' CHECK (status IN ('registered', 'validated', 'invalid')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (training_run_id, scope, aggregation)
);

CREATE TABLE IF NOT EXISTS evaluation.prediction (
    prediction_id BIGSERIAL PRIMARY KEY,
    prediction_set_id BIGINT NOT NULL REFERENCES evaluation.prediction_set(prediction_set_id),
    enrollment_id BIGINT NOT NULL REFERENCES education.enrollment(enrollment_id),
    target_label TEXT,
    predicted_label TEXT NOT NULL,
    positive_probability DOUBLE PRECISION CHECK (positive_probability IS NULL OR positive_probability BETWEEN 0 AND 1),
    probabilities JSONB NOT NULL DEFAULT '{}'::jsonb,
    uncertainty DOUBLE PRECISION CHECK (uncertainty IS NULL OR uncertainty >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (prediction_set_id, enrollment_id)
);

CREATE TABLE IF NOT EXISTS evaluation.metric (
    metric_id BIGSERIAL PRIMARY KEY,
    prediction_set_id BIGINT REFERENCES evaluation.prediction_set(prediction_set_id),
    training_run_id BIGINT NOT NULL REFERENCES experiment.training_run(training_run_id),
    metric_name TEXT NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL CHECK (
        metric_value = metric_value
        AND metric_value NOT IN ('Infinity'::float8, '-Infinity'::float8)
    ),
    scope TEXT NOT NULL CHECK (scope IN ('inner', 'outer', 'oof', 'validation', 'future', 'transfer')),
    aggregation TEXT NOT NULL CHECK (aggregation IN ('fold', 'seed', 'ensemble', 'pooled_oof')),
    fold INTEGER CHECK (fold IS NULL OR fold >= 0),
    seed INTEGER,
    class_label TEXT,
    detail JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE NULLS NOT DISTINCT (training_run_id, metric_name, scope, aggregation, fold, seed, class_label)
);

CREATE TABLE IF NOT EXISTS evaluation.claim (
    claim_id BIGSERIAL PRIMARY KEY,
    study_id BIGINT NOT NULL REFERENCES experiment.study(study_id),
    claim_text TEXT NOT NULL,
    verdict TEXT NOT NULL CHECK (verdict IN ('SUPPORTED', 'PARTIALLY_SUPPORTED', 'NOT_SUPPORTED', 'NOT_EVALUATED')),
    evidence JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (study_id, claim_text)
);

COMMIT;
