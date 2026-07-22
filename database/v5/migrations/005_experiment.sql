BEGIN;

CREATE TABLE IF NOT EXISTS experiment.study (
    study_id BIGSERIAL PRIMARY KEY,
    study_name TEXT NOT NULL UNIQUE,
    dataset_id BIGINT NOT NULL REFERENCES source.dataset(dataset_id),
    research_question TEXT NOT NULL,
    primary_metric TEXT NOT NULL,
    protocol_version TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'registered' CHECK (status IN ('registered', 'running', 'completed', 'archived')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS experiment.split (
    split_id BIGSERIAL PRIMARY KEY,
    study_id BIGINT NOT NULL REFERENCES experiment.study(study_id),
    snapshot_id BIGINT NOT NULL REFERENCES feature.snapshot(snapshot_id),
    split_name TEXT NOT NULL,
    scope TEXT NOT NULL CHECK (scope IN ('inner', 'outer', 'oof', 'future_locked', 'transfer')),
    fold INTEGER CHECK (fold IS NULL OR fold >= 0),
    seed INTEGER NOT NULL,
    manifest_sha256 CHAR(64) NOT NULL CHECK (manifest_sha256 ~ '^[0-9a-f]{64}$'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (study_id, split_name, fold, seed)
);

CREATE TABLE IF NOT EXISTS experiment.split_member (
    split_id BIGINT NOT NULL REFERENCES experiment.split(split_id),
    snapshot_id BIGINT NOT NULL,
    enrollment_id BIGINT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('train', 'validation', 'test', 'future_locked', 'excluded')),
    PRIMARY KEY (split_id, enrollment_id),
    FOREIGN KEY (snapshot_id, enrollment_id) REFERENCES feature.snapshot_member(snapshot_id, enrollment_id)
);

CREATE TABLE IF NOT EXISTS experiment.model_version (
    model_version_id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_family TEXT NOT NULL CHECK (model_family IN ('logistic_regression', 'decision_tree', 'random_forest', 'svm', 'hist_gradient_boosting', 'xgboost', 'mlp', 'cnn', 'bilstm', 'cnn_bilstm')),
    version_label TEXT NOT NULL,
    config JSONB NOT NULL,
    config_sha256 CHAR(64) NOT NULL CHECK (config_sha256 ~ '^[0-9a-f]{64}$'),
    protocol_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (model_name, version_label),
    UNIQUE (model_family, config_sha256)
);

CREATE TABLE IF NOT EXISTS experiment.training_run (
    training_run_id BIGSERIAL PRIMARY KEY,
    study_id BIGINT NOT NULL REFERENCES experiment.study(study_id),
    dataset_version_id BIGINT NOT NULL REFERENCES source.dataset_version(dataset_version_id),
    snapshot_id BIGINT NOT NULL REFERENCES feature.snapshot(snapshot_id),
    split_id BIGINT NOT NULL REFERENCES experiment.split(split_id),
    model_version_id BIGINT NOT NULL REFERENCES experiment.model_version(model_version_id),
    seed INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed', 'pruned')),
    hardware JSONB NOT NULL DEFAULT '{}'::jsonb,
    checkpoint_path TEXT,
    checkpoint_sha256 CHAR(64) CHECK (checkpoint_sha256 IS NULL OR checkpoint_sha256 ~ '^[0-9a-f]{64}$'),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_summary TEXT,
    UNIQUE (study_id, split_id, model_version_id, seed),
    CHECK (status <> 'completed' OR checkpoint_sha256 IS NOT NULL),
    CHECK ((status IN ('completed', 'failed', 'pruned')) = (completed_at IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS experiment.artifact (
    artifact_id BIGSERIAL PRIMARY KEY,
    training_run_id BIGINT REFERENCES experiment.training_run(training_run_id),
    study_id BIGINT NOT NULL REFERENCES experiment.study(study_id),
    artifact_kind TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    sha256 CHAR(64) NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    byte_count BIGINT NOT NULL CHECK (byte_count >= 0),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (study_id, storage_path, sha256)
);

COMMIT;
