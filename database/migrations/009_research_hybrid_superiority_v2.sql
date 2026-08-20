-- Research schema for hybrid_superiority_v2. Idempotent. Does not drop serving tables.
CREATE SCHEMA IF NOT EXISTS research;
CREATE SCHEMA IF NOT EXISTS optuna_hs_v2;
CREATE SCHEMA IF NOT EXISTS recommendation;

CREATE TABLE IF NOT EXISTS research.protocol (
    protocol_id TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    git_commit TEXT,
    frozen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    payload_jsonb JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS research.data_manifest (
    dataset TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    feature_hash TEXT,
    split_hash TEXT,
    cohort_jsonb JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (dataset, source_hash)
);

CREATE TABLE IF NOT EXISTS research.study (
    study_id TEXT PRIMARY KEY,
    optuna_name TEXT,
    dataset TEXT,
    candidate TEXT,
    objective_version TEXT,
    protocol_id TEXT REFERENCES research.protocol (protocol_id)
);

CREATE TABLE IF NOT EXISTS research.run (
    run_uuid UUID PRIMARY KEY,
    study_id TEXT,
    trial_number INTEGER,
    dataset TEXT,
    model TEXT,
    outer_fold INTEGER,
    inner_fold INTEGER,
    seed INTEGER,
    git_commit TEXT,
    config_hash TEXT,
    data_hash TEXT,
    status TEXT,
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    device_jsonb JSONB,
    parameter_count INTEGER,
    peak_vram DOUBLE PRECISION,
    runtime_seconds DOUBLE PRECISION,
    outer_test_used BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS research.metric (
    run_uuid UUID NOT NULL,
    split TEXT NOT NULL,
    stage TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    value DOUBLE PRECISION,
    PRIMARY KEY (run_uuid, split, stage, metric_name)
);

CREATE TABLE IF NOT EXISTS research.prediction (
    run_uuid UUID NOT NULL,
    record_key_hash TEXT NOT NULL,
    group_key_hash TEXT NOT NULL,
    stage TEXT NOT NULL,
    y_true SMALLINT NOT NULL,
    probability DOUBLE PRECISION,
    threshold DOUBLE PRECISION,
    PRIMARY KEY (run_uuid, record_key_hash, stage)
);

CREATE TABLE IF NOT EXISTS research.artifact (
    run_uuid UUID NOT NULL,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT,
    size_bytes BIGINT,
    PRIMARY KEY (run_uuid, kind, path)
);

CREATE TABLE IF NOT EXISTS research.stat_test (
    comparison TEXT NOT NULL,
    stage TEXT NOT NULL,
    method TEXT NOT NULL,
    estimate DOUBLE PRECISION,
    ci_low DOUBLE PRECISION,
    ci_high DOUBLE PRECISION,
    p_raw DOUBLE PRECISION,
    p_adjusted DOUBLE PRECISION,
    payload_jsonb JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS research.event_log (
    id BIGSERIAL PRIMARY KEY,
    run_uuid UUID,
    level TEXT,
    event_type TEXT,
    payload_jsonb JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS run_study_idx ON research.run (study_id, dataset, model);
CREATE INDEX IF NOT EXISTS metric_run_idx ON research.metric (run_uuid, stage);
CREATE INDEX IF NOT EXISTS prediction_run_idx ON research.prediction (run_uuid, stage);

CREATE TABLE IF NOT EXISTS recommendation.llm_quota_ledger (
    model_id TEXT NOT NULL,
    quota_day DATE NOT NULL,
    successful INTEGER NOT NULL DEFAULT 0,
    attempted INTEGER NOT NULL DEFAULT 0,
    reserved INTEGER NOT NULL DEFAULT 20,
    hard_cap INTEGER NOT NULL DEFAULT 500,
    safe_cap INTEGER NOT NULL DEFAULT 480,
    PRIMARY KEY (model_id, quota_day)
);

CREATE TABLE IF NOT EXISTS recommendation.llm_request (
    idempotency_key TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    case_hash TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    status TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS recommendation.llm_response (
    idempotency_key TEXT PRIMARY KEY REFERENCES recommendation.llm_request (idempotency_key),
    valid BOOLEAN,
    latency_ms INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    payload_jsonb JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS recommendation.weak_label (
    case_hash TEXT NOT NULL,
    source TEXT NOT NULL,
    payload_jsonb JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (case_hash, source)
);

CREATE TABLE IF NOT EXISTS recommendation.label_source (
    source TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    notes TEXT
);
