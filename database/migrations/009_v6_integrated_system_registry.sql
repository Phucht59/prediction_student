BEGIN;

SELECT pg_advisory_xact_lock(hashtext('009_v6_integrated_system_registry'));
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '10min';

-- Additive V6 evidence registry. Existing V4-V5.4 tables are not changed.
CREATE TABLE IF NOT EXISTS v6_prediction_runs (
    prediction_run_id TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    lineage JSONB NOT NULL CHECK (jsonb_typeof(lineage) = 'object'),
    checksum CHAR(64) NOT NULL CHECK (checksum ~ '^[0-9a-f]{64}$'),
    status TEXT NOT NULL,
    superseded_by TEXT REFERENCES v6_prediction_runs(prediction_run_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS v6_student_risk_profiles (
    risk_profile_id TEXT PRIMARY KEY,
    prediction_run_id TEXT NOT NULL REFERENCES v6_prediction_runs(prediction_run_id) ON DELETE RESTRICT,
    record_id TEXT NOT NULL,
    version TEXT NOT NULL,
    payload JSONB NOT NULL CHECK (jsonb_typeof(payload) = 'object'),
    lineage JSONB NOT NULL CHECK (jsonb_typeof(lineage) = 'object'),
    checksum CHAR(64) NOT NULL CHECK (checksum ~ '^[0-9a-f]{64}$'),
    status TEXT NOT NULL,
    superseded_by TEXT REFERENCES v6_student_risk_profiles(risk_profile_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (prediction_run_id, record_id, version)
);

CREATE TABLE IF NOT EXISTS v6_recommendation_plans (
    plan_id TEXT PRIMARY KEY,
    risk_profile_id TEXT NOT NULL REFERENCES v6_student_risk_profiles(risk_profile_id) ON DELETE RESTRICT,
    version TEXT NOT NULL,
    payload JSONB NOT NULL CHECK (jsonb_typeof(payload) = 'object'),
    lineage JSONB NOT NULL CHECK (jsonb_typeof(lineage) = 'object'),
    checksum CHAR(64) NOT NULL CHECK (checksum ~ '^[0-9a-f]{64}$'),
    status TEXT NOT NULL,
    superseded_by TEXT REFERENCES v6_recommendation_plans(plan_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS v6_recommendation_actions (
    recommendation_action_id BIGSERIAL PRIMARY KEY,
    plan_id TEXT NOT NULL REFERENCES v6_recommendation_plans(plan_id) ON DELETE RESTRICT,
    action_id TEXT NOT NULL,
    version TEXT NOT NULL,
    payload JSONB NOT NULL CHECK (jsonb_typeof(payload) = 'object'),
    lineage JSONB NOT NULL CHECK (jsonb_typeof(lineage) = 'object'),
    checksum CHAR(64) NOT NULL CHECK (checksum ~ '^[0-9a-f]{64}$'),
    status TEXT NOT NULL,
    superseded_by BIGINT REFERENCES v6_recommendation_actions(recommendation_action_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (plan_id, action_id, version)
);

CREATE TABLE IF NOT EXISTS v6_expert_evaluations (
    expert_evaluation_id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL REFERENCES v6_recommendation_plans(plan_id) ON DELETE RESTRICT,
    version TEXT NOT NULL,
    blinded_expert_key TEXT NOT NULL,
    payload JSONB NOT NULL CHECK (jsonb_typeof(payload) = 'object'),
    lineage JSONB NOT NULL CHECK (jsonb_typeof(lineage) = 'object'),
    checksum CHAR(64) NOT NULL CHECK (checksum ~ '^[0-9a-f]{64}$'),
    status TEXT NOT NULL,
    superseded_by TEXT REFERENCES v6_expert_evaluations(expert_evaluation_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS v6_model_registry (
    model_registry_id TEXT PRIMARY KEY, version TEXT NOT NULL, lineage JSONB NOT NULL,
    checksum CHAR(64) NOT NULL CHECK (checksum ~ '^[0-9a-f]{64}$'), status TEXT NOT NULL,
    superseded_by TEXT REFERENCES v6_model_registry(model_registry_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS v6_policy_registry (
    policy_registry_id TEXT PRIMARY KEY, version TEXT NOT NULL, lineage JSONB NOT NULL,
    checksum CHAR(64) NOT NULL CHECK (checksum ~ '^[0-9a-f]{64}$'), status TEXT NOT NULL,
    superseded_by TEXT REFERENCES v6_policy_registry(policy_registry_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS v6_artifact_registry (
    artifact_registry_id TEXT PRIMARY KEY, version TEXT NOT NULL, path TEXT NOT NULL,
    lineage JSONB NOT NULL, checksum CHAR(64) NOT NULL CHECK (checksum ~ '^[0-9a-f]{64}$'),
    status TEXT NOT NULL,
    superseded_by TEXT REFERENCES v6_artifact_registry(artifact_registry_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_v6_risk_profile_record ON v6_student_risk_profiles(record_id);
CREATE INDEX IF NOT EXISTS idx_v6_plan_profile ON v6_recommendation_plans(risk_profile_id);
CREATE INDEX IF NOT EXISTS idx_v6_action_plan ON v6_recommendation_actions(plan_id);

COMMIT;
