-- Phase D governed recommendation lineage.  These tables are append-only;
-- recommendation outcomes are deliberately separated from prediction snapshots.

CREATE TABLE IF NOT EXISTS recommendation_policies (
    policy_id TEXT NOT NULL, policy_version TEXT PRIMARY KEY, schema_version TEXT NOT NULL,
    status TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), approved_at TIMESTAMPTZ,
    approved_by TEXT, feature_registry_hash TEXT NOT NULL, action_catalog_hash TEXT NOT NULL,
    uncertainty_policy_hash TEXT NOT NULL, model_bundle_hash TEXT NOT NULL, source_commit TEXT NOT NULL,
    evidence_reference TEXT NOT NULL,
    CONSTRAINT chk_recommendation_policy_status CHECK (status IN ('draft','technical_validated','expert_review_pending','expert_approved','deprecated'))
);
CREATE TABLE IF NOT EXISTS recommendation_feature_registry (
    policy_version TEXT NOT NULL REFERENCES recommendation_policies(policy_version), feature_name TEXT NOT NULL,
    definition JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (policy_version, feature_name)
);
CREATE TABLE IF NOT EXISTS recommendation_action_catalog (
    policy_version TEXT NOT NULL REFERENCES recommendation_policies(policy_version), action_id TEXT NOT NULL,
    definition JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), PRIMARY KEY(policy_version, action_id)
);
CREATE TABLE IF NOT EXISTS prediction_snapshots (
    prediction_snapshot_id TEXT PRIMARY KEY, policy_version TEXT NOT NULL REFERENCES recommendation_policies(policy_version),
    model_bundle_id TEXT NOT NULL, model_candidate_id TEXT NOT NULL, model_version TEXT NOT NULL,
    student_source_reference TEXT NOT NULL, prediction_timestamp TIMESTAMPTZ NOT NULL,
    input_snapshot_timestamp TIMESTAMPTZ NOT NULL, predicted_class INTEGER NOT NULL, class_scores JSONB NOT NULL,
    ensemble_seed_predictions JSONB NOT NULL, ensemble_seed_disagreement DOUBLE PRECISION NOT NULL,
    predictive_entropy DOUBLE PRECISION NOT NULL, max_model_score DOUBLE PRECISION NOT NULL,
    r0_reference_class INTEGER NOT NULL, n0_r0_agreement BOOLEAN NOT NULL,
    feature_contract_hash TEXT NOT NULL, preprocessor_hash TEXT NOT NULL, checkpoint_bundle_hash TEXT NOT NULL,
    probability_available BOOLEAN NOT NULL DEFAULT TRUE, uncertainty_available BOOLEAN NOT NULL DEFAULT TRUE,
    deterministic_rule BOOLEAN NOT NULL DEFAULT FALSE,
    CONSTRAINT chk_prediction_snapshot_class CHECK (predicted_class BETWEEN 0 AND 2),
    CONSTRAINT chk_prediction_snapshot_scores CHECK (max_model_score BETWEEN 0 AND 1)
);
CREATE TABLE IF NOT EXISTS recommendation_instances (
    recommendation_instance_id TEXT PRIMARY KEY, prediction_snapshot_id TEXT NOT NULL REFERENCES prediction_snapshots(prediction_snapshot_id),
    policy_version TEXT NOT NULL REFERENCES recommendation_policies(policy_version), recommendation_review_status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), created_by TEXT NOT NULL,
    CONSTRAINT chk_recommendation_review_status CHECK (recommendation_review_status IN ('eligible_for_draft','advisor_review_required','insufficient_information','invalid_prediction','stale_prediction'))
);
CREATE TABLE IF NOT EXISTS recommendation_revisions (
    revision_id TEXT PRIMARY KEY, recommendation_instance_id TEXT NOT NULL REFERENCES recommendation_instances(recommendation_instance_id),
    supersedes_revision_id TEXT REFERENCES recommendation_revisions(revision_id), revision_reason TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), created_by TEXT NOT NULL, policy_version TEXT NOT NULL REFERENCES recommendation_policies(policy_version), payload JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS recommendation_goals (goal_id TEXT PRIMARY KEY, revision_id TEXT NOT NULL REFERENCES recommendation_revisions(revision_id), payload JSONB NOT NULL);
CREATE TABLE IF NOT EXISTS recommendation_actions (action_instance_id BIGSERIAL PRIMARY KEY, action_id TEXT NOT NULL, goal_id TEXT NOT NULL REFERENCES recommendation_goals(goal_id), revision_id TEXT NOT NULL REFERENCES recommendation_revisions(revision_id), payload JSONB NOT NULL);
CREATE TABLE IF NOT EXISTS advisor_decisions (advisor_decision_id TEXT PRIMARY KEY, revision_id TEXT NOT NULL REFERENCES recommendation_revisions(revision_id), decision TEXT NOT NULL, advisor_reference TEXT NOT NULL, decision_timestamp TIMESTAMPTZ NOT NULL, reason TEXT NOT NULL, modified_fields JSONB NOT NULL DEFAULT '[]'::jsonb, CONSTRAINT chk_advisor_decision CHECK (decision IN ('approve','modify','reject','request_more_information')));
CREATE TABLE IF NOT EXISTS recommendation_follow_ups (follow_up_id TEXT PRIMARY KEY, action_instance_id BIGINT REFERENCES recommendation_actions(action_instance_id), payload JSONB NOT NULL, recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
CREATE TABLE IF NOT EXISTS recommendation_outcomes (outcome_id BIGSERIAL PRIMARY KEY, recommendation_instance_id TEXT NOT NULL REFERENCES recommendation_instances(recommendation_instance_id), outcome_event JSONB NOT NULL, recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
CREATE TABLE IF NOT EXISTS expert_review_cases (case_id TEXT PRIMARY KEY, recommendation_instance_id TEXT, stratum JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
CREATE TABLE IF NOT EXISTS expert_review_ratings (rating_id BIGSERIAL PRIMARY KEY, case_id TEXT NOT NULL REFERENCES expert_review_cases(case_id), expert_reference TEXT NOT NULL, rating JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());

CREATE OR REPLACE FUNCTION reject_governed_recommendation_mutation() RETURNS TRIGGER AS $$
BEGIN RAISE EXCEPTION '% is append-only; % is not allowed', TG_TABLE_NAME, TG_OP USING ERRCODE='integrity_constraint_violation'; END; $$ LANGUAGE plpgsql;
DO $$ DECLARE t TEXT; BEGIN
  FOREACH t IN ARRAY ARRAY['recommendation_policies','recommendation_feature_registry','recommendation_action_catalog','prediction_snapshots','recommendation_instances','recommendation_revisions','recommendation_goals','recommendation_actions','advisor_decisions','recommendation_follow_ups','recommendation_outcomes','expert_review_cases','expert_review_ratings'] LOOP
    EXECUTE format('DROP TRIGGER IF EXISTS trg_%I_append_only ON %I', t, t);
    EXECUTE format('CREATE TRIGGER trg_%I_append_only BEFORE UPDATE OR DELETE ON %I FOR EACH ROW EXECUTE FUNCTION reject_governed_recommendation_mutation()', t, t);
  END LOOP;
END $$;
