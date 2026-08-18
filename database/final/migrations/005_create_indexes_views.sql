BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '5min';
SELECT pg_advisory_xact_lock(hashtext('final_database_v1'));

CREATE INDEX IF NOT EXISTS dataset_version_dataset_idx ON catalog.dataset_version(dataset_id);
CREATE INDEX IF NOT EXISTS record_student_key_idx ON catalog.record(student_key) WHERE student_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS model_dataset_idx ON ml.model(dataset_id, model_key);
CREATE INDEX IF NOT EXISTS run_model_status_idx ON ml.run(model_id, status);
CREATE INDEX IF NOT EXISTS run_dataset_version_idx ON ml.run(dataset_version_id);
CREATE INDEX IF NOT EXISTS artifact_run_kind_idx ON ml.artifact(run_id, artifact_kind);
CREATE INDEX IF NOT EXISTS artifact_dataset_idx ON ml.artifact(dataset_version_id) WHERE dataset_version_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS metric_natural_key_idx
    ON ml.metric(run_id, metric_name, scope, aggregation, class_label, budget, fold, seed)
    NULLS NOT DISTINCT;
CREATE INDEX IF NOT EXISTS risk_profile_record_idx ON recommendation.risk_profile(record_pk);
CREATE INDEX IF NOT EXISTS risk_profile_run_idx ON recommendation.risk_profile(run_id);
CREATE INDEX IF NOT EXISTS plan_risk_profile_idx ON recommendation.plan(risk_profile_id, revision_no);
CREATE INDEX IF NOT EXISTS action_plan_week_idx ON recommendation.action(plan_id, week_no);
CREATE INDEX IF NOT EXISTS review_plan_type_idx ON recommendation.review(plan_id, review_type);

CREATE OR REPLACE VIEW ml.final_model_results AS
SELECT d.slug AS dataset,
       m.model_key,
       m.official_name,
       m.is_selected,
       MAX(mt.metric_value) FILTER (WHERE mt.metric_name = 'accuracy' AND mt.scope = 'overall') AS accuracy,
       MAX(mt.metric_value) FILTER (WHERE mt.metric_name = 'balanced_accuracy' AND mt.scope = 'overall') AS balanced_accuracy,
       MAX(mt.metric_value) FILTER (WHERE mt.metric_name IN ('precision', 'macro_precision') AND mt.scope = 'overall') AS precision,
       MAX(mt.metric_value) FILTER (WHERE mt.metric_name IN ('recall', 'macro_recall') AND mt.scope = 'overall') AS recall,
       MAX(mt.metric_value) FILTER (WHERE mt.metric_name = 'macro_f1' AND mt.scope = 'overall') AS macro_f1,
       MAX(mt.metric_value) FILTER (WHERE mt.metric_name = 'pr_auc' AND mt.scope = 'overall') AS pr_auc,
       MAX(mt.metric_value) FILTER (WHERE mt.metric_name = 'roc_auc' AND mt.scope = 'overall') AS roc_auc,
       MAX(mt.metric_value) FILTER (WHERE mt.metric_name = 'brier' AND mt.scope = 'overall') AS brier,
       MAX(mt.metric_value) FILTER (WHERE mt.metric_name = 'nll' AND mt.scope = 'overall') AS nll,
       MAX(mt.metric_value) FILTER (WHERE mt.metric_name = 'ece' AND mt.scope = 'overall') AS ece
FROM ml.model m
JOIN catalog.dataset d ON d.dataset_id = m.dataset_id
JOIN ml.run r ON r.model_id = m.model_id AND r.status = 'completed'
LEFT JOIN ml.metric mt ON mt.run_id = r.run_id
GROUP BY d.slug, m.model_key, m.official_name, m.is_selected;

CREATE OR REPLACE VIEW recommendation.plan_summary AS
SELECT cr.source_record_id,
       rp.risk_probability,
       rp.risk_band,
       rp.uncertainty,
       rp.escalation_required,
       p.plan_id,
       p.status AS plan_status,
       COUNT(DISTINCT a.action_id) AS action_count,
       COALESCE(SUM(a.workload_minutes), 0) AS workload_minutes,
       COALESCE(MAX(rv.status), 'PENDING_EXPERT_LABELS') AS review_status
FROM recommendation.plan p
JOIN recommendation.risk_profile rp ON rp.risk_profile_id = p.risk_profile_id
JOIN catalog.record cr ON cr.record_pk = rp.record_pk
LEFT JOIN recommendation.action a ON a.plan_id = p.plan_id
LEFT JOIN recommendation.review rv ON rv.plan_id = p.plan_id
GROUP BY cr.source_record_id, rp.risk_probability, rp.risk_band, rp.uncertainty,
         rp.escalation_required, p.plan_id, p.status;
COMMIT;
