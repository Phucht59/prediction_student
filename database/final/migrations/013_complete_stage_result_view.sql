BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '5min';
SELECT pg_advisory_xact_lock(hashtext('unified_stage_aware_v1'));

CREATE OR REPLACE VIEW ml.stage_model_results AS
SELECT d.slug AS dataset,
       m.model_key,
       m.official_name,
       m.is_selected,
       mt.prediction_stage,
       MAX(mt.metric_value) FILTER (WHERE mt.metric_name = 'accuracy') AS accuracy,
       MAX(mt.metric_value) FILTER (WHERE mt.metric_name = 'balanced_accuracy') AS balanced_accuracy,
       MAX(mt.metric_value) FILTER (WHERE mt.metric_name = 'macro_f1') AS macro_f1,
       MAX(mt.metric_value) FILTER (WHERE mt.metric_name = 'pr_auc') AS pr_auc,
       MAX(mt.metric_value) FILTER (WHERE mt.metric_name = 'roc_auc') AS roc_auc,
       MAX(mt.metric_value) FILTER (WHERE mt.metric_name = 'brier') AS brier,
       MAX(mt.metric_value) FILTER (WHERE mt.metric_name = 'nll') AS nll,
       MAX(mt.metric_value) FILTER (WHERE mt.metric_name = 'ece') AS ece,
       MAX(mt.metric_value) FILTER (WHERE mt.metric_name = 'macro_precision') AS precision,
       MAX(mt.metric_value) FILTER (WHERE mt.metric_name = 'macro_recall') AS recall
FROM ml.model m
JOIN catalog.dataset d ON d.dataset_id = m.dataset_id
JOIN ml.run r ON r.model_id = m.model_id AND r.status = 'completed'
JOIN ml.metric mt ON mt.run_id = r.run_id AND mt.scope = 'stage'
GROUP BY d.slug, m.model_key, m.official_name, m.is_selected, mt.prediction_stage;
COMMIT;
