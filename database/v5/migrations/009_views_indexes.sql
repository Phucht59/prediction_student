BEGIN;

CREATE INDEX IF NOT EXISTS idx_dataset_version_dataset ON source.dataset_version(dataset_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_enrollment_student ON education.enrollment(student_id);
CREATE INDEX IF NOT EXISTS idx_enrollment_dataset_version ON education.enrollment(dataset_version_id);
CREATE INDEX IF NOT EXISTS idx_snapshot_dataset_version ON feature.snapshot(dataset_version_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_split_member_role ON experiment.split_member(split_id, role);
CREATE INDEX IF NOT EXISTS idx_training_run_study_status ON experiment.training_run(study_id, status);
CREATE INDEX IF NOT EXISTS idx_prediction_set_run ON evaluation.prediction_set(training_run_id);
CREATE INDEX IF NOT EXISTS idx_prediction_enrollment ON evaluation.prediction(enrollment_id);
CREATE INDEX IF NOT EXISTS idx_metric_run_name ON evaluation.metric(training_run_id, metric_name);
CREATE INDEX IF NOT EXISTS idx_recommendation_case_enrollment ON recommendation.case(enrollment_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_recommendation_action_week ON recommendation.action(plan_id, week_no);

CREATE OR REPLACE VIEW evaluation.model_performance AS
SELECT
    s.study_name,
    mv.model_name,
    mv.model_family,
    tr.seed,
    m.metric_name,
    m.metric_value,
    m.scope,
    m.aggregation,
    m.fold
FROM evaluation.metric AS m
JOIN experiment.training_run AS tr ON tr.training_run_id = m.training_run_id
JOIN experiment.study AS s ON s.study_id = tr.study_id
JOIN experiment.model_version AS mv ON mv.model_version_id = tr.model_version_id;

CREATE OR REPLACE VIEW recommendation.case_status AS
SELECT
    c.case_id,
    c.enrollment_id,
    c.uncertainty_status,
    c.escalation_required,
    p.plan_id,
    p.revision_no,
    p.status,
    MAX(r.reviewed_at) AS latest_reviewed_at
FROM recommendation.case AS c
LEFT JOIN recommendation.plan AS p ON p.case_id = c.case_id
LEFT JOIN recommendation.review AS r ON r.plan_id = p.plan_id
GROUP BY c.case_id, c.enrollment_id, c.uncertainty_status, c.escalation_required, p.plan_id, p.revision_no, p.status;

CREATE OR REPLACE FUNCTION source.protect_sealed_dataset_version() RETURNS TRIGGER AS $$
BEGIN
    IF OLD.status = 'sealed' THEN
        RAISE EXCEPTION 'sealed dataset versions are immutable';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_protect_sealed_dataset_version ON source.dataset_version;
CREATE TRIGGER trg_protect_sealed_dataset_version
BEFORE UPDATE OR DELETE ON source.dataset_version
FOR EACH ROW EXECUTE FUNCTION source.protect_sealed_dataset_version();

CREATE OR REPLACE FUNCTION experiment.protect_completed_training_run() RETURNS TRIGGER AS $$
BEGIN
    IF OLD.status = 'completed' THEN
        RAISE EXCEPTION 'completed training runs are immutable';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_protect_completed_training_run ON experiment.training_run;
CREATE TRIGGER trg_protect_completed_training_run
BEFORE UPDATE OR DELETE ON experiment.training_run
FOR EACH ROW EXECUTE FUNCTION experiment.protect_completed_training_run();

GRANT SELECT ON evaluation.model_performance, recommendation.case_status TO student_predict_reader, student_predict_writer;

COMMIT;
