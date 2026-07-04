-- Source-record lineage and ML audit schema.
--
-- Hash policy: every hash field in this schema uses SHA-256. This includes
-- content_hash, ingestion_contract_hash, target_definition_hash,
-- split_manifest_hash, source_diff_hash, and environment_lock_hash.
--
-- Legacy paper_* tables are intentionally not modified by this migration.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'student_predict_app') THEN
        CREATE ROLE student_predict_app
            LOGIN
            NOSUPERUSER
            NOCREATEDB
            NOCREATEROLE
            NOREPLICATION;
    ELSE
        ALTER ROLE student_predict_app
            LOGIN
            NOSUPERUSER
            NOCREATEDB
            NOCREATEROLE
            NOREPLICATION;
    END IF;
END;
$$;

CREATE TABLE IF NOT EXISTS source_dataset_versions (
    dataset_version_id BIGSERIAL PRIMARY KEY,
    dataset_code VARCHAR(64) NOT NULL,
    source_locator TEXT NOT NULL,
    hash_algorithm VARCHAR(32) NOT NULL DEFAULT 'sha256',
    content_hash VARCHAR(128) NOT NULL,
    ingestion_contract JSONB NOT NULL,
    ingestion_contract_hash_algorithm VARCHAR(32) NOT NULL DEFAULT 'sha256',
    ingestion_contract_hash VARCHAR(128) NOT NULL,
    row_count INTEGER NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_source_dataset_version_identity UNIQUE (
        dataset_code,
        hash_algorithm,
        content_hash,
        ingestion_contract_hash_algorithm,
        ingestion_contract_hash
    ),
    CONSTRAINT chk_source_dataset_versions_row_count CHECK (row_count >= 0),
    CONSTRAINT chk_source_dataset_versions_hash_algorithm CHECK (hash_algorithm = 'sha256'),
    CONSTRAINT chk_source_dataset_versions_contract_hash_algorithm CHECK (ingestion_contract_hash_algorithm = 'sha256'),
    CONSTRAINT chk_source_dataset_versions_contract_object CHECK (jsonb_typeof(ingestion_contract) = 'object'),
    CONSTRAINT chk_source_dataset_versions_metadata_object CHECK (jsonb_typeof(metadata) = 'object')
);

CREATE TABLE IF NOT EXISTS source_records (
    record_id BIGSERIAL PRIMARY KEY,
    dataset_version_id BIGINT NOT NULL REFERENCES source_dataset_versions(dataset_version_id) ON DELETE RESTRICT,
    source_row_number INTEGER NOT NULL,
    raw_payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_source_records_dataset_record UNIQUE (dataset_version_id, record_id),
    CONSTRAINT uq_source_records_dataset_row UNIQUE (dataset_version_id, source_row_number),
    CONSTRAINT chk_source_records_row_number CHECK (source_row_number >= 0),
    CONSTRAINT chk_source_records_payload_object CHECK (jsonb_typeof(raw_payload) = 'object')
);

CREATE TABLE IF NOT EXISTS ml_experiment_runs (
    run_id UUID PRIMARY KEY,
    dataset_version_id BIGINT NOT NULL REFERENCES source_dataset_versions(dataset_version_id) ON DELETE RESTRICT,
    model_name VARCHAR(128) NOT NULL,
    task_type VARCHAR(32) NOT NULL,
    target_definition JSONB NOT NULL,
    target_definition_hash VARCHAR(128) NOT NULL,
    split_manifest_uri TEXT NOT NULL,
    split_manifest_hash VARCHAR(128) NOT NULL,
    git_commit TEXT NOT NULL,
    working_tree_state VARCHAR(16) NOT NULL,
    source_diff_uri TEXT,
    source_diff_hash VARCHAR(128),
    environment_lock_uri TEXT NOT NULL,
    environment_lock_hash VARCHAR(128) NOT NULL,
    train_config JSONB NOT NULL,
    artifact_uri TEXT NOT NULL,
    status VARCHAR(16) NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    CONSTRAINT uq_ml_experiment_runs_run_dataset UNIQUE (run_id, dataset_version_id),
    CONSTRAINT chk_ml_runs_task_type CHECK (task_type = 'classification'),
    CONSTRAINT chk_ml_runs_status CHECK (status IN ('running', 'completed', 'failed')),
    CONSTRAINT chk_ml_runs_working_tree CHECK (working_tree_state IN ('clean', 'dirty')),
    CONSTRAINT chk_ml_runs_clean_tree_diff CHECK (
        (working_tree_state = 'clean' AND source_diff_uri IS NULL AND source_diff_hash IS NULL)
        OR (working_tree_state = 'dirty' AND source_diff_uri IS NOT NULL AND source_diff_hash IS NOT NULL)
    ),
    CONSTRAINT chk_ml_runs_environment_lock_uri CHECK (length(trim(environment_lock_uri)) > 0),
    CONSTRAINT chk_ml_runs_environment_lock_hash CHECK (length(trim(environment_lock_hash)) > 0),
    CONSTRAINT chk_ml_runs_completed_at_status CHECK (
        (status = 'running' AND completed_at IS NULL)
        OR (status IN ('completed', 'failed') AND completed_at IS NOT NULL)
    ),
    CONSTRAINT chk_ml_runs_completed_at_order CHECK (completed_at IS NULL OR completed_at >= started_at),
    CONSTRAINT chk_ml_runs_target_definition_object CHECK (jsonb_typeof(target_definition) = 'object'),
    CONSTRAINT chk_ml_runs_train_config_object CHECK (jsonb_typeof(train_config) = 'object'),
    CONSTRAINT chk_ml_runs_metadata_object CHECK (jsonb_typeof(metadata) = 'object')
);

CREATE TABLE IF NOT EXISTS ml_run_record_splits (
    run_id UUID NOT NULL,
    dataset_version_id BIGINT NOT NULL,
    record_id BIGINT NOT NULL,
    split_name VARCHAR(16) NOT NULL,
    exclusion_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT pk_ml_run_record_splits PRIMARY KEY (run_id, record_id),
    CONSTRAINT uq_ml_run_record_splits_prediction_ref UNIQUE (run_id, record_id, split_name),
    CONSTRAINT fk_ml_run_record_splits_run_dataset FOREIGN KEY (run_id, dataset_version_id)
        REFERENCES ml_experiment_runs(run_id, dataset_version_id) ON DELETE RESTRICT,
    CONSTRAINT fk_ml_run_record_splits_source_record FOREIGN KEY (dataset_version_id, record_id)
        REFERENCES source_records(dataset_version_id, record_id) ON DELETE RESTRICT,
    CONSTRAINT chk_ml_run_record_splits_name CHECK (split_name IN ('train', 'validation', 'test', 'excluded')),
    CONSTRAINT chk_ml_run_record_splits_exclusion_reason CHECK (
        (split_name = 'excluded' AND exclusion_reason IS NOT NULL AND length(trim(exclusion_reason)) > 0)
        OR (split_name <> 'excluded' AND exclusion_reason IS NULL)
    )
);

CREATE TABLE IF NOT EXISTS ml_predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL,
    record_id BIGINT NOT NULL,
    split_name VARCHAR(16) NOT NULL,
    true_label INTEGER NOT NULL,
    predicted_label INTEGER NOT NULL,
    confidence REAL NOT NULL,
    probability JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_ml_predictions_run_record_split UNIQUE (run_id, record_id, split_name),
    CONSTRAINT fk_ml_predictions_split FOREIGN KEY (run_id, record_id, split_name)
        REFERENCES ml_run_record_splits(run_id, record_id, split_name) ON DELETE RESTRICT,
    CONSTRAINT chk_ml_predictions_split CHECK (split_name IN ('train', 'validation', 'test')),
    CONSTRAINT chk_ml_predictions_confidence CHECK (confidence BETWEEN 0 AND 1),
    CONSTRAINT chk_ml_predictions_probability_object CHECK (jsonb_typeof(probability) = 'object')
);

CREATE TABLE IF NOT EXISTS ml_run_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES ml_experiment_runs(run_id) ON DELETE RESTRICT,
    split_name VARCHAR(16) NOT NULL,
    metric_name VARCHAR(128) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    label_scope VARCHAR(64) NOT NULL DEFAULT '__all__',
    metric_context JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_ml_run_metrics_key UNIQUE (run_id, split_name, metric_name, label_scope),
    CONSTRAINT chk_ml_run_metrics_split CHECK (split_name IN ('train', 'validation', 'test')),
    CONSTRAINT chk_ml_run_metrics_name CHECK (length(trim(metric_name)) > 0),
    CONSTRAINT chk_ml_run_metrics_label_scope CHECK (length(trim(label_scope)) > 0),
    CONSTRAINT chk_ml_run_metrics_context_object CHECK (jsonb_typeof(metric_context) = 'object')
);

CREATE TABLE IF NOT EXISTS ml_recommendations (
    recommendation_id BIGSERIAL PRIMARY KEY,
    prediction_id BIGINT NOT NULL REFERENCES ml_predictions(prediction_id) ON DELETE CASCADE,
    policy_version VARCHAR(128) NOT NULL,
    risk_band VARCHAR(64) NOT NULL,
    learning_path JSONB NOT NULL,
    explanation JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_ml_recommendations_prediction_policy UNIQUE (prediction_id, policy_version),
    CONSTRAINT chk_ml_recommendations_policy CHECK (length(trim(policy_version)) > 0),
    CONSTRAINT chk_ml_recommendations_risk_band CHECK (length(trim(risk_band)) > 0),
    CONSTRAINT chk_ml_recommendations_learning_path_object CHECK (jsonb_typeof(learning_path) = 'object'),
    CONSTRAINT chk_ml_recommendations_explanation_object CHECK (jsonb_typeof(explanation) = 'object')
);

CREATE INDEX IF NOT EXISTS idx_source_dataset_versions_code
ON source_dataset_versions(dataset_code);

CREATE INDEX IF NOT EXISTS idx_source_dataset_versions_content_hash
ON source_dataset_versions(hash_algorithm, content_hash);

CREATE INDEX IF NOT EXISTS idx_source_records_dataset
ON source_records(dataset_version_id);

CREATE INDEX IF NOT EXISTS idx_ml_runs_dataset
ON ml_experiment_runs(dataset_version_id);

CREATE INDEX IF NOT EXISTS idx_ml_runs_status
ON ml_experiment_runs(status);

CREATE INDEX IF NOT EXISTS idx_ml_runs_git_commit
ON ml_experiment_runs(git_commit);

CREATE INDEX IF NOT EXISTS idx_ml_runs_environment_lock_hash
ON ml_experiment_runs(environment_lock_hash);

CREATE INDEX IF NOT EXISTS idx_ml_run_record_splits_run
ON ml_run_record_splits(run_id);

CREATE INDEX IF NOT EXISTS idx_ml_run_record_splits_record
ON ml_run_record_splits(dataset_version_id, record_id);

CREATE INDEX IF NOT EXISTS idx_ml_run_record_splits_split
ON ml_run_record_splits(split_name);

CREATE INDEX IF NOT EXISTS idx_ml_predictions_run
ON ml_predictions(run_id);

CREATE INDEX IF NOT EXISTS idx_ml_predictions_record
ON ml_predictions(record_id);

CREATE INDEX IF NOT EXISTS idx_ml_run_metrics_run
ON ml_run_metrics(run_id);

CREATE INDEX IF NOT EXISTS idx_ml_run_metrics_name
ON ml_run_metrics(metric_name);

CREATE INDEX IF NOT EXISTS idx_ml_recommendations_prediction
ON ml_recommendations(prediction_id);

CREATE INDEX IF NOT EXISTS idx_ml_recommendations_risk_band
ON ml_recommendations(risk_band);

CREATE OR REPLACE FUNCTION reject_append_only_update_delete()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION '% is append-only; % is not allowed', TG_TABLE_NAME, TG_OP
        USING ERRCODE = 'integrity_constraint_violation';
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION reject_source_record_insert_after_run()
RETURNS TRIGGER AS $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM ml_experiment_runs
        WHERE dataset_version_id = NEW.dataset_version_id
    ) THEN
        RAISE EXCEPTION 'Cannot insert source record after dataset version % has been used by a run', NEW.dataset_version_id
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION validate_ml_experiment_run_insert()
RETURNS TRIGGER AS $$
DECLARE
    source_count BIGINT;
    distinct_source_rows BIGINT;
    min_source_row INTEGER;
    max_source_row INTEGER;
    expected_rows INTEGER;
BEGIN
    IF NEW.status <> 'running' OR NEW.completed_at IS NOT NULL THEN
        RAISE EXCEPTION 'Experiment runs must be inserted as running with completed_at NULL'
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;

    SELECT row_count
    INTO expected_rows
    FROM source_dataset_versions
    WHERE dataset_version_id = NEW.dataset_version_id;

    SELECT COUNT(*), COUNT(DISTINCT source_row_number), MIN(source_row_number), MAX(source_row_number)
    INTO source_count, distinct_source_rows, min_source_row, max_source_row
    FROM source_records
    WHERE dataset_version_id = NEW.dataset_version_id;

    IF source_count <> expected_rows
       OR distinct_source_rows <> expected_rows
       OR expected_rows <= 0
       OR min_source_row <> 0
       OR max_source_row <> expected_rows - 1 THEN
        RAISE EXCEPTION 'Dataset version % source records are not fully ingested as a zero-based contiguous range', NEW.dataset_version_id
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION validate_ml_experiment_run_update()
RETURNS TRIGGER AS $$
DECLARE
    expected_rows INTEGER;
    source_count BIGINT;
    distinct_source_rows BIGINT;
    min_source_row INTEGER;
    max_source_row INTEGER;
    split_count BIGINT;
    missing_membership_count BIGINT;
    extra_membership_count BIGINT;
    test_count BIGINT;
    test_prediction_count BIGINT;
BEGIN
    IF OLD.run_id IS DISTINCT FROM NEW.run_id
       OR OLD.dataset_version_id IS DISTINCT FROM NEW.dataset_version_id
       OR OLD.model_name IS DISTINCT FROM NEW.model_name
       OR OLD.task_type IS DISTINCT FROM NEW.task_type
       OR OLD.target_definition IS DISTINCT FROM NEW.target_definition
       OR OLD.target_definition_hash IS DISTINCT FROM NEW.target_definition_hash
       OR OLD.split_manifest_uri IS DISTINCT FROM NEW.split_manifest_uri
       OR OLD.split_manifest_hash IS DISTINCT FROM NEW.split_manifest_hash
       OR OLD.git_commit IS DISTINCT FROM NEW.git_commit
       OR OLD.working_tree_state IS DISTINCT FROM NEW.working_tree_state
       OR OLD.source_diff_uri IS DISTINCT FROM NEW.source_diff_uri
       OR OLD.source_diff_hash IS DISTINCT FROM NEW.source_diff_hash
       OR OLD.environment_lock_uri IS DISTINCT FROM NEW.environment_lock_uri
       OR OLD.environment_lock_hash IS DISTINCT FROM NEW.environment_lock_hash
       OR OLD.train_config IS DISTINCT FROM NEW.train_config
       OR OLD.artifact_uri IS DISTINCT FROM NEW.artifact_uri
       OR OLD.started_at IS DISTINCT FROM NEW.started_at
       OR OLD.metadata IS DISTINCT FROM NEW.metadata THEN
        RAISE EXCEPTION 'Experiment run immutable metadata cannot be changed'
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;

    IF OLD.status <> 'running' THEN
        RAISE EXCEPTION 'Terminal experiment run % cannot transition from % to %', OLD.run_id, OLD.status, NEW.status
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;

    IF NEW.status NOT IN ('completed', 'failed') THEN
        RAISE EXCEPTION 'Only running -> completed or running -> failed transitions are allowed'
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;

    IF NEW.completed_at IS NULL OR NEW.completed_at < OLD.started_at THEN
        RAISE EXCEPTION 'Terminal transition requires completed_at >= started_at'
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;

    IF NEW.status = 'completed' THEN
        SELECT row_count
        INTO expected_rows
        FROM source_dataset_versions
        WHERE dataset_version_id = NEW.dataset_version_id;

        SELECT COUNT(*), COUNT(DISTINCT source_row_number), MIN(source_row_number), MAX(source_row_number)
        INTO source_count, distinct_source_rows, min_source_row, max_source_row
        FROM source_records
        WHERE dataset_version_id = NEW.dataset_version_id;

        IF source_count <> expected_rows
           OR distinct_source_rows <> expected_rows
           OR expected_rows <= 0
           OR min_source_row <> 0
           OR max_source_row <> expected_rows - 1 THEN
            RAISE EXCEPTION 'Dataset version % source records are not fully ingested as a zero-based contiguous range', NEW.dataset_version_id
                USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        SELECT COUNT(*)
        INTO split_count
        FROM ml_run_record_splits
        WHERE run_id = NEW.run_id;

        IF split_count <> source_count THEN
            RAISE EXCEPTION 'Run % split ledger does not cover every source record', NEW.run_id
                USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        SELECT COUNT(*)
        INTO missing_membership_count
        FROM source_records sr
        LEFT JOIN ml_run_record_splits s
            ON s.run_id = NEW.run_id
           AND s.dataset_version_id = sr.dataset_version_id
           AND s.record_id = sr.record_id
        WHERE sr.dataset_version_id = NEW.dataset_version_id
          AND s.record_id IS NULL;

        IF missing_membership_count <> 0 THEN
            RAISE EXCEPTION 'Run % has source records without split membership', NEW.run_id
                USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        SELECT COUNT(*)
        INTO extra_membership_count
        FROM ml_run_record_splits s
        LEFT JOIN source_records sr
            ON sr.dataset_version_id = s.dataset_version_id
           AND sr.record_id = s.record_id
        WHERE s.run_id = NEW.run_id
          AND sr.record_id IS NULL;

        IF extra_membership_count <> 0 THEN
            RAISE EXCEPTION 'Run % has split membership outside its source records', NEW.run_id
                USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        SELECT COUNT(*)
        INTO test_count
        FROM ml_run_record_splits
        WHERE run_id = NEW.run_id
          AND split_name = 'test';

        IF test_count = 0 THEN
            RAISE EXCEPTION 'Run % cannot complete with an empty test split', NEW.run_id
                USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        SELECT COUNT(*)
        INTO test_prediction_count
        FROM ml_predictions
        WHERE run_id = NEW.run_id
          AND split_name = 'test';

        IF test_prediction_count <> test_count THEN
            RAISE EXCEPTION 'Run % cannot complete until every test record has a prediction', NEW.run_id
                USING ERRCODE = 'integrity_constraint_violation';
        END IF;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION require_running_run_by_run_id()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM ml_experiment_runs
        WHERE run_id = NEW.run_id
          AND status = 'running'
    ) THEN
        RAISE EXCEPTION 'Run % must be running before inserting into %', NEW.run_id, TG_TABLE_NAME
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION require_running_run_for_recommendation()
RETURNS TRIGGER AS $$
DECLARE
    parent_run_id UUID;
BEGIN
    SELECT p.run_id
    INTO parent_run_id
    FROM ml_predictions p
    JOIN ml_experiment_runs r ON r.run_id = p.run_id
    WHERE p.prediction_id = NEW.prediction_id
      AND r.status = 'running';

    IF parent_run_id IS NULL THEN
        RAISE EXCEPTION 'Recommendation can only be inserted while its parent run is running'
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_source_dataset_versions_append_only ON source_dataset_versions;
CREATE TRIGGER trg_source_dataset_versions_append_only
BEFORE UPDATE OR DELETE ON source_dataset_versions
FOR EACH ROW EXECUTE FUNCTION reject_append_only_update_delete();

DROP TRIGGER IF EXISTS trg_source_records_append_only ON source_records;
CREATE TRIGGER trg_source_records_append_only
BEFORE UPDATE OR DELETE ON source_records
FOR EACH ROW EXECUTE FUNCTION reject_append_only_update_delete();

DROP TRIGGER IF EXISTS trg_source_records_sealed_dataset ON source_records;
CREATE TRIGGER trg_source_records_sealed_dataset
BEFORE INSERT ON source_records
FOR EACH ROW EXECUTE FUNCTION reject_source_record_insert_after_run();

DROP TRIGGER IF EXISTS trg_ml_runs_insert_lifecycle ON ml_experiment_runs;
CREATE TRIGGER trg_ml_runs_insert_lifecycle
BEFORE INSERT ON ml_experiment_runs
FOR EACH ROW EXECUTE FUNCTION validate_ml_experiment_run_insert();

DROP TRIGGER IF EXISTS trg_ml_runs_update_lifecycle ON ml_experiment_runs;
CREATE TRIGGER trg_ml_runs_update_lifecycle
BEFORE UPDATE ON ml_experiment_runs
FOR EACH ROW EXECUTE FUNCTION validate_ml_experiment_run_update();

DROP TRIGGER IF EXISTS trg_ml_run_record_splits_append_only ON ml_run_record_splits;
CREATE TRIGGER trg_ml_run_record_splits_append_only
BEFORE UPDATE OR DELETE ON ml_run_record_splits
FOR EACH ROW EXECUTE FUNCTION reject_append_only_update_delete();

DROP TRIGGER IF EXISTS trg_ml_run_record_splits_running_run ON ml_run_record_splits;
CREATE TRIGGER trg_ml_run_record_splits_running_run
BEFORE INSERT ON ml_run_record_splits
FOR EACH ROW EXECUTE FUNCTION require_running_run_by_run_id();

DROP TRIGGER IF EXISTS trg_ml_predictions_append_only ON ml_predictions;
CREATE TRIGGER trg_ml_predictions_append_only
BEFORE UPDATE OR DELETE ON ml_predictions
FOR EACH ROW EXECUTE FUNCTION reject_append_only_update_delete();

DROP TRIGGER IF EXISTS trg_ml_predictions_running_run ON ml_predictions;
CREATE TRIGGER trg_ml_predictions_running_run
BEFORE INSERT ON ml_predictions
FOR EACH ROW EXECUTE FUNCTION require_running_run_by_run_id();

DROP TRIGGER IF EXISTS trg_ml_run_metrics_append_only ON ml_run_metrics;
CREATE TRIGGER trg_ml_run_metrics_append_only
BEFORE UPDATE OR DELETE ON ml_run_metrics
FOR EACH ROW EXECUTE FUNCTION reject_append_only_update_delete();

DROP TRIGGER IF EXISTS trg_ml_run_metrics_running_run ON ml_run_metrics;
CREATE TRIGGER trg_ml_run_metrics_running_run
BEFORE INSERT ON ml_run_metrics
FOR EACH ROW EXECUTE FUNCTION require_running_run_by_run_id();

DROP TRIGGER IF EXISTS trg_ml_recommendations_append_only ON ml_recommendations;
CREATE TRIGGER trg_ml_recommendations_append_only
BEFORE UPDATE OR DELETE ON ml_recommendations
FOR EACH ROW EXECUTE FUNCTION reject_append_only_update_delete();

DROP TRIGGER IF EXISTS trg_ml_recommendations_running_run ON ml_recommendations;
CREATE TRIGGER trg_ml_recommendations_running_run
BEFORE INSERT ON ml_recommendations
FOR EACH ROW EXECUTE FUNCTION require_running_run_for_recommendation();

DO $$
DECLARE
    schema_name TEXT := current_schema();
    database_name TEXT := current_database();
BEGIN
    EXECUTE format('GRANT CONNECT ON DATABASE %I TO student_predict_app', database_name);
    EXECUTE format('GRANT USAGE ON SCHEMA %I TO student_predict_app', schema_name);
    EXECUTE format('GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA %I TO student_predict_app', schema_name);
END;
$$;

REVOKE ALL ON
    source_dataset_versions,
    source_records,
    ml_experiment_runs,
    ml_run_record_splits,
    ml_predictions,
    ml_run_metrics,
    ml_recommendations
FROM student_predict_app;

GRANT SELECT, INSERT ON
    source_dataset_versions,
    source_records,
    ml_run_record_splits,
    ml_predictions,
    ml_run_metrics,
    ml_recommendations
TO student_predict_app;

GRANT SELECT, INSERT ON ml_experiment_runs TO student_predict_app;
GRANT UPDATE(status, completed_at) ON ml_experiment_runs TO student_predict_app;
