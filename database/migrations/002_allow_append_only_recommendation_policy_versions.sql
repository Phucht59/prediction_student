-- Allow append-only materialization of new recommendation policy versions
-- after a run has completed, without relaxing other ML ledger writes.

CREATE OR REPLACE FUNCTION require_materializable_recommendation_parent_run()
RETURNS TRIGGER AS $$
DECLARE
    parent_run_status TEXT;
BEGIN
    SELECT r.status
    INTO parent_run_status
    FROM ml_predictions p
    JOIN ml_experiment_runs r ON r.run_id = p.run_id
    WHERE p.prediction_id = NEW.prediction_id;

    IF parent_run_status IS NULL THEN
        RAISE EXCEPTION 'Recommendation requires an existing parent prediction and run'
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;

    IF parent_run_status NOT IN ('running', 'completed') THEN
        RAISE EXCEPTION 'Recommendation can only be inserted while its parent run is running or completed'
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_ml_recommendations_running_run ON ml_recommendations;
CREATE TRIGGER trg_ml_recommendations_running_run
BEFORE INSERT ON ml_recommendations
FOR EACH ROW EXECUTE FUNCTION require_materializable_recommendation_parent_run();
