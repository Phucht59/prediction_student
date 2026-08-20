-- Drop leftover objects in student_db that the current project does not use.
-- Keep: raw.*, catalog.student/course/enrollment, prediction.*,
--       recommendation.action / recommendation / recommendation_item.

-- V2 recommendation.final_freeze.v1 runtime dump (not V3 Five-EBM-C0).
DROP TABLE IF EXISTS recommendation.explanation CASCADE;
DROP TABLE IF EXISTS recommendation.score CASCADE;
DROP TABLE IF EXISTS recommendation.plan CASCADE;
DROP TABLE IF EXISTS recommendation.run CASCADE;
DROP TABLE IF EXISTS recommendation.state_snapshot CASCADE;
DROP TABLE IF EXISTS recommendation.bundle CASCADE;

-- V2 action catalog rows; V3 items keep ASSESSMENT_COMPLETION etc.
DELETE FROM recommendation.action a
WHERE NOT EXISTS (
    SELECT 1
    FROM recommendation.recommendation_item i
    WHERE i.action_id = a.action_id
);

-- Empty unused log schema.
DROP SCHEMA IF EXISTS audit CASCADE;

-- Derived JSONB feature copies. Source of truth is raw.* + parquet artifacts.
DROP SCHEMA IF EXISTS data CASCADE;
