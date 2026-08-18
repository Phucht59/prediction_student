-- Additive Phase 11 recommendation runtime schema. Safe to re-run.
-- Does not drop existing catalog/prediction/recommendation research data.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

ALTER TABLE catalog.enrollment
    ADD COLUMN IF NOT EXISTS external_enrollment_id text;

CREATE UNIQUE INDEX IF NOT EXISTS enrollment_external_enrollment_id_uidx
    ON catalog.enrollment (external_enrollment_id)
    WHERE external_enrollment_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS enrollment_student_id_idx ON catalog.enrollment (student_id);
CREATE INDEX IF NOT EXISTS enrollment_course_id_idx ON catalog.enrollment (course_id);

INSERT INTO recommendation.action (action_key, action_name, description, is_active)
VALUES
    ('assessment_recovery', 'Assessment Recovery', 'Prioritize completing or recovering missing assessments.', true),
    ('re_engagement', 'Re-engagement', 'Encourage returning to the learning environment.', true),
    ('study_planning', 'Study Planning', 'Improve study rhythm and organize a regular plan.', true),
    ('progress_monitoring', 'Progress Monitoring', 'Review currently observed learning progress.', true),
    ('retrieval_practice', 'Retrieval Practice', 'Practice recalling knowledge through quizzes or self-tests.', true)
ON CONFLICT (action_key) DO UPDATE
SET action_name = EXCLUDED.action_name,
    description = EXCLUDED.description,
    is_active = true;

CREATE TABLE IF NOT EXISTS recommendation.bundle (
    bundle_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    bundle_version text NOT NULL UNIQUE,
    freeze_version text NOT NULL,
    checksums jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS recommendation.state_snapshot (
    snapshot_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    enrollment_id uuid NOT NULL REFERENCES catalog.enrollment(enrollment_id),
    stage text NOT NULL,
    state_version text NOT NULL,
    case_id text,
    risk_probability double precision NOT NULL CHECK (risk_probability BETWEEN 0 AND 1),
    inactive_streak double precision,
    active_days_ratio double precision,
    recent_activity double precision,
    activity_trend double precision,
    assessment_completion double precision,
    missing_assessments double precision,
    quiz_activity double precision,
    vle_available boolean,
    features jsonb NOT NULL DEFAULT '{}'::jsonb,
    source_lineage jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (enrollment_id, stage, state_version)
);

CREATE INDEX IF NOT EXISTS state_snapshot_stage_idx ON recommendation.state_snapshot (stage);
CREATE INDEX IF NOT EXISTS state_snapshot_case_id_idx ON recommendation.state_snapshot (case_id);

CREATE TABLE IF NOT EXISTS recommendation.run (
    run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    enrollment_id uuid NOT NULL REFERENCES catalog.enrollment(enrollment_id),
    snapshot_id uuid REFERENCES recommendation.state_snapshot(snapshot_id),
    bundle_id uuid NOT NULL REFERENCES recommendation.bundle(bundle_id),
    stage text NOT NULL,
    request_key text NOT NULL UNIQUE,
    plan_status text NOT NULL,
    risk_probability double precision,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS run_enrollment_idx ON recommendation.run (enrollment_id);
CREATE INDEX IF NOT EXISTS run_stage_idx ON recommendation.run (stage);
CREATE INDEX IF NOT EXISTS run_created_at_idx ON recommendation.run (created_at);

CREATE TABLE IF NOT EXISTS recommendation.score (
    score_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id uuid NOT NULL REFERENCES recommendation.run(run_id) ON DELETE CASCADE,
    action_id uuid NOT NULL REFERENCES recommendation.action(action_id),
    raw_score double precision NOT NULL,
    relevance_score double precision NOT NULL CHECK (relevance_score BETWEEN 0 AND 3),
    rank integer NOT NULL CHECK (rank > 0),
    feasibility_status text NOT NULL,
    release_status text NOT NULL,
    quality_warning text,
    model_version text NOT NULL,
    UNIQUE (run_id, action_id)
);

CREATE INDEX IF NOT EXISTS score_run_idx ON recommendation.score (run_id);
CREATE INDEX IF NOT EXISTS score_action_idx ON recommendation.score (action_id);

CREATE TABLE IF NOT EXISTS recommendation.explanation (
    explanation_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id uuid NOT NULL REFERENCES recommendation.run(run_id) ON DELETE CASCADE,
    action_id uuid NOT NULL REFERENCES recommendation.action(action_id),
    intercept double precision,
    top_positive jsonb NOT NULL DEFAULT '[]'::jsonb,
    top_negative jsonb NOT NULL DEFAULT '[]'::jsonb,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    explanation_version text NOT NULL,
    UNIQUE (run_id, action_id)
);

CREATE TABLE IF NOT EXISTS recommendation.plan (
    plan_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id uuid NOT NULL UNIQUE REFERENCES recommendation.run(run_id) ON DELETE CASCADE,
    plan_status text NOT NULL,
    top_action_keys text[] NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
