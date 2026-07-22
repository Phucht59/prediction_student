BEGIN;

CREATE TABLE IF NOT EXISTS recommendation.policy (
    policy_id BIGSERIAL PRIMARY KEY,
    policy_name TEXT NOT NULL,
    version_label TEXT NOT NULL,
    rules JSONB NOT NULL,
    policy_sha256 CHAR(64) NOT NULL CHECK (policy_sha256 ~ '^[0-9a-f]{64}$'),
    status TEXT NOT NULL CHECK (status IN ('draft', 'active', 'retired')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (policy_name, version_label),
    UNIQUE (policy_name, policy_sha256)
);

CREATE TABLE IF NOT EXISTS recommendation.case (
    case_id BIGSERIAL PRIMARY KEY,
    prediction_set_id BIGINT NOT NULL REFERENCES evaluation.prediction_set(prediction_set_id),
    enrollment_id BIGINT NOT NULL REFERENCES education.enrollment(enrollment_id),
    snapshot_id BIGINT NOT NULL REFERENCES feature.snapshot(snapshot_id),
    model_version_id BIGINT NOT NULL REFERENCES experiment.model_version(model_version_id),
    policy_id BIGINT NOT NULL REFERENCES recommendation.policy(policy_id),
    uncertainty_status TEXT NOT NULL CHECK (uncertainty_status IN ('confident', 'uncertain', 'abstained')),
    escalation_required BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (prediction_set_id, enrollment_id, policy_id)
);

CREATE TABLE IF NOT EXISTS recommendation.plan (
    plan_id BIGSERIAL PRIMARY KEY,
    case_id BIGINT NOT NULL REFERENCES recommendation.case(case_id),
    revision_no INTEGER NOT NULL CHECK (revision_no >= 1),
    goal TEXT NOT NULL,
    rationale TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('draft', 'approved', 'modified', 'rejected', 'completed')),
    supersedes_plan_id BIGINT REFERENCES recommendation.plan(plan_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (case_id, revision_no)
);

CREATE TABLE IF NOT EXISTS recommendation.action (
    action_id BIGSERIAL PRIMARY KEY,
    plan_id BIGINT NOT NULL REFERENCES recommendation.plan(plan_id),
    week_no INTEGER NOT NULL CHECK (week_no BETWEEN 1 AND 4),
    action_code TEXT NOT NULL,
    action_text TEXT NOT NULL,
    workload_minutes INTEGER NOT NULL CHECK (workload_minutes BETWEEN 0 AND 600),
    priority INTEGER NOT NULL DEFAULT 1 CHECK (priority BETWEEN 1 AND 5),
    UNIQUE (plan_id, week_no, action_code)
);

CREATE TABLE IF NOT EXISTS recommendation.review (
    review_id BIGSERIAL PRIMARY KEY,
    plan_id BIGINT NOT NULL REFERENCES recommendation.plan(plan_id),
    reviewer_role TEXT NOT NULL,
    reviewer_reference TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('approve', 'modify', 'reject', 'request_more_information')),
    reason TEXT NOT NULL,
    reviewed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (plan_id, reviewer_reference, reviewed_at)
);

CREATE TABLE IF NOT EXISTS recommendation.follow_up (
    follow_up_id BIGSERIAL PRIMARY KEY,
    action_id BIGINT NOT NULL REFERENCES recommendation.action(action_id),
    scheduled_for DATE NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('scheduled', 'completed', 'missed', 'cancelled')),
    observation TEXT,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMIT;
