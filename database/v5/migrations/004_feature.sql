BEGIN;

CREATE TABLE IF NOT EXISTS feature.snapshot (
    snapshot_id BIGSERIAL PRIMARY KEY,
    dataset_version_id BIGINT NOT NULL REFERENCES source.dataset_version(dataset_version_id),
    snapshot_name TEXT NOT NULL,
    cutoff_id TEXT,
    storage_path TEXT NOT NULL,
    sha256 CHAR(64) NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    row_count BIGINT NOT NULL CHECK (row_count >= 0),
    channel_count INTEGER CHECK (channel_count IS NULL OR channel_count > 0),
    feature_contract JSONB NOT NULL,
    target_location TEXT NOT NULL,
    generator TEXT NOT NULL,
    protocol_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (dataset_version_id, snapshot_name, sha256)
);

CREATE TABLE IF NOT EXISTS feature.snapshot_member (
    snapshot_id BIGINT NOT NULL REFERENCES feature.snapshot(snapshot_id),
    enrollment_id BIGINT NOT NULL REFERENCES education.enrollment(enrollment_id),
    row_position BIGINT NOT NULL CHECK (row_position >= 0),
    member_role TEXT NOT NULL DEFAULT 'observation' CHECK (member_role IN ('observation', 'excluded', 'future_locked')),
    target_label TEXT,
    PRIMARY KEY (snapshot_id, enrollment_id),
    UNIQUE (snapshot_id, row_position)
);

COMMIT;
