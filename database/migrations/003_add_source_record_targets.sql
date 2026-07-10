-- Store targets separately from the feature payload used by DB-native loaders.
-- The table is append-only and keyed by the same dataset/version + record
-- identity as source_records.  Existing rows are backfilled from the legacy
-- raw_payload once; model code must read this table when it needs labels.

CREATE TABLE IF NOT EXISTS source_record_targets (
    dataset_version_id BIGINT NOT NULL,
    record_id BIGINT NOT NULL,
    target_name VARCHAR(64) NOT NULL,
    raw_target_value JSONB NOT NULL,
    encoded_target_value INTEGER NOT NULL,
    target_contract_hash VARCHAR(128) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT pk_source_record_targets PRIMARY KEY (dataset_version_id, record_id, target_name),
    CONSTRAINT fk_source_record_targets_source_record
        FOREIGN KEY (dataset_version_id, record_id)
        REFERENCES source_records(dataset_version_id, record_id) ON DELETE RESTRICT,
    CONSTRAINT chk_source_record_targets_raw_object CHECK (raw_target_value IS NOT NULL),
    CONSTRAINT chk_source_record_targets_encoded CHECK (encoded_target_value >= 0)
);

CREATE INDEX IF NOT EXISTS idx_source_record_targets_dataset
    ON source_record_targets(dataset_version_id, target_name);

DROP TRIGGER IF EXISTS trg_source_record_targets_append_only ON source_record_targets;
CREATE TRIGGER trg_source_record_targets_append_only
BEFORE UPDATE OR DELETE ON source_record_targets
FOR EACH ROW EXECUTE FUNCTION reject_append_only_update_delete();

GRANT SELECT, INSERT ON source_record_targets TO student_predict_app;

-- Backfill the existing student-mat dataset when the legacy payload still
-- contains G3.  This is idempotent and intentionally does not alter the
-- append-only source_records table.
INSERT INTO source_record_targets (
    dataset_version_id, record_id, target_name,
    raw_target_value, encoded_target_value, target_contract_hash
)
SELECT
    sr.dataset_version_id,
    sr.record_id,
    'G3',
    to_jsonb(sr.raw_payload -> 'G3'),
    CASE
        WHEN ((sr.raw_payload ->> 'G3')::numeric) <= 9 THEN 0
        WHEN ((sr.raw_payload ->> 'G3')::numeric) <= 14 THEN 1
        ELSE 2
    END,
    '797081e5751784633dbba3ed0e1c53d4c9a6850f338a0be5079e7360bb23689b'
FROM source_records sr
JOIN source_dataset_versions dv USING (dataset_version_id)
WHERE dv.dataset_code = 'student-mat'
  AND sr.raw_payload ? 'G3'
ON CONFLICT (dataset_version_id, record_id, target_name) DO NOTHING;
