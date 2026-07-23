# Database Data Dictionary

## `system`

### `schema_migration`

One immutable row per migration: filename, numeric version, SHA-256, apply
time, applying role, and JSON metadata.

## `catalog`

### `dataset`

Canonical dataset identity (`slug`), display name, task type, ordered class
labels, source/license notes, and creation time.

### `dataset_version`

Sealed input contract: dataset, version label, source SHA-256, final cohort
row count, data schema, source-file manifest, lifecycle status, and seal time.

### `record`

One final cohort record. The natural key is dataset version plus
`source_record_id`. Optional student/module/presentation fields support OULAD;
targets are normalized, while non-feature lineage remains in `attributes`.
Raw feature vectors are not stored.

## `ml`

### `model`

One row per model–dataset identity: stable model key, official name, family,
selected flag, JSON config, config checksum, protocol, and status.

### `run`

One final result run per model–dataset/scope: run type, aggregation, protocol,
split and feature hashes, Git commit, fixed-seed summary, hardware metadata,
status, and timestamps.

### `artifact`

Unified file registry: run/dataset links, kind, repository path, SHA-256,
bytes, rows, media type, and metadata.

### `metric`

Unified numeric/result table. `scope`, `aggregation`, `class_label`, `budget`,
`fold`, and `seed` distinguish overall, per-class, Top-k, calibration,
stability, multitask, bootstrap, and recommendation metrics. Structured
values such as confusion matrices live in `detail`.

## `recommendation`

### `policy`

Versioned policy identity, complete JSON rules, policy checksum, and status.

### `risk_profile`

OULAD record/run identity, risk probability and label, risk band,
uncertainty, escalation flag, unchanged source payload, lineage, and checksum.

### `plan`

Risk profile and policy links, revision/supersession, priority, mapped goal and
rationale, source status, unchanged source payload, and checksum.

### `action`

Plan link, source action code, target week, priority, workload, status, text,
unchanged source payload, and checksum.

### `review`

Optional advisor, expert, follow-up, or system validation review. Expert rows
are absent until real labels arrive; the release status remains
`PENDING_EXPERT_LABELS`.
