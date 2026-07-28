# Database Metric Reconciliation Root Cause

## Incident

The first 30-model database cutover stopped at strict validation with
`all_metrics_match_canonical_json = false`. The write phase had already
committed, leaving `student_predict` in a partial 30-model state:

- models: 30
- runs: 30
- metrics: 1,000
- risk profiles: 15,378
- plans: 15,378
- actions: 27,355
- reviews: 0

The failure affected 229 value-comparison rows in
`artifacts/final/database/metric_reconciliation.csv`. The affected runs were
the six safe-revalidated UCI comparators—Logistic Regression, Decision Tree,
Random Forest, HistGradientBoosting, SVM, and XGBoost—on Student-Mat and
Student-Por. The frozen CNN-BiLSTM and MLP scientific values were not changed.

## Exact faulty code path

The cutover path was:

1. `_apply_migrations_target(dsn)`
2. `load_canonical(dsn)`
3. `validate_database(dsn, strict_public=False)`

Each of the first two functions committed independently. In
`load_canonical`, canonical metric rows were loaded using:

```sql
INSERT INTO ml.metric (...) VALUES ...
ON CONFLICT DO NOTHING
```

The database already contained the 27 pre-completion runs and their historical
metric rows. The unique index
`metric_natural_key_idx` correctly defines the metric identity as:

```text
run_id, metric_name, scope, aggregation,
class_label, budget, fold, seed
```

with PostgreSQL `NULLS NOT DISTINCT` semantics. Consequently, canonical rows
for existing runs conflicted with their historical rows and
`ON CONFLICT DO NOTHING` preserved the old values. Only genuinely missing
rows, including the three MLP runs, were inserted. Validation then compared
the database against the safe-revalidated `artifacts/final/final_results.json`
and reported 229 value mismatches.

The problem was therefore database synchronization, not model training,
prediction generation, split selection, or scientific evaluation.

## Why the database remained partial

Migrations and canonical loading committed before strict validation. A later
validation exception could not roll back those earlier commits. This explains
the observed partial state of 30 models, 30 runs, and 1,000 metrics.

## Canonical repair

The repair introduces one shared nullable natural-key normalization contract
and a set-based staging reconciliation:

1. Build the complete expected metric set exclusively from locked canonical
   release artifacts.
2. Match all eight natural-key fields, using `IS NOT DISTINCT FROM` for
   nullable fields.
3. Update stale `metric_value`, `unit`, and `detail`.
4. Insert missing canonical rows.
5. Delete only extra rows belonging to the canonical final-run scope.
6. Verify exact keys, values, details, row count, and duplicate count before
   commit.

The exact canonical metric count is 995. From the verified 27-model backup,
the first repaired reconciliation:

- matched 886 canonical keys;
- inserted 109 missing rows;
- updated 339 rows whose value, unit, or detail was stale;
- deleted 5 obsolete recommendation metric rows;
- produced 0 missing rows;
- produced 0 extra rows;
- produced 0 duplicate natural keys;
- produced 0 value mismatches;
- produced 0 detail mismatches.

The 339 update count is larger than the 229 originally reported value
mismatches because canonical JSON provenance/detail is reconciled as well.
The five removed rows were old `recommendation/final` keys (`conflicts`,
`coverage`, `duplicate_plans`, `escalation_rate`, and `plans_generated`) that
are no longer in the locked canonical source.

## Atomicity repair

Pending migrations, model/run insertion, metric reconciliation, legacy-table
disposition, and strict canonical checks now execute on one PostgreSQL
connection in one transaction. `COMMIT` occurs only after the in-transaction
validator passes. Any exception executes `ROLLBACK`.

The controlled failure hook is private to Python tests and is not exposed by
the production CLI.

PostgreSQL identity sequences are non-transactional, so a failed transaction
may advance an unused sequence value. Logical rows, schema objects, migration
ledger, model/run/metric counts, and recommendation data roll back.

## Validation evidence

The repair was tested against two databases restored independently from:

`student_predict_pre_30_model_cutover_20260728T043451Z.dump`

SHA-256:

`725d26e93493038f5f6f87812e29137287d6c43092c0617846bcefd70eee62b2`

Results:

- repaired disposable cutover: PASS;
- strict-public validation: PASS;
- second cutover replay: 0 inserted, 0 updated, 0 deleted;
- controlled validation failure: PASS;
- rollback state: 27 models, 27 runs, 891 metrics;
- recommendation counts after rollback: unchanged.

Machine-readable evidence:

- `artifacts/final/database/failed_cutover_metric_reconciliation.json`
- `artifacts/final/database/failed_partial_cutover_backup.json`
- `artifacts/final/database/disposable_cutover_reconciliation_validation.json`
- `artifacts/final/database/disposable_atomic_rollback_validation.json`

## Tests added

Automated coverage includes:

- nullable natural-key normalization;
- canonical-key uniqueness and exact expected count;
- stale metric update;
- missing metric insertion;
- extra canonical-scope metric deletion;
- idempotent replay;
- safe-revalidated UCI values;
- MLP values;
- OULAD value preservation;
- recommendation-table preservation;
- rollback on controlled validation failure;
- commit only after successful validation.
