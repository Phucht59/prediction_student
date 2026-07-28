# Final Project Lock

## Git state

- Branch: `main`
- Local commit before project lock:
  `1e8249d60eefa408944ad6bb867bceb093c5cc6a`
- Origin main before push:
  `b20747dea0b2e80242b92e737e6080a24b022a3f`
- Release commit: the commit containing this report
- Force push: not used

## Scientific freeze

| Final model | Macro-F1 |
|---|---:|
| CNN-BiLSTM MAT | 0.9014601961315334 |
| CNN-BiLSTM POR | 0.8622587167738002 |
| CNN-BiLSTM OULAD | 0.8280835945631038 |
| MLP MAT comparator | 0.8595069898734821 |
| MLP POR comparator | 0.8303986867455508 |
| MLP OULAD comparator | 0.8282857900281345 |

No CNN-BiLSTM was retrained or reselected. No threshold, label, split, seed
selection, or prediction artifact changed.

## Comparator state

- Models per dataset: 10
- Total model–dataset identities: 30
- Selected family: CNN-BiLSTM
- Selected technical identities:
  `cnn_bilstm_mat`, `cnn_bilstm_por`, `cnn_bilstm_oulad`

## Fair UCI timing study

- Scope: `FAIR_TIMING_NO_TRANSFER`
- Coverage: 10 models × 3 scenarios × 2 datasets = 60 results
- Per-class coverage: 180 Low/Medium/High rows
- Paired bootstrap: 5,000 stratified replicates per comparison
- Grade-band reference: diagnostic only, not a model identity
- Deep contract: fixed `[batch, 2, 7]` plus explicit availability mask

S0 and S1 are early-warning scenarios; S2 is late-stage. The fair benchmark
does not use the official Student-Por→Student-Mat transfer and does not replace
the frozen official models. Full results are in
`reports/final/UCI_TIMING_SCENARIO_REPORT.md`.

| Dataset/scenario | Best model Macro-F1 | CNN-BiLSTM Macro-F1 (rank) | Best Low Recall |
|---|---|---|---|
| MAT S0 | Random Forest 0.4595 | 0.4157 (4) | BiLSTM-only 0.5385 |
| MAT S1 | SVM 0.7516 | 0.7176 (7) | CNN-only 0.9231 |
| MAT S2 | Decision Tree 0.9024 | 0.8829 (4) | BiLSTM-only 0.9385 |
| POR S0 | Random Forest 0.5180 | 0.4571 (5) | BiLSTM-only/CNN-BiLSTM 0.6400 |
| POR S1 | Decision Tree 0.8101 | 0.7539 (5) | CNN-only 0.8800 |
| POR S2 | XGBoost 0.8677 | 0.8446 (6) | BiLSTM-only/CNN-BiLSTM 0.9100 |

## OULAD temporal input audit

The audit confirms 47 channels per valid weekly timestep: 16 observed weekly
channels plus 31 deterministic current/past-only dynamics. Padding uses a
separate mask; static and compact aggregate branches are not repeated in the
sequence; post-cutoff/future/sensitive inputs are absent. OULAD was not
retrained.

## Backup safety

### Verified 27-model recovery backup

- File:
  `student_predict_pre_30_model_cutover_20260728T043451Z.dump`
- SHA-256:
  `725d26e93493038f5f6f87812e29137287d6c43092c0617846bcefd70eee62b2`
- Size: 9,450,715 bytes
- Restore test: PASS
- Restored state: 27 models, 27 runs, 891 metrics

### Verified 30-model backup

- File:
  `student_predict_validated_30_model_20260728T050437Z.dump`
- SHA-256:
  `9e6f6dd011edc9f93a894eddf8b9e224982310f709909c778e606244ac422060`
- Size: 9,466,681 bytes
- Archive readable: PASS
- Independent restore test: PASS
- Restored state: 30 models, 30 runs, 995 metrics
- Schema, ledger, views, constraints, recommendation integrity, and sequences:
  PASS

### Forensic backup

- File:
  `student_predict_failed_partial_cutover_20260728T044215Z.dump`
- SHA-256:
  `3b56938e9cefcf0c887b4058712f0ecfec40899c365ec4ab446776b531637758`
- Status: `FAILED_PARTIAL_STATE`
- Cutover authorization: none

Dump binaries remain outside Git under the repository backup policy. Their
manifests and validation evidence are public.

## Failed restore incident

The first in-place recovery attempt stopped because three migration-011 tables
were not archive members and blocked `DROP SCHEMA recommendation`. The
affected database was not repaired in place or dropped. Recovery moved to a
fresh replacement database.

Detailed evidence:

- `reports/final/DATABASE_RECOVERY_INCIDENT.md`
- `artifacts/final/database/failed_in_place_restore.json`

This was a database integration incident, not a scientific-result incident.

## Migration 011 recovery

Canonical migration:
`011_create_v6_2_expert_review_validation.sql`

- Version: 11
- SHA-256:
  `2f8d1f5d85e2fe0aff6cb55de23c04fc1d3e52102bc15e77e740e0d1b3db6b02`
- First replacement invocation: applied
- Second invocation: no-op
- Expert review cases: 0
- Expert plan reviews: 0
- Expert action reviews: 0

No expert label was fabricated.

## Atomic metric reconciliation

Root cause: the previous loader used `ON CONFLICT DO NOTHING`, preserving
historical UCI metric values for existing natural keys.

The repaired loader uses the full registered key:

`run_id, metric_name, scope, aggregation, class_label, budget, fold, seed`

Nullable fields use PostgreSQL `IS NOT DISTINCT FROM` semantics.

| Reconciliation field | Result |
|---|---:|
| Expected canonical metrics | 995 |
| Actual canonical metrics | 995 |
| Inserted | 109 |
| Updated | 339 |
| Deleted stale | 5 |
| Missing | 0 |
| Extra | 0 |
| Duplicate natural keys | 0 |
| Value mismatches | 0 |
| Detail mismatches | 0 |

Idempotent replay:

- inserted: 0
- updated: 0
- deleted: 0
- sequence state changed: NO
- deterministic replay: PASS

Migration, canonical model/run/metric loading, reconciliation, legacy
disposition, and strict canonical validation execute in one transaction.
A controlled failure test rolled back to 27 models, 27 runs, and 891 metrics.

## Replacement and controlled swap

- Validated replacement:
  `student_predict_replacement_20260728t050000`
- Canonical database after swap: `student_predict`
- Quarantined broken database:
  `student_predict_broken_restore_20260728t050600`
- Broken database dropped: NO
- Quarantine accepts connections: NO
- Controlled swap: PASS

The replacement was validated before rename. Post-swap validation used the
normal project DSN and passed.

## Database state

| Entity | Before cutover | Final |
|---|---:|---:|
| Models | 27 | 30 |
| Runs | 27 | 30 |
| Metrics | 891 | 995 |
| Risk profiles | 15,378 | 15,378 |
| Plans | 15,378 | 15,378 |
| Actions | 27,355 | 27,355 |
| Reviews | 0 | 0 |

`ml.final_model_results` contains 10 rows per dataset, including MLP and the
safe-revalidated UCI baselines. Selected rows remain CNN-BiLSTM.

`recommendation.plan_summary` contains 15,378 plans with unchanged semantics.
ABSTAINED plans have no actions.

## Locked status

- Future OULAD: `LOCKED_NOT_EXECUTED`
- Recommendation expert evaluation: `PENDING_EXPERT_LABELS`
- xAPI: `ABSENT`
- Fake expert review rows: 0

## Validation

- Full project suite: 76 passed
- Explicit database suite: 23 passed, 23 environment-gated tests skipped
- Fair early-warning test module: 38 passed
- Ruff: PASS
- Final status: READY
- Final report: PASS
- Final validator: PASS
- Teacher-feedback validator: PASS
- Database strict-public: PASS, 21 checks
- Database checksum replay: PASS
- Release, teacher-feedback, timing, and evidence checksum replay: PASS
- Evidence manifest: 395 files
- Official deep models retrained: NO
- Future OULAD accessed: NO
- DOCX/PDF modified: NO

## Git change classification

- `KEEP_SOURCE`: backup safety, canonical reconciliation, atomic cutover,
  deterministic sequence handling, and evidence-manifest helpers.
- `KEEP_TEST`: backup-currentness, nullable metric keys, stale/missing/extra
  reconciliation, rollback, successful commit, and idempotency tests.
- `KEEP_VALID_EVIDENCE`: backup, restore, disposable, recovery, swap,
  reconciliation, checksum, and final-state artifacts.
- `REGENERATE`: database/release/evidence checksum manifests and generated
  final reports.
- `TEMPORARY`: ignored dump binaries and removed disposable databases.

## Final status

`PROJECT_LOCKED_READY_FOR_THESIS_UPDATE`
