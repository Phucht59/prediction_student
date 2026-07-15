# Technical Project Contract

## Problem definition

Predict the three-class final-achievement target from late-stage prior grades and generate a governed, structured learning-path draft. Prediction and recommendation are separate contracts: prediction supplies evidence; the recommendation policy maps admissible evidence to non-causal actions subject to advisor review.

## Scope

- Dataset: UCI Student Performance `student-mat`, 395 total rows.
- Official development population: 316 records.
- Quarantined population: 79 `legacy_heldout_observed` records, prohibited for selection, tuning, calibration or confirmation.
- Final prediction roles are frozen: R0 overall, N0 thesis hybrid.
- Recommendation technical validation is complete; expert/effectiveness validation is not.
- No production-readiness or external-generalization claim is authorized.

## Data contracts

PostgreSQL is the lineage system of record. `source_records` stores source payload and source-row identity; target values are separated by migration 003. Development access must use the immutable source-row allowlist. Training/evaluation may access target storage only through the registered target contract. Phase D prediction snapshots require no true outcome.

Repository data commands must never fetch 79 observed records unless a future protocol explicitly changes their role; they can never become an untouched confirmation set again.

## Target contract

- Raw target: `G3` on 0–20 scale.
- Three classes: Low 0–9 (`0`), Medium 10–14 (`1`), High 15–20 (`2`).
- Contract bins: `[0, 9, 14, 20]`, `include_lowest=True`.
- Macro-F1 is the primary classification metric.
- Raw G3 is forbidden as a prediction/recommendation input.

## Feature contracts

Prediction feature contract is exactly raw `[G1, G2]`. No G3/G3-derived values, context/profile fields, row IDs, fold IDs, version IDs or prediction metadata are model inputs.

Recommendation core registry permits G1, G2 and deterministic `G2-G1`; the trajectory is explanation/planning-only and cannot change the frozen prediction model. `studytime`, `failures`, `schoolsup`, `famsup`, `activities`, `internet` and `absences` remain disabled with `timing_or_semantic_contract_unverified` until separately approved. Missing values are unknown, never default risk values.

## Model roles

| Candidate | Contract role |
| --- | --- |
| R0 — deterministic G2 thresholds | `final_overall_model`; reference and agreement guardrail; no probability/uncertainty |
| M1 — Random Forest | practical-tie stochastic ML comparator; highest point Macro-F1 |
| M2 — SVM RBF | practical-tie deterministic ML comparator |
| N0 — nominal CNN–BiLSTM | `final_thesis_hybrid_model`; five-seed score ensemble |
| N1 — ordinal CNN–BiLSTM | ordinal research comparator |

R0, M1 and M2 remain a practical tie. R0 is selected by the pre-registered tie-break and simplicity, not a statistical superiority claim. N0/N1 have no clear superiority; N0 is selected under expanded stability evidence.

## Validation rules

1. Five immutable outer folds and three inner folds on development-only records.
2. Selection uses nested development evidence; no outer-label calibration or threshold tuning.
3. Neural estimator creation, criterion, resampling and refit contracts remain canonical and replayable.
4. Phase E new seeds are 202601–202605; best-seed selection is prohibited.
5. Deterministic candidates must not receive fake seed rows or fake stability evidence.
6. Macro-F1 is primary. Accuracy, precision, recall, F1 and PR-AUC are classification metrics.
7. Continuous-G3 MAE/RMSE/R² are secondary. R0 maps to raw G2; probability models use training-partition class means. R² on encoded 0/1/2 classes is prohibited.
8. No post-hoc composite score.
9. No experiment may reopen final selection during repository closure.

## Official final metrics

| Model | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Weighted-F1 | High F1 | Macro PR-AUC | RMSE G3 | R² G3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R0 | 0.8924 | 0.9078 | 0.8935 | 0.8988 | 0.8925 | 0.9246 | 0.8461 | 2.0086 | 0.8050 |
| M1 | 0.8924 | 0.9079 | 0.8924 | 0.9000 | 0.8920 | 0.9332 | 0.9526 | 2.4609 | 0.7065 |
| M2 | 0.8829 | 0.9035 | 0.8798 | 0.8901 | 0.8829 | 0.9246 | 0.9602 | 2.3605 | 0.7305 |
| N0 | 0.8462 | 0.8606 | 0.8535 | 0.8504 | 0.8450 | 0.8694 | 0.9510 | 2.4632 | 0.7067 |
| N1 | 0.8315 | 0.8435 | 0.8621 | 0.8383 | 0.8289 | 0.8701 | 0.9457 | 2.4329 | 0.7128 |

## Evidence hierarchy

1. `official_final_development_freeze`: corrected Phase E run `strategy-b-phase-e-prediction-20260714-9007144`.
2. `official_development_evidence`: Phase C main comparison `strategy-b-phase-c-20260714-5d34a66`.
3. `official_technical_recommendation_evidence`: Phase D `strategy-b-phase-d-recommendation-20260715-407ac0f`.
4. Phase A–B establishes protocol/correctness and the quarantine registry.
5. Historical, diagnostic, invalid, smoke and legacy-observed evidence is audit context only, never headline evidence.

Immutable artifact contents must not be edited to align with current claims. Classification is added through closure registries and warning/index documents.

## Recommendation governance

Pipeline: frozen N0 scores + R0 agreement → canonical target-free prediction snapshot → uncertainty/agreement gate → feature governance → structured goals/actions → explanation/limitation → advisor decision → follow-up/revision.

- All 316 generated drafts remain inactive and require advisor approval.
- 245 pass the normal draft gate; 71 trigger uncertainty/agreement review (22.47%).
- 1,313 actions; zero conflicts, duplicates and workload violations.
- Goal/action/explanation completeness are 100% under structural checks.
- Expert casebook contains 60 cases across 23 strata.
- `technical_validation = PASS`, `expert_validation = PENDING`, `effectiveness_validation = NOT PERFORMED`.

Policy revisions are append-only. Ratings must come from at least two real independent experts; synthetic ratings or inferred approval are forbidden.

## Prohibited claims

- “CNN–BiLSTM outperforms machine learning” or “deep learning is best”.
- “The model was confirmed on an untouched test set”.
- “Generalization is proven” or “production validated”.
- “Recommendation improves grades”, “causes improvement” or is scientifically effective.
- “Expert validation passed” before real ratings/adjudication.
- “R0 confidence is 100%” or one-hot scores are calibrated probability.

## Repository conventions

- Source and documentation changes precede immutable evidence commits.
- Artifacts use unique run IDs and checksum manifests; official artifacts are never overwritten.
- Expensive Phase runners require explicit authorization. Validators and test commands are the default entrypoints.
- Old unsafe recommendation materialization remains fail-closed.
- No credentials, `.env`, database dumps, caches or temporary logs are committed.
- Migrations are ordered numerically and destructive integration tests run only against disposable DSNs.

## Reproduction commands

Environment and tests:

```powershell
python -m pip install -r requirements-lock.txt
python -m pytest -q -rs
```

Portable final evidence check:

```powershell
python scripts/verify_final_evidence.py --skip-db
```

Authorized PostgreSQL ingestion:

```powershell
python scripts/ingest_dataset_to_postgres.py --dataset student-mat
```

Apply migrations 001→004 with `psql -v ON_ERROR_STOP=1 -f <path>` only to an authorized database. Migration integration tests require `POSTGRES_TEST_DSN`, `POSTGRES_TEST_APP_DSN` and `psql`.

Phase A–B/C/E/D evidence validation is normally performed from each run's `strict_validation.json` and `artifact_checksums.json`. `run_strategy_b_phase_c.py`, `run_strategy_b_phase_e_prediction.py` and Optuna entrypoints are expensive historical reproduction commands and must not be used for quick validation.

## Known limitations

- Small, single-course Portuguese secondary-school dataset.
- Only two temporal grade inputs; CNN incremental value is not established.
- No untouched external dataset; 79 historical records are contaminated by observation.
- N0 is uncalibrated and R0 cannot supply uncertainty.
- Context features are disabled pending semantic/timing verification.
- Expert recommendation review and prospective effectiveness study are pending.
- PostgreSQL migration 004 is statically validated in closure when disposable DSN is unavailable, but not destructively executed on production.

## Future research

1. Independent external validation on a genuinely unseen dataset.
2. Real two-expert review and adjudication of the Phase D casebook.
3. Shadow/prospective study before any effectiveness claim.
4. Longer longitudinal sequences to test whether CNN/recurrent inductive bias adds value.
5. Locally governed feature registry and context capture-time validation.
6. Drift, calibration, fairness and safety monitoring before deployment.
