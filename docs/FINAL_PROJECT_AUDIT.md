# Final Project Audit

Audit date: 2026-07-10. Scope: code, experiment protocol and reproducibility
artifacts only; no DOCX report was edited in this work.

## Verified current state

- Final tests: `88 passed, 5 skipped` with Python 3.10.
- Dataset: `data/raw/student-mat.csv`, 395 rows; deterministic stratified
  train/locked-test membership is saved with SHA-256 hashes.
- `G3`/`G3_raw` and lineage fields are excluded from sequence inputs. Scaling,
  feature selection and resampling are fit on fold-training data only.
- PostgreSQL schema/integration contracts are covered by tests; integration
  tests skip when a database is unavailable.

## Root causes corrected

| Finding | Correction |
| --- | --- |
| Final training ignored Optuna-selected epoch/patience values | Final training now reads `max_epochs`, `patience` and `scheduler_patience` from frozen params. |
| SMOTE was always combined with class weights | `class_weight_mode` is independent and explicitly ablated. |
| Two dropout values had one effective role | Sequence dropout acts before BiLSTM; head dropout acts after BiLSTM pooling. |
| Kernel 3 had no defensible temporal meaning for two inputs | Student search uses kernel 1. |
| Final command could tune while evaluating | Non-debug `run_pipeline.py` requires frozen `--selection-config-json`. |
| README/PROJECT and legacy manifest were inconsistent | Project docs now point to machine-generated evidence and label legacy artifacts. |
| Recommender used socially sensitive proxy variables | Automatic policy removes sex, school, address, guardian, paid classes, alcohol and going-out fields; human review disclaimer is mandatory. |

## Actual evidence results

The latest machine-generated bundle is identified by
`artifacts/final/LATEST_RUN.txt`. Its `baseline_results.csv` and
`deep_ablation_results.csv` are the source of truth.

- G2 rule: OOF Macro-F1 0.8988; locked-test Accuracy 0.9241; Macro-F1 0.9365.
- CNN–BiLSTM, single seed: OOF Macro-F1 0.8422; locked-test Macro-F1 0.9098.
- CNN–BiLSTM, selected 11-seed ensemble: OOF Macro-F1 0.8505; locked-test
  Macro-F1 0.8876.
- Early-warning without G2: best predefined baseline OOF Macro-F1 0.6974.
- Pre-assessment without G1/G2: best predefined baseline OOF Macro-F1 0.4344.

## Final selection and database run

- Full nested protocol: 5 outer folds, 3 inner folds, 30 Optuna trials per
  inner search, fixed model-selection seed 42, fixed argmax/no calibration.
- Selection run: `nested-full-20260710`; frozen config checksum:
  `cda38460197627ac1d71e764f61d784e4c03cf6f86775339d38787c6890678ad`.
- Final database run: `a2945d79-9845-4979-b148-159f4853eca3`, status
  `completed`; 316 train memberships, 79 test memberships, 79 test
  predictions and 79 recommendations.
- Nested outer Macro-F1: `0.8781 ± 0.0448`. Final locked-test Accuracy:
  `0.9114`; Macro-F1: `0.9262`; QWK: `0.9152`; ordinal MAE: `0.0886`.
- Final evidence: `artifacts/final/final-a2945d79-9845-4979-b148-159f4853eca3/`.

The deep model does not beat the G2 baseline. This is a scientific result, not
an implementation error to hide. The held-out HistGradientBoosting score is
higher than G2, but it was not selected because its OOF score is lower.

## Known limitations and next action

1. The earlier DB-first smoke run remains non-final and is explicitly marked in
   `reports/final/nested_model_selection/SMOKE_RUN.md`; it was superseded by
   the full selection and final run recorded above.
2. The final frozen run uses no calibration by protocol; the prior calibration
   ablation remains descriptive only.
   recommendation confidence should remain advisory, not a decision threshold.
3. Fairness slices are descriptive because the 79-row locked test has small
   groups. No fairness conclusion should be claimed from them.
4. Recommendation evaluation is structural only. Expert ratings are blank by
   design in `recommendation_expert_review_cases.csv`.
