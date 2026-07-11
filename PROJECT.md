# Project status - frozen technical release

## Active architecture

- Research classifier: sequence-only CNN-BiLSTM (G1,G2 -> CNN -> BiLSTM -> linear head).
- Selection: nested 5 outer x 3 inner stratified folds, 30 Optuna trials, seed 42.
- Data source: PostgreSQL `student_predict`; CSV is ingestion-only.
- Recommender: deterministic rule-based advisory policy `student_mat_rule_policy_v3`.

## Status

| Work item | Status |
| --- | --- |
| Repository cleanup | DONE |
| PostgreSQL-first source architecture | DONE |
| Migration 003 implementation | DONE |
| Live migration application | DONE |
| PostgreSQL integration tests | DONE (5/5) |
| DB-first live reproducibility | DONE |
| Frozen scientific evidence | DONE |
| Main branch release | DONE |
| Expert recommendation review | PENDING |
| Thesis DOCX rewrite | NEXT |

## Frozen scientific conclusion

CNN-BiLSTM is technically reproducible but does not demonstrate added value
over the G2 rule: final locked Macro-F1 is 0.9262 versus 0.9365 for G2.
HistGradientBoosting locked Macro-F1 is 0.9463, but its nested outer score is
0.8690 and the locked score was not used for selection. These results must not
be replaced by a more favorable split.

## Release references

- Selection run: `nested-full-20260710`
- Final scientific DB run: `a2945d79-9845-4979-b148-159f4853eca3`
- DB-first verification run: `5a0b5041-5216-4a48-9e46-b0c16ab14866`
- Frozen config: `artifacts/model_selection/nested-full-20260710/selected_config.json`
- Active evidence: `artifacts/final/LATEST_RUN.txt`

The repository contains source, migrations, tests, technical documentation and
frozen evidence. Old DOCX/PDF reports are removed; report revision is the next
stage and is not performed by this pipeline.

The live database has one dataset version, 395 source records and 395 immutable
target rows. Migration 003, credentialed PostgreSQL tests and frozen DB-first
verification are complete. Expert recommendation review remains pending.
