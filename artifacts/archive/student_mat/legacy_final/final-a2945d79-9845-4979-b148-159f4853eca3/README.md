# Final DB-first evidence run

Data source: PostgreSQL `student_predict`, dataset version 1; CSV was used only
for the one-time ingestion step. Predictions and recommendations are joined to
source-record lineage and the split ledger in the database.

- Final run: `a2945d79-9845-4979-b148-159f4853eca3` (completed)
- Selection: nested-full-20260710, 5 outer × 3 inner folds, 30 trials, fixed seed 42.
- Locked-test Macro-F1: 0.9262.
- CNN–BiLSTM is not claimed to beat the G2 baseline; see `baseline_results.csv`.
- No expert review score is fabricated.
