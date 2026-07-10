# Final project audit

Audit date: 2026-07-10. DOCX files were not edited.

## Current state

- Tests after cleanup: 87 passed, 5 skipped (the deleted skips are legacy
  recommender-output tests; PostgreSQL integration remains environment-gated).
- Dataset version contains 395 rows with Low/Medium/High counts 130/192/73.
- Final split ledger contains 316 train and 79 locked-test records.
- Final DB run `a2945d79-9845-4979-b148-159f4853eca3` is completed and stores
  79 predictions and 79 recommendations.
- Reproducibility run `c719439e-bb88-42ff-bb98-d258c21d204e` has an exact
  prediction checksum match with the scientific run.

## PostgreSQL-first contract

CSV is read only by `ingest_dataset_to_postgres.py` and
`ingest_dataset_csv_to_postgres`. The model-selection and final pipeline load
rows through `src/data/postgres_dataset_loader.py` / `src/postgres_data_source.py`.
Migration 003 adds immutable `source_record_targets`; target labels are not
used as feature columns. Source-record identity, dataset version and split
ledger remain composite-key lineage constraints.

## Scientific results

- CNN-BiLSTM nested outer Macro-F1: 0.8781 +/- 0.0448.
- CNN-BiLSTM locked-test Macro-F1: 0.9262.
- G2 rule locked-test Macro-F1: 0.9365.
- HGB nested outer Macro-F1: 0.8690; HGB locked-test Macro-F1: 0.9463.

HGB 0.8969 is a separate train-pool OOF protocol and is not mixed with the
nested comparison; see `MODEL_COMPARISON_PROTOCOL.md`. No locked-test result
was used for model selection.

## Cleanup and limitations

Legacy checkpoints, report context, old DOCX/PDF outputs and smoke artifacts
were removed from the working tree. Expert recommendation ratings remain blank.
Fairness slices are descriptive only because the locked test has small groups.
The project is ready for thesis report revision from frozen evidence.
