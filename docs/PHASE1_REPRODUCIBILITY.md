# Phase 1: Reproducibility Protocol

This document defines the evidence required before any model optimization work begins.

## Scope

Phase 1 does **not** change the CNN-BiLSTM architecture, feature set, sampling strategy, threshold, or final model selection. Its purpose is to make the current baseline reproducible from raw data and to preserve all artifacts needed for a scientifically valid comparison in later phases.

## Required artifacts per dataset

For each dataset (`student-mat`, `student-por`, `xapi`), the pipeline must create:

- `data/manifests/<dataset>_raw_manifest.json`: source metadata, file SHA-256, schema, row count, and class distribution.
- `data/processed/final/<dataset>_3class_split_indices.json`: row-level split membership generated from the raw file.
- `data/processed/final/<dataset>_3class_train_pool.csv` and `..._locked_test.csv`: deterministic split files.
- `models/saved/final/<dataset>_3class_best_params.json`: frozen parameters used for a run.
- `models/saved/final/<dataset>_3class_experiment_manifest.json`: run metadata, source commit, split hash, feature list, seed list, and evaluation protocol.
- `reports/final/predictions/<dataset>_3class_predictions.csv`: locked-test labels, predicted labels, confidence, and class probabilities.
- `reports/final/metrics/<dataset>_3class_locked_test_metrics.json`: final metrics.

## Data integrity rules

1. Raw datasets are never committed to Git.
2. Dataset identity is verified using SHA-256 and schema metadata.
3. The locked test split is generated once from the raw file using `DEFAULT_SEED=42` and stored as raw-row indices.
4. A split is reused only when its dataset file SHA-256 matches the current raw file.
5. The locked test set is not used to tune hyperparameters, thresholds, feature engineering, or model architecture.
6. Resampling happens only inside a training fold.

## Commands

Create and validate deterministic splits:

```powershell
py -3.10 scripts\prepare_reproducibility.py --dataset xapi
py -3.10 scripts\prepare_reproducibility.py --dataset student-mat
py -3.10 scripts\prepare_reproducibility.py --dataset student-por
```

Validate all available datasets without overwriting existing splits:

```powershell
py -3.10 scripts\prepare_reproducibility.py --dataset all --verify-only
```

Run the test suite:

```powershell
py -3.10 -m pytest -q
```

## Phase-1 exit criteria

Phase 1 is complete for a dataset only when:

- Raw file exists and matches its manifest.
- Locked split files and split-index manifest exist.
- The split passes overlap and class-distribution checks.
- The pipeline can load the split without regenerating it.
- An experiment manifest is written with the exact configuration and artifact references.

No metric from a later optimization phase should be called final until these requirements are satisfied.
