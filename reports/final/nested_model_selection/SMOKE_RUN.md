# Nested model-selection smoke run — not final evidence

This directory was generated on 2026-07-10 with:

```powershell
py -3.10 scripts/optimize_model_selection.py --dataset student-mat --dataset-version-id 1 --n-trials 1 --outer-folds 2 --inner-folds 2
```

It verifies the PostgreSQL DB-first split reconstruction, train-only nested
selection and selected-config export path. It is not a final nested-CV result:
it has only one Optuna trial and two outer/inner folds. Do not cite its metrics
in the thesis or feed its `selected_config.json` to a final locked-test run.

For a final selection run, use at least five outer folds, four or five inner
folds and a predeclared trial budget, then keep its output in a distinct run
directory with an explicit final manifest.
