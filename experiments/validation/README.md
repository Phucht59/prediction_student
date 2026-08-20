# Scientific evaluation of frozen Hybrid CNN–BiLSTM

Isolated from `src/prediction`. Does not change architecture, weights, or HPO.

```powershell
python experiments/validation/run_validation.py --dataset uci
python experiments/validation/run_validation.py --dataset oulad
```

OULAD uses existing C0 inner checkpoints (folds 0–2, seed 42). Other OULAD seeds are not materialized.

Outer-test Phase 8 parquet is **not** treated as C0 evidence.
