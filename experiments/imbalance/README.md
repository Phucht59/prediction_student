# Imbalance experiment: SMOTE / ADASYN on frozen Hybrid CNN–BiLSTM

Isolated module. Does **not** change production prediction or recommendation.

Question: under the exact frozen Hybrid configuration, does SMOTE or ADASYN on **training data only** improve VALID metrics vs CONTROL?

```text
experiments/imbalance/          code
artifacts/experiments/imbalance/  tables + audits
reports/prediction/experiments/imbalance/  report
```

```powershell
python experiments/imbalance/run_imbalance_experiment.py --dataset uci
python experiments/imbalance/run_imbalance_experiment.py --dataset oulad
python -m pytest tests/test_imbalance_experiment.py tests/prediction/test_phase4_authority.py -q
```

SMOTE/ADASYN run only on FIT tensors after FIT-only preprocessing. STOP/VALID/outer are never resampled. Production Hybrid under `src/prediction` is not modified.
