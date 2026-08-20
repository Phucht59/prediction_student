# Block A — C0 integration

**STATUS: PASS**

C0 row-level OOF did not exist on disk. Materialized with frozen Hybrid CNN-BiLSTM (`src.prediction.Hybrid`), frozen HPO, verified inner splits, no outer, no HPO.

| Fold | stop macro PR-AUC | 20pct | 35pct | 50pct | 75pct |
|---|---:|---:|---:|---:|---:|
| 0 | 0.841 | 0.762 | 0.814 | 0.860 | 0.901 |
| 1 | 0.843 | 0.762 | 0.808 | 0.840 | 0.880 |
| 2 | 0.850 | 0.745 | 0.801 | 0.843 | 0.887 |

66,685 student×course×stage OOF rows. Uncertainty = H2(p). 100pct excluded from the intervention table. Identity: `query_id = id_student::module::presentation::stage`.

Artifacts: `artifacts/recommend_hybrid/v3/data/c0_oof_predictions.parquet`, `C0_PREDICTION_PROVENANCE.json`, `learner_stage_features.parquet`.
