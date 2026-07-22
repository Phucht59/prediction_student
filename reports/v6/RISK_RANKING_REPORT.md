# V6 risk-ranking report

| Candidate | Lambda | Macro-F1 | PR-AUC | Recall@10% | Gate |
|---|---:|---:|---:|---:|---:|
| R0 | 0.05 | 0.801960 | 0.890236 | 0.250108 | false |
| R1 | 0.10 | 0.816274 | 0.889631 | 0.249626 | false |

Selected: **C_TEMPORAL_MULTITASK**. Pairs use only records in the same module,
presentation and course-progress bucket. Near-probability matching uses the
cross-fitted V5.4 XGBoost teacher on outer-training fold 0. No outer-test or
Future OULAD record was accessed.
