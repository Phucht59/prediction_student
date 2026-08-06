# Four-Stage Conditional Action Head

## Scope

The frozen Hybrid CNN–BiLSTM representation was not modified. Only the integrated conditional action head was trained. Thresholds were calibrated on validation rows and all reported final metrics use held-out out-of-fold rows.

## Overall held-out evidence

- Conditional Precision@1: **1.0000**
- NDCG@3: **1.0000**
- MRR: **1.0000**
- End-to-end Precision@1: **1.0000**
- Positive coverage: **0.9973**
- Abstention: **0.0758**

## Per-stage held-out evidence

| Stage | Groups | Conditional P@1 | NDCG@3 | MRR | E2E P@1 | Coverage | Abstention |
|---|---:|---:|---:|---:|---:|---:|---:|
| EARLY_20 | 16566 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9867 | 0.2861 |
| EARLY_35 | 15946 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| MIDDLE_50 | 15378 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| LATE_75 | 14635 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

## Release gate

Status: **FOUR_STAGE_CONDITIONAL_RANKING_OFFLINE_VALIDATED**

- four_stage_coverage: PASS — actual=['EARLY_20', 'EARLY_35', 'LATE_75', 'MIDDLE_50'], required=['EARLY_20', 'EARLY_35', 'MIDDLE_50', 'LATE_75']
- late_75_group_count: PASS — actual=14635, required=100
- overall_ranking_precision: PASS — actual=1.0, required=0.85
- overall_ndcg_at_3: PASS — actual=1.0, required=0.9
- overall_mrr: PASS — actual=1.0, required=0.9
- minimum_stage_precision: PASS — actual=1.0, required=0.8
- action_diversity: PASS — actual=5, required=4
- student_split_leakage: PASS — actual=0, required=0
- frozen_hybrid_modified: PASS — actual=False, required=False

## Claim boundary

This evidence validates four-stage offline conditional action ranking against train-only scientific silver labels. It does not establish expert agreement, user acceptance, end-to-end deployment effectiveness, or causal grade improvement.
