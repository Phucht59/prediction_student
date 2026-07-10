# Project status — final-model evidence

## Active architecture

- Production research classifier: sequence-only CNN–BiLSTM (`G1,G2` → CNN →
  BiLSTM → linear head), without Context MLP.
- Ablation variants: CNN-only, BiLSTM-only and CNN–BiLSTM.
- Recommender: deterministic rule-based advisory policy, version
  `student_mat_rule_policy_v3`; it is not an MLP or a learned recommender.
- Final selected config: `artifacts/model_selection/nested-full-20260710/selected_config.json`
  (single seed 42, no resampling/class weight); database run
  `a2945d79-9845-4979-b148-159f4853eca3` is completed.
- G2 threshold remains stronger on locked test (Macro-F1 0.9365) than the
  frozen CNN–BiLSTM final run (0.9262).

## Completed evidence work

| Work item | Status |
| --- | --- |
| Deterministic 80/20 split, hashes and dataset manifest | DONE |
| Leakage guard tests and train-only preprocessing | DONE |
| G2, Logistic Regression and HistGradientBoosting baselines | DONE |
| Late-stage, early-warning and pre-assessment scenarios | DONE |
| CNN-only/BiLSTM-only/CNN–BiLSTM and imbalance ablations | DONE |
| 11-seed ensemble evaluation | DONE |
| Calibration, ordinal metrics, PR data and bootstrap CIs | DONE |
| Recommendation schema, deterministic checks and fairness slices | DONE |
| PostgreSQL source-record lineage tests | DONE (integration tests skip without DB) |
| Full nested selection and frozen DB-first final run | DONE |

## Remaining before thesis report revision

1. Obtain human/expert ratings for the supplied recommendation review cases;
   do not invent these scores.
2. Rewrite the report from evidence artifacts. The current DOCX remains
   untouched in this code-completion phase.
