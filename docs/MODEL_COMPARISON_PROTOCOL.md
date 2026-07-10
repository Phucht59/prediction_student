# Model Comparison Protocol

All primary model-selection rows are loaded from the PostgreSQL dataset version
and use the same source-record lineage. CSV is ingestion-only.

Các số HGB dưới đây thuộc các protocol khác nhau và không được trộn lẫn.

| Result name | Model | Scenario | Validation protocol | Feature set | Tuned | Selection basis | Metric |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline evidence | HistGradientBoosting | late-stage | Train-pool 5-fold OOF; fixed 80/20 test | all valid source features | Candidate model selected by train OOF | OOF Macro-F1 | OOF 0.8969; locked test 0.9463 |
| Nested comparison | HistGradientBoosting | late-stage | Same 5 outer folds as CNN nested run | Engineered tabular features after fold-local preprocessing/selection | Fixed estimator; no HGB hyperparameter Optuna | Outer-fold evaluation only | Outer Macro-F1 0.8690 |
| Final scientific model | CNN–BiLSTM | late-stage | 5 outer × 3 inner folds, 30 Optuna trials | G1, G2 sequence only | CNN params/preprocessing selected inside inner CV | Mean inner-CV Macro-F1; fixed seed 42 | Outer Macro-F1 0.8781 ± 0.0448; locked test 0.9262 |

## Interpretation

- The primary like-for-like nested-CV comparison is CNN–BiLSTM `0.8781 ±
  0.0448` versus HGB `0.8690` on the nested outer folds.
- HGB `0.8969` is a different train-pool OOF protocol with a different feature
  pipeline; it is retained as a transparent baseline-evidence result, not
  substituted into the nested comparison.
- HGB locked-test `0.9463` and G2-rule locked-test `0.9365` are final
  evaluations only. They were not used to select the CNN–BiLSTM config.
- The CNN final locked-test Macro-F1 `0.9262` remains below both simple
  late-stage baselines. No claim of overall superiority is justified.
