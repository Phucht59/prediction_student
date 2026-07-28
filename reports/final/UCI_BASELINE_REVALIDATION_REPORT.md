# UCI Baseline Revalidation Report

This report discloses every classical UCI comparator row superseded by the safe S2 revalidation. The historical rows came from a path that applied plain SMOTE/ADASYN after categorical preprocessing. The replacement rows use the same frozen outer folds, training-only preprocessing, inner-only selection, and no synthetic resampling. The official CNN-BiLSTM results were not changed.

| Dataset | Model | Old Macro-F1 | New Macro-F1 | Old Accuracy | New Accuracy | Reason for revalidation | Same outer splits | Synthetic resampling used | Status |
|---|---|---:|---:|---:|---:|---|---|---|---|
| Student-Mat | Logistic Regression | 0.879318 | 0.895198 | 0.873418 | 0.886076 | Unsafe historical synthetic-sampling path superseded | YES | NO | PASS |
| Student-Mat | Decision Tree | 0.906654 | 0.902425 | 0.898734 | 0.893671 | Unsafe historical synthetic-sampling path superseded | YES | NO | PASS |
| Student-Mat | Random Forest | 0.901387 | 0.899833 | 0.893671 | 0.891139 | Unsafe historical synthetic-sampling path superseded | YES | NO | PASS |
| Student-Mat | HistGradientBoosting | 0.878546 | 0.869686 | 0.870886 | 0.860759 | Unsafe historical synthetic-sampling path superseded | YES | NO | PASS |
| Student-Mat | SVM | 0.814271 | 0.871030 | 0.810127 | 0.863291 | Unsafe historical synthetic-sampling path superseded | YES | NO | PASS |
| Student-Mat | XGBoost | 0.888000 | 0.881527 | 0.878481 | 0.873418 | Unsafe historical synthetic-sampling path superseded | YES | NO | PASS |
| Student-Por | Logistic Regression | 0.820541 | 0.837877 | 0.852080 | 0.864407 | Unsafe historical synthetic-sampling path superseded | YES | NO | PASS |
| Student-Por | Decision Tree | 0.848718 | 0.846088 | 0.870570 | 0.882897 | Unsafe historical synthetic-sampling path superseded | YES | NO | PASS |
| Student-Por | Random Forest | 0.869244 | 0.851381 | 0.892142 | 0.879815 | Unsafe historical synthetic-sampling path superseded | YES | NO | PASS |
| Student-Por | HistGradientBoosting | 0.850630 | 0.844056 | 0.882897 | 0.876733 | Unsafe historical synthetic-sampling path superseded | YES | NO | PASS |
| Student-Por | SVM | 0.782477 | 0.850191 | 0.838213 | 0.882897 | Unsafe historical synthetic-sampling path superseded | YES | NO | PASS |
| Student-Por | XGBoost | 0.866388 | 0.867659 | 0.895223 | 0.895223 | Unsafe historical synthetic-sampling path superseded | YES | NO | PASS |

Machine-readable authority: `artifacts/final/teacher_feedback_validation/baseline_revalidation.json`.

All current canonical UCI comparator rows use frozen outer partitions and training-only preprocessing. No current canonical UCI comparator uses plain SMOTE or ADASYN on mixed categorical data. The safe revalidation did not reselect or retrain any official CNN-BiLSTM.
