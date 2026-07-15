# Limitations and allowed claims

## Allowed

- Study B independently evaluates `student-por` under its own folds and search.
- Frozen mathematics-to-Portuguese transfer measures cross-subject domain shift, subject to quasi-identity overlap.
- Study C evaluates at-risk classification at three preregistered landmarks using cutoff-valid weekly OULAD activity.
- The Study C flagship provides real temporal modeling but **did not establish incremental advantage over the strongest ML baseline** under the preregistered rule.
- Future-presentation evaluation is a chronological/domain-shift test with global student exclusion.

## Prohibited

- Do not claim OULAD proved CNN-BiLSTM superior.
- Do not call Study B transfer fully independent external validation.
- Do not treat F1/F2/F3 as one fixed cohort or as multiple semesters for every learner.
- Do not infer causality from activity features or model explanations.
- Do not use the 79 observed Study A records as an untouched test.
- Do not hide failed/collapsed models or the skipped OULAD SVM.

## Important limitations

- OULAD `final_result` is an operational at-risk label, not a causal outcome.
- Cohorts shrink across landmarks because withdrawals before each cutoff are excluded by the landmark definition.
- Hyperparameter budgets were compute-constrained and preregistered.
- Deep stability uses declared seeds; seeds are not independent datasets.
- No model result changes the frozen Study A conclusion.
