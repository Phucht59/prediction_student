# Stage-Aware Causal Recommendation Evidence

## Claim boundary

This report contains observational target-trial estimates under measured-confounding, positivity, consistency, and model assumptions. It does not prove randomized or deployed recommendation effectiveness.

## Protocol

- Stages: EARLY_20, EARLY_35, MIDDLE_50, LATE_75
- Actions: ASSESSMENT_COMPLETION, STUDY_REGULARITY, VLE_ENGAGEMENT, QUIZ_OR_RETRIEVAL_PRACTICE, CONTENT_REVIEW
- Cross-fit folds: 3
- Student-cluster bootstrap iterations: 1000
- Recommendation lifecycle: latest valid recommendation wins.

## Stage-action effects

| Stage | Action | Status | N | Treated | Control | ATE | 95% CI | Max SMD | ESS fraction |
|---|---|---|---:|---:|---:|---:|---|---:|---:|
| EARLY_20 | ASSESSMENT_COMPLETION | CAUSAL_EVIDENCE_NOT_IDENTIFIABLE | 2205 | 1221 | 984 | 0.1232 | [0.0820, 0.1643] | 0.3252 | 0.6673 |
| EARLY_20 | STUDY_REGULARITY | CAUSAL_EVIDENCE_NOT_IDENTIFIABLE | 465 | 43 | 422 | 0.2377 | [0.1884, 0.2870] | 0.4546 | 0.9715 |
| EARLY_20 | VLE_ENGAGEMENT | CAUSAL_EVIDENCE_NOT_IDENTIFIABLE | 236 | 12 | 224 | 0.1703 | [0.1177, 0.2228] | 0.8304 | 0.9732 |
| EARLY_20 | QUIZ_OR_RETRIEVAL_PRACTICE | CAUSAL_EVIDENCE_NOT_IDENTIFIABLE | 5744 | 1214 | 4530 | 0.1237 | [0.0856, 0.1618] | 0.1116 | 0.9361 |
| EARLY_20 | CONTENT_REVIEW | CAUSAL_EVIDENCE_NOT_IDENTIFIABLE | 3585 | 636 | 2949 | 0.1965 | [0.1524, 0.2406] | 0.2672 | 0.9621 |
| EARLY_35 | ASSESSMENT_COMPLETION | CAUSAL_EVIDENCE_NOT_IDENTIFIABLE | 2436 | 1332 | 1104 | 0.0855 | [0.0530, 0.1180] | 0.3253 | 0.6312 |
| EARLY_35 | STUDY_REGULARITY | CAUSAL_EVIDENCE_NOT_IDENTIFIABLE | 2866 | 574 | 2292 | 0.2236 | [0.1779, 0.2693] | 0.1281 | 0.9444 |
| EARLY_35 | VLE_ENGAGEMENT | CAUSAL_EVIDENCE_NOT_IDENTIFIABLE | 2689 | 527 | 2162 | 0.2583 | [0.2156, 0.3010] | 0.2513 | 0.9490 |
| EARLY_35 | QUIZ_OR_RETRIEVAL_PRACTICE | CAUSAL_EVIDENCE_NOT_IDENTIFIABLE | 6689 | 1678 | 5011 | 0.1746 | [0.1439, 0.2054] | 0.1205 | 0.8974 |
| EARLY_35 | CONTENT_REVIEW | CAUSAL_EVIDENCE_NOT_IDENTIFIABLE | 9851 | 2517 | 7334 | 0.2032 | [0.1763, 0.2302] | 0.0365 | 0.9020 |
| MIDDLE_50 | ASSESSMENT_COMPLETION | CAUSAL_EVIDENCE_NOT_IDENTIFIABLE | 2078 | 807 | 1271 | 0.1589 | [0.1128, 0.2050] | 0.5067 | 0.7280 |
| MIDDLE_50 | STUDY_REGULARITY | CAUSAL_EVIDENCE_NOT_IDENTIFIABLE | 497 | 45 | 452 | 0.1829 | [0.1365, 0.2294] | 0.4240 | 0.9826 |
| MIDDLE_50 | VLE_ENGAGEMENT | CAUSAL_EVIDENCE_NOT_IDENTIFIABLE | 1117 | 103 | 1014 | 0.2991 | [0.2730, 0.3252] | 0.3061 | 0.9806 |
| MIDDLE_50 | QUIZ_OR_RETRIEVAL_PRACTICE | CAUSAL_EVIDENCE_NOT_IDENTIFIABLE | 2477 | 404 | 2073 | 0.1430 | [0.1027, 0.1833] | 0.1981 | 0.9654 |
| MIDDLE_50 | CONTENT_REVIEW | CAUSAL_EVIDENCE_NOT_IDENTIFIABLE | 2672 | 417 | 2255 | 0.2360 | [0.2008, 0.2713] | 0.2743 | 0.9702 |
| LATE_75 | ASSESSMENT_COMPLETION | CAUSAL_EVIDENCE_NOT_IDENTIFIABLE | 1860 | 684 | 1176 | 0.2091 | [0.1674, 0.2508] | 0.3944 | 0.7294 |
| LATE_75 | STUDY_REGULARITY | CAUSAL_EVIDENCE_NOT_IDENTIFIABLE | 186 | 11 | 175 | 0.2303 | [0.1899, 0.2708] | 0.6749 | 0.9547 |
| LATE_75 | VLE_ENGAGEMENT | CAUSAL_EVIDENCE_NOT_IDENTIFIABLE | 191 | 22 | 169 | 0.3264 | [0.2931, 0.3596] | 0.6268 | 0.9770 |
| LATE_75 | QUIZ_OR_RETRIEVAL_PRACTICE | CAUSAL_EVIDENCE_NOT_IDENTIFIABLE | 1790 | 435 | 1355 | 0.1112 | [0.0789, 0.1436] | 0.1008 | 0.8842 |
| LATE_75 | CONTENT_REVIEW | CAUSAL_EVIDENCE_NOT_IDENTIFIABLE | 862 | 118 | 744 | 0.2035 | [0.1523, 0.2546] | 0.4412 | 0.9671 |

## Frozen Hybrid imbalance sensitivity

These experiments retrain only an identical linear head over frozen Hybrid embeddings. They do not replace the canonical checkpoint.

| Mode | Train rows fitted | Threshold | ROC-AUC | PR-AUC | Precision | Recall | F1 | Balanced accuracy | Specificity | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 20930 | 0.6336 | 0.8438 | 0.8214 | 0.7437 | 0.6745 | 0.7074 | 0.7587 | 0.8428 | 0.1672 |
| class_weight | 20930 | 0.7569 | 0.8446 | 0.8219 | 0.7902 | 0.6259 | 0.6985 | 0.7568 | 0.8876 | 0.1841 |
| smote | 24592 | 0.8714 | 0.8457 | 0.8218 | 0.8722 | 0.5175 | 0.6496 | 0.7331 | 0.9487 | 0.1909 |
| adasyn | 24974 | 0.8462 | 0.8473 | 0.8233 | 0.8982 | 0.4788 | 0.6246 | 0.7211 | 0.9633 | 0.1814 |

## Interpretation rules

- `CAUSAL_EFFECT_ESTIMATED` means the preregistered overlap, count, balance, ESS, and bootstrap gates passed.
- `CAUSAL_EVIDENCE_NOT_IDENTIFIABLE` means no causal-effect claim is allowed for that action-stage pair.
- Positive ATE/CATE estimates describe the overlap population represented by the observational data.
- The absence of expert labels, deployment, and randomized assignment remains a limitation.

Validation status: **PASS**
