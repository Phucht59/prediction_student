# Canonical V3 final model results

## Main results

| dataset | model | macro_f1 | pr_auc | roc_auc | nll | brier | ece |
|---|---|---|---|---|---|---|---|
| student_mat | decision_tree | 0.906654 | 0.860926 | 0.930036 | 0.376802 | 0.181622 | 0.012826 |
| student_mat | hist_gradient_boosting | 0.878546 | 0.931758 | 0.960916 | 0.359275 | 0.188875 | 0.059364 |
| student_mat | hybrid | 0.852841 | 0.936565 | 0.961563 | 0.410247 | 0.236851 | 0.078774 |
| student_mat | logistic_regression | 0.879318 | 0.950031 | 0.969941 | 0.295236 | 0.181299 | 0.022350 |
| student_mat | mlp | 0.859507 | 0.950314 | 0.968733 | 0.338544 | 0.198479 | 0.079674 |
| student_mat | random_forest | 0.901387 | 0.955038 | 0.972050 | 0.279860 | 0.166895 | 0.032820 |
| student_mat | svm | 0.814271 | 0.882667 | 0.931646 | 0.444855 | 0.270200 | 0.049072 |
| student_mat | xgboost | 0.888000 | 0.950628 | 0.968927 | 0.296137 | 0.173018 | 0.038104 |
| student_por | decision_tree | 0.848718 | 0.896624 | 0.957625 | 0.328184 | 0.177744 | 0.031940 |
| student_por | hist_gradient_boosting | 0.850630 | 0.902272 | 0.956602 | 0.361683 | 0.178999 | 0.055296 |
| student_por | hybrid | 0.851931 | 0.915007 | 0.966353 | 0.345678 | 0.199642 | 0.026486 |
| student_por | logistic_regression | 0.820541 | 0.912455 | 0.955490 | 0.338273 | 0.194548 | 0.038501 |
| student_por | mlp | 0.830399 | 0.914738 | 0.960210 | 0.302195 | 0.173544 | 0.047514 |
| student_por | random_forest | 0.869244 | 0.930861 | 0.968927 | 0.272203 | 0.156863 | 0.030607 |
| student_por | svm | 0.782477 | 0.829693 | 0.910326 | 0.479510 | 0.250667 | 0.036357 |
| student_por | xgboost | 0.866388 | 0.936122 | 0.968917 | 0.263150 | 0.151166 | 0.028089 |
| oulad | decision_tree | 0.875871 | 0.899444 | 0.915129 | 0.338855 | 0.091986 | 0.020567 |
| oulad | hist_gradient_boosting | 0.891350 | 0.932776 | 0.942661 | 0.249208 | 0.072944 | 0.017528 |
| oulad | hybrid | 0.894071 | 0.934988 | 0.944963 | 0.242288 | 0.071513 | 0.007871 |
| oulad | logistic_regression | 0.891358 | 0.931829 | 0.941912 | 0.264721 | 0.079279 | 0.023842 |
| oulad | mlp | 0.895349 | 0.935441 | 0.945597 | 0.240305 | 0.071245 | 0.006895 |
| oulad | random_forest | 0.889279 | 0.931477 | 0.942321 | 0.255000 | 0.075328 | 0.030488 |
| oulad | svm | 0.892274 | 0.929159 | 0.937072 | 0.255241 | 0.074488 | 0.011038 |
| oulad | xgboost | 0.892991 | 0.936186 | 0.946118 | 0.240311 | 0.071112 | 0.004584 |

Secondary stage results remain separate and are never averaged into an endpoint.

## Scientific interpretation

- Student-Mat: the Hybrid ranks 7th at the main endpoint (Macro-F1 0.852841);
  Decision Tree is best by Macro-F1 at 0.906654.
- Student-Por: the Hybrid ranks 3rd at the main endpoint (0.851931); Random
  Forest is best at 0.869244.
- OULAD FINAL: H1 ranks 2nd at 0.894071, only 0.001278 below MLP. The paired
  95% bootstrap interval is [-0.004222, 0.001668], supporting a practical tie,
  not robust Hybrid superiority.
- The canonical FINAL result is 0.041580 above H1 at 75%, so the historical
  0.8503-to-0.7984 drop was not caused by later legitimate information. Phase
  7 evaluated a separately trained 50% F2 endpoint and was not protocol-compatible
  with the 75% result.

The unified architecture evaluation therefore **partially supports** the thesis:
the OULAD H1 family is highly competitive and leads at several early-warning
stages, while the frozen UCI Hybrid does not dominate classical ML at the main
endpoints.
