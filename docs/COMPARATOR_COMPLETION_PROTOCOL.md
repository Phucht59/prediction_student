# Final Comparator Completion Protocol

Protocol ID: `final-comparator-completion-20260723-v1`

Status: **PREREGISTERED BEFORE COMPARATOR TRAINING**

This protocol completes the missing cells in the final nine-model comparison. It does not reopen model selection for the official CNN-BiLSTM systems, does not retrain any official deep model, and does not change the released recommendation system. Future OULAD remains `LOCKED_NOT_EXECUTED`.

## Immutable boundaries

- The official CNN-BiLSTM architecture, checkpoints, ensemble members, thresholds, registry selection, and released predictions are immutable.
- The Student Risk-Based Recommendation System, its input model, plans, technical results, and expert status are immutable.
- The integrated verdict remains `INTEGRATED_SYSTEM_PASS_EXPERT_PENDING`.
- OULAD evaluation is restricted to the 15,378-record historical-development `F2_MIDDLE` cohort. Future labels and predictions are prohibited.
- Existing historical comparator summaries that lack record-aligned native probabilities or use another feature/split contract are `DO_NOT_IMPORT`.

## Missing-result actions

`DERIVE_ONLY` is used when frozen record-aligned probabilities exist. `REPLAY_INFERENCE` is allowed only when a complete model, preprocessor, split, and config bundle exists and replay agrees with its registry result. `TRAIN_COMPLETION_MODEL` is limited to Student-Mat XGBoost, Student-Por XGBoost, and the six registered OULAD ML comparators.

The inventory found no complete OULAD ML replay bundle with the final unified feature contract. Therefore all six OULAD ML comparators are trained under the same contract. This is not a performance-driven choice: it prevents mixing historical incompatible protocols.

## Data and split contracts

Student-Mat and Student-Por retain their V5.1 class mapping, safe temporal/context features, five frozen outer folds, and three inner folds. XGBoost receives exactly the same raw information as the existing ML comparators. `G3` is never an input.

OULAD uses the frozen V5 grouped outer manifest, three outer folds, three grouped inner folds, and group key `id_student`. The unified ML vector is the deterministic compact aggregate allowlist plus the frozen static features implemented by `src/studies/v5_1/oulad/data.py`. Imputation, scaling, and one-hot vocabularies are fit separately inside each current training partition.

Canonical split checksums:

| Dataset | Manifest SHA-256 |
| --- | --- |
| Student-Mat | `3b1dbfc8e359f415e70e1e607a0f14b44d26bfc1bd0616650d0c2346509171f5` |
| Student-Por | `2ea07b2d17714fc9df3eec23150579090d245a1a45f33270b2864c88653d2de2` |
| OULAD | `ae8d20773ccfda9123a9ef5e7162f0a324f80edd8dcb152ae2c97f1b6814cbb0` |

## Selection and ensemble

All hyperparameters are selected solely by three-fold inner CV within each outer-training fold. The primary criterion is Macro-F1. UCI XGBoost ties are resolved by balanced accuracy, macro PR-AUC, lower Brier, then smaller estimator count. Outer-validation labels are used only for final OOF evaluation and never for selection.

The fixed seeds are `42, 1201, 2026, 3407, 7319`. No seed may be selected. Final record probability is the arithmetic mean of all five seed probabilities. Metrics are calculated from that ensemble probability, while seed stability is reported separately.

For OULAD, each model and outer fold receives a threshold chosen only from pooled inner-OOF five-seed ensemble predictions. The threshold maximizes Macro-F1, with risk recall, risk precision, and proximity to 0.5 as deterministic tie-breakers. It is then frozen for that outer fold.

## Search budgets

Student-Mat and Student-Por independently evaluate 12 deterministic sampled XGBoost configurations per outer fold. OULAD budgets per model and outer fold are: LR 6, DT 8, RF 6, HGB 6, RBF-SVM 6, and XGBoost 6. The complete discrete ranges are machine-locked in `configs/final/comparator_completion_protocol.yaml`.

RBF-SVM is `SVC(kernel="rbf", probability=True)`. It is run serially, without subsampling or kernel substitution. XGBoost uses CPU `tree_method="hist"`. RF and XGBoost use at most three worker threads.

## Metrics and artifacts

All nine models receive Accuracy, Balanced Accuracy, Macro Precision, Macro Recall, Macro-F1, Weighted-F1, PR-AUC, ROC-AUC, Brier, NLL, and ECE. OULAD additionally receives risk precision/recall/F1 and common 5%, 10%, and 20% top-k metrics.

Every metric carries the prediction artifact, checksum, protocol hash, split-manifest hash, feature-contract hash, and calculation method. Per-class rows contain only class, precision, recall, F1, and support. Confusion matrices are retained for all nine models on all three datasets.

Paired bootstrap uses records for UCI and complete `id_student` groups for OULAD. The only allowed verdicts are `CNN_BILSTM_HIGHER`, `COMPARATOR_HIGHER`, and `PRACTICAL_TIE`; it never changes the official selected model.

## Failure, resume, and amendments

Artifacts are flushed after every outer fold and accepted completed folds are reused only after checksum validation. An interrupted run resumes without duplicating valid predictions. Applicable `N/A` cannot be converted to PASS.

A resource or implementation defect may require a separately committed amendment. An amendment must be registered before rerunning affected work and may not alter the protocol merely to improve an observed result.

