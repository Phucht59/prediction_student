# Missing Result Audit

Audit scope: final release at starting commit `f702aea20e0df5d8c23cc0f637c31b83c7c58748`.

The `final_results_v1` release contained 252 serialized missing fields: 27 for Student-Mat, 27 for Student-Por, and 198 for OULAD. Some of these are duplicated per-class `macro_f1` fields that are removed in `final_results_v2`; all remaining applicable cells must be populated.

| Dataset | Model | Existing evidence | Classification | Resolution |
| --- | --- | --- | --- | --- |
| Student-Mat | CNN-BiLSTM, CNN-only, BiLSTM-only, LR, DT, RF, HGB, SVM | Frozen OOF probabilities | DERIVE_ONLY | Recalculate complete common schema |
| Student-Mat | XGBoost | No final model or OOF probability | TRAIN_COMPLETION_MODEL | Nested-CV XGBoost completion |
| Student-Por | CNN-BiLSTM, CNN-only, BiLSTM-only, LR, DT, RF, HGB, SVM | Frozen OOF probabilities | DERIVE_ONLY | Recalculate complete common schema |
| Student-Por | XGBoost | No final model or OOF probability | TRAIN_COMPLETION_MODEL | Nested-CV XGBoost completion |
| OULAD | CNN-BiLSTM, CNN-only, BiLSTM-only | Frozen OOF probabilities; ROC-AUC absent in old table | DERIVE_ONLY | Calculate all probability and label metrics |
| OULAD | LR, HGB, XGBoost | Historical aggregate metrics, no complete protocol-matched probability/model bundle | DO_NOT_IMPORT + TRAIN_COMPLETION_MODEL | Train with unified OULAD ML contract |
| OULAD | DT, RF, SVM | No final protocol-matched probability/model bundle | TRAIN_COMPLETION_MODEL | Train with unified OULAD ML contract |

No candidate qualified for `REPLAY_INFERENCE`: no complete OULAD comparator bundle combined model, fitted preprocessing, selected configuration, fold threshold, and exact final-protocol record mapping.

Historical OULAD comparator metrics are retained only as reference evidence and are not imported. This prevents mixing the matched-vector/oracle and compact-feature protocols, hard-label metrics, or any artifact whose development/future scope cannot be proven record by record.

## Immutable evidence guard

Before training, the completion run snapshots byte hashes for all official deep checkpoints, official OOF/seed predictions, recommendation artifacts, expert-status artifacts, and the Future-lock protocol. The final validator must prove exact equality with this snapshot.

Future OULAD remains `LOCKED_NOT_EXECUTED`; the completion runner accepts only `development_oof` and `F2_MIDDLE`.

