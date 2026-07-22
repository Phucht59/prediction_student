# Final Model Review

This review was generated only after all V5 studies and the controlled joint-learning experiment completed.

| dataset | final_model | cnn_bilstm_macro_f1 | strongest_ml | strongest_ml_macro_f1 | delta |
| --- | --- | --- | --- | --- | --- |
| student-mat | decision_tree | 0.8799168720699821 | decision_tree | 0.9018875313283208 | -0.021970659258338632 |
| student-por | random_forest | 0.8491516177055304 | random_forest | 0.8605087126041724 | -0.011357094898642006 |
| oulad | xgboost | 0.828002638861856 | xgboost | 0.8283814220319712 | -0.00037878317011519336 |

## Answers to the final scientific review

1. **Best model by dataset.** `student-mat`: Decision Tree (Macro-F1 0.901888). `student-por`: Random Forest (0.860509). OULAD: immutable V4 XGBoost comparator (0.828381).
2. **CNN–BiLSTM change from V4.** `student-mat`: +0.029552; `student-por`: +0.002190; OULAD: -0.001305. These are point-estimate deltas, not external-test claims.
3. **Source of improvement.** UCI gains are consistent with the controlled context branch, nested tuning and imbalance selection; attribution is associative, not causal.
4. **CNN contribution.** OULAD `cnn_only` is competitive but does not establish superiority.
5. **BiLSTM contribution.** `bilstm_only` is also competitive; the combined model has no stable superiority over both ablations or XGBoost.
6. **Joint learning.** `KEEP_STANDALONE`. Mean inner-validation delta 0.002406; 3/5 seeds and 3/5 outer-training partitions improved.
7. **OULAD augmentation.** Inner-only screening selected different strategies by fold; no global augmentation benefit is claimed.
8. **Overfit evidence.** Early stopping, nested folds, pruning and replay reduce risk, but small UCI cohorts leave residual risk.
9. **Seed stability.** All five fixed seeds are reported and averaged; best-seed selection was prohibited.
10. **Complexity trade-off.** CNN–BiLSTM is useful as the thesis architecture, but the operational results favor simpler models.
11. **Final prediction sources.** Decision Tree for `student-mat`, Random Forest for `student-por`, and XGBoost for OULAD.
12. **Recommendation source.** Registry-selected operational models feed the rule-based advisor-in-the-loop policy; CNN–BiLSTM remains separately reported.
13. **Allowed claims.** Nested/grouped historical-development OOF performance, checkpoint replay, and technical policy validation.
14. **Prohibited claims.** External generalization, causal improvement, production readiness, future OULAD performance, or CNN–BiLSTM superiority.
15. **What to simplify.** Use operational tree/boosting models for deployment and reserve CNN–BiLSTM for the sequence-mechanism question.

CNN–BiLSTM remains the thesis model. The operational model is selected independently by valid OOF Macro-F1; no future benchmark or best-seed selection is used.

Claims remain limited to grouped/nested development OOF. Recommendation effectiveness and causal impact are not established.

Recommendation technical validation: `PASS`; expert review `PENDING`.
