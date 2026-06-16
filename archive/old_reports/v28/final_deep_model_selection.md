# Final Deep Model Selection V28

- Selection uses CV/OOF only: Macro F1, then Recall Low, then F1 Low.
- Locked test is used only after selection for final evaluation.
- Gated fusion is not selected unless it wins CV/OOF over sequence-only.
- Regression head is not claimed in V28.

## CV/OOF Selected Models

| dataset | scenario | candidate_id | variant | prediction_mode | macro_f1 | recall_low | f1_low |
| --- | --- | --- | --- | --- | --- | --- | --- |
| student-mat | late | seq_k2_c64_h96_attn_cbf | sequence_cnn_bilstm_v28_focal | low_f1_tuned | 0.8804 | 0.9327 | 0.8778 |
| student-por | late | seq_k2_c64_h96_attn_cbf | sequence_cnn_bilstm_v28_focal | low_f1_tuned | 0.8684 | 0.7625 | 0.7722 |
| student-por | midterm | seq_k2_c64_h96_attn_cbf | sequence_cnn_bilstm_v28_focal | low_f1_tuned | 0.8009 | 0.7125 | 0.7261 |
| xapi | xapi | gated_k3_c32_h64_attn_cw | gated_fusion_v28 | low_f1_tuned | 0.7521 | 0.9010 | 0.8545 |