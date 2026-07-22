# V5.1 reproduction report

Status: **PASS**

The locked V5.1 CNN–BiLSTM was replayed from all 15 registered checkpoints
(3 outer folds × 5 fixed seeds). Every probability matched the frozen OOF
evidence within `1e-6`; the ensemble Macro-F1 is
`0.8274221017` versus the official `0.8274221017`.

- Records: 15378
- Parameters: 99,443
- At-risk F1: 0.7840139010
- PR-AUC: 0.8935499908
- Brier: 0.1137572993
- ECE: 0.0156731919
- Future OULAD: `LOCKED_NOT_EXECUTED`

This is exact checkpoint reproduction, not new model selection.
