# V6 minimal temporal pretraining

Status: **GATE_PASS**  
Selected: **P1_MASKED_AND_NEXT_WEEK**

P1 adds only masked-week categorical/state reconstruction and next-week state
prediction to the locked V5.1 temporal encoder. It uses outer-training fold 0,
three inner folds and seed 42; outer test and Future OULAD are not accessed.

- Mean Macro-F1 gain: +0.001911
- Mean PR-AUC gain: +0.002283
- Mean At-risk F1 gain: +0.000150
- Mean Brier change: -0.001851
- Positive Macro-F1 inner folds: 1/3
- Positive PR-AUC inner folds: 3/3
- Qualifying metric: pr_auc

The gate requires Macro-F1 or PR-AUC gain of at least `0.002`, at least two
positive folds for that qualifying metric, and the registered At-risk F1/Brier
guardrails.
