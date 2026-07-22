# V6 domain-generalization report

| Protocol | Holdouts | Deep Macro-F1 | Deep PR-AUC | Deep Brier | Fixed XGBoost Macro-F1 |
|---|---:|---:|---:|---:|---:|
| leave_one_presentation_out | 22 | 0.713572 | 0.849550 | 0.180417 | 0.726034 |
| leave_one_module_out | 7 | 0.707129 | 0.834984 | 0.183395 | 0.720407 |

Conclusion: **NO_GENERALIZATION_ADVANTAGE**. Each holdout uses the frozen Candidate C, seed 42,
five pretraining epochs, eight multi-task epochs, and a threshold frozen before
domain evaluation. Presentation metrics use the stricter corresponding
leave-one-module-out model, so the target presentation and its entire module are
unseen. Holdout records never enter training, threshold, epoch, loss
or architecture selection. This conclusion is separate from the standard
grouped benchmark.
