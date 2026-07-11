# Tables and figures plan

| Item | Chapter | Source artifact | Interpretation |
| --- | --- | --- | --- |
| System pipeline | 3 | architecture context | CSV boundary to evidence |
| Class distribution | 3 | dataset manifest | imbalance and support |
| Three scenarios | 3 | scenario results | information availability |
| PostgreSQL ERD | 3 | migration/schema | lineage and target separation |
| Nested CV diagram | 3 | protocol manifest | selection/test separation |
| CNN-BiLSTM diagram | 3 | selected config/model | two-step architecture |
| Selected config | 3 | selected_config.json | frozen hyperparameters |
| Baseline comparison | 4 | baseline_results.csv | G2/HGB context |
| Deep/imbalance ablation | 4 | deep_ablation_results.csv | resampling effects |
| Final confusion matrix | 4 | confusion_matrix.csv | class errors |
| PR/reliability curves | 4 | PR/reliability CSV | probability diagnostics |
| Ordinal/calibration table | 4 | JSON metrics | error severity |
| Recommendation evaluation | 4 | recommendation evaluation | structural advisory quality |
| Reproducibility/tests | 4/appendix | manifests/tests | verification |
| Limitations | 5 | limitations context | claim boundaries |
# Add to final results table

Include Accuracy 0.9114 (95% CI 0.8481–0.9620) and Macro-F1 0.9262 (95% CI
0.8704–0.9694), bootstrap 2,000 resamples, seed 42. Label baseline locked-test
tables as post-hoc comparisons and identify the separate G1/G2 HGB control.
