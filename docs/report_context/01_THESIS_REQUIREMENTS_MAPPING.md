# Requirements mapping

| Requirement | Final component | Evidence | Status | Report treatment |
| --- | --- | --- | --- | --- |
| Python | final scripts and packages | `requirements.txt`, scripts | implemented | implementation environment |
| PyTorch | CNN-BiLSTM training | `src/models`, `src/train_pipeline.py` | implemented | model chapter |
| scikit-learn | baselines/metrics/split | baseline and evidence CSV | implemented | experiment chapter |
| imbalanced-learn | SMOTE/class-weight final ablation; ADASYN supplementary numeric-input ablation | `deep_ablation_results.csv`; supplementary artifacts | ADASYN was not in final selection | not selected final policy |
| CNN and BiLSTM | sequence classifier | selected config/model code | implemented | primary research architecture |
| Optuna | nested selection | selection manifest/trials | implemented | protocol chapter |
| Dropout/early stopping | selected config | `selected_config.json` | implemented | regularization section |
| PostgreSQL | lineage schema and loader | migrations/loader and DB-first evidence | migration applied; 395 targets verified | architecture and lineage verification |
| prediction system | frozen single-seed classifier | final run manifest | implemented | results chapter |
| recommendation | rule-based advisory policy | recommendation evaluation | implemented structurally | describe as advisory policy |
| Accuracy/F1/PR | evidence metrics | final run manifest | reported | evaluation tables |
| RMSE/R2 | ordinal diagnostic values in DB metrics | database manifest | diagnostic only | do not call a regression experiment |

There is no validated continuous-G3 regression branch in the frozen scientific
evidence. RMSE and R2 may be reported only as stored ordinal-label diagnostics
where the artifact identifies them; they must not be presented as regression
performance or used to claim a regression model.
