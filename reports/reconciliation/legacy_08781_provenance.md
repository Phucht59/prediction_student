# Provenance of historical 0.8781

The unique source is `artifacts/model_selection/nested-full-20260710/selection_manifest.json` and its duplicated final-evidence copies. It records `cv_f1_macro_mean = 0.8780892327767106`, `cv_f1_macro_std = 0.04482875094405285`, and fold values `[0.9074851, 0.8232240, 0.8235674, 0.9171220, 0.9190476]`.

Record-level evidence exists at `outer_oof_predictions.csv` (316 rows, SHA-256 `b6b3397df2a564607d5f6de89047e40daf84afb665deab9fcde666aa9c284a64`). The estimator is the arithmetic mean of five outer-fold Macro-F1 values under one fixed seed (42), not a best seed, ensemble, or pooled OOF Macro-F1. Pooled OOF Macro-F1 is `0.8779264`, a distinct value.

The historical score is metric-auditable because OOF predictions, fold IDs, selected config, and summary survive. It is not fully training-reproducible: no per-fold model checkpoints or epoch histories survive in the historical model-selection directory. It is historical-reference-only because its development/legacy boundary was superseded by Protocol V2, even though the actual 316-row development membership is the same.
