# Optimization Protocol Audit

- **Feature engineering fit state**: `PASS` - Feature engineering is stateless; preprocessing fit is scoped to fold-training only.
- **Imputation/encoding/scaling**: `PASS` - `DataPreprocessor.fit_transform()` is called only on model-train partitions; validation/scoring folds use `transform()`.
- **Feature selection**: `PASS` - `FeatureSelector.fit_transform()` is called only on model-train partitions.
- **SMOTE/ADASYN**: `PASS` - Oversampling is only inside `fit_transform(..., apply_oversampling=True)` for gradient-training rows.
- **Class weights**: `PASS` - Class weights use `model_train_fold` labels only, not early-stop or scoring folds.
- **Early stopping**: `PASS` - Early stopping uses a split carved from fold-training data; outer validation is only scored.
- **Outer validation role**: `PASS` - Outer folds are evaluated after inner CV freezes params/strategy/calibration/threshold.
- **OOF row coverage**: `PASS` - Outer StratifiedKFold assigns each train source row to exactly one validation fold.
- **Ensemble/threshold/calibration**: `PASS` - Strategy, weights, temperature and thresholds are fit on inner OOF only, then frozen for outer evaluation.
- **Metadata leakage**: `PASS` - `__source_row_number`, DB IDs and `G3_raw` are excluded from model inputs by preprocessing/dataset code.

Locked test is not passed to Optuna, nested CV, threshold fitting or calibration fitting.
