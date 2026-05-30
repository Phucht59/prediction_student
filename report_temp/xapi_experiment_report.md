# xAPI Experiment Report

## Dataset info
- Dataset: xAPI-Edu-Data
- Rows: 480
- Scenario: xapi_behavior
- Target: Class
- Class mapping: L -> 0 = Low, M -> 1 = Middle, H -> 2 = High
- Class distribution total: {'0': 127, '1': 211, '2': 142}

## Preprocessing
- Split: train/validation/test = 70/15/15
- Stratified by target_class
- Numeric: SimpleImputer median + MinMaxScaler
- Categorical: SimpleImputer most_frequent + OneHotEncoder(handle_unknown='ignore')
- Preprocessor fit only on train, validation/test only transformed
- Processed output: C:\Huflit\kltn\data\processed\xapi\xapi_behavior
- Leakage checks: {'contains_target_in_features': False, 'passed': True}

## Baseline results
Dataset | Scenario | Model | Strategy | Val Macro-F1 | Test Macro-F1 | Test Accuracy | Test Recall weak
--- | --- | --- | --- | --- | --- | --- | ---
xapi | xapi_behavior | random_forest | borderline_smote | 0.8110 | 0.8235 | 0.8194 | 0.8947

## Deep results
Dataset | Scenario | Split mode | Preset | Early stopping | Model | Loss weight | Imbalance | Feature selection | Val Macro-F1 | Test Macro-F1 | Test Accuracy | Test Recall weak
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
xapi | xapi_behavior | processed | default | val_f1 | cnn_bilstm_xapi | none | none | pearson_chi2 | 0.7755 | 0.7708 | 0.7639 | 0.8947

## Final multi-seed results
- Not generated in the cleaned final pipeline. Run a dedicated multi-seed experiment if needed.

## Nhan xet
- Best deep model by validation Macro-F1: cnn_bilstm_xapi with loss_weight=none, imbalance=none, feature_selection=pearson_chi2.
- xAPI has no G1/G2/G3 grade-period setup, so it uses scenario xapi_behavior.
- CNN-BiLSTM-XAPI uses all processed xAPI features as one input tensor.
- xAPI is classification-only in this project; no G3 regression is created.

## Han che
- xAPI target Class is already categorical and not directly equivalent to G3-based labels.
- CNN-BiLSTM-XAPI treats the processed feature vector as an ordered pseudo-sequence.
- Traditional baselines remain comparison references; final thesis model strategy prioritizes CNN-BiLSTM-XAPI for xAPI.
