# Experiment protocol

## Frozen outer evaluation

- Student-Mat và Student-Por: 5 outer folds, 3 inner folds.
- OULAD: 3 grouped outer folds theo `id_student`.
- Preprocessing, imputation, encoding và scaling chỉ fit trên training
  partition hiện hành.
- Inner validation chọn hyperparameter, epoch và threshold.
- Outer rows không được dùng cho selection.
- Five fixed seeds: 42, 1201, 2026, 3407, 7319.
- Probability ensemble là mean qua toàn bộ seed; không chọn best seed.

## Unified stage evaluation

Một training run tạo một checkpoint phục vụ tất cả stage của cùng
dataset/fold/seed:

- UCI: S0 không G1/G2, S1 chỉ G1, S2 có G1+G2.
- OULAD: E1 20%, E2 35%, M1 50%, L1 75%.

Availability mask/cutoff bảo đảm stage trước không đọc thông tin stage sau.
G3 chỉ tạo target và không phải predictor.

## Comparators

Mười model family dùng chung outer partitions: Logistic Regression, Decision
Tree, Random Forest, HistGradientBoosting, SVM, XGBoost, MLP, CNN-only,
BiLSTM-only và CNN-BiLSTM. CNN-only/BiLSTM-only là ablation; model hybrid chính
vẫn là CNN-BiLSTM.

Không dùng plain SMOTE/ADASYN trên mixed categorical label-coded UCI và không
synthetic oversample raw OULAD tensor.

## Metrics

UCI báo Accuracy, Balanced Accuracy, Macro/Weighted-F1, per-class
Precision/Recall/F1, PR-AUC, ROC-AUC, Brier, NLL, ECE và confusion matrix.
OULAD báo thêm Risk Precision/Recall/F1. Paired bootstrap có 5,000 replicate;
OULAD resample theo student.

## Freeze boundary

Validation chỉ replay evidence. Không train lại official model, không mở Future
OULAD, không tune bằng outer data và không thay recommendation semantics.

Machine-readable contracts:

- `configs/final/uci_prediction.yaml`
- `configs/final/oulad_prediction.yaml`
- `configs/final/recommendation.yaml`
