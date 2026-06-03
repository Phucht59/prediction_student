# Paper Replication Report

Generated at: 2026-06-03T20:18:10

## Scope

- Pipeline này làm lại theo hướng bài báo CNN-BiLSTM, không dùng kết quả V2/V3 cũ.
- Kết quả bên dưới là kết quả chạy thật từ project hiện tại.
- Optuna đã được dựng riêng, nhưng full sweep chưa chạy.

## Assumptions

- PDF không ghi rõ split ratio, optimizer, learning rate, class bins G3; pipeline dùng stratified 64/16/20 train/val/test, Adam lr=0.001, và G3 bins 0-4/5-8/9-12/13-16/17-20.
- PDF chỉ ghi Pearson feature selection và kết luận Student dùng G1/G2, xAPI dùng raisedhands/VisitedResources/StudentAbsenceDays; pipeline dùng đúng các feature này.
- Metric Precision/Recall/F1 trong project được tính macro và weighted; bảng chính hiển thị macro để công bằng cho multiclass.

## PostgreSQL

- Status: skipped
- Note: PostgreSQL chưa cấu hình.

## Best Test Results From This Run

| dataset | stage | model_name | accuracy | precision_macro | recall_macro | f1_macro | effective_imbalance_strategy | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| student-mat | baseline | gbm | 0.7975 | 0.7867 | 0.7477 | 0.7461 | smote |  |
| student-mat | deep | paper_cnn_bilstm | 0.7595 | 0.8029 | 0.7191 | 0.7135 |  | 18.0000 |
| student-por | baseline | random_forest | 0.7846 | 0.6267 | 0.5885 | 0.5998 | none |  |
| student-por | deep | paper_cnn_bilstm | 0.7769 | 0.6169 | 0.6043 | 0.5967 |  | 34.0000 |
| xapi | baseline | svm_poly | 0.6667 | 0.6663 | 0.7094 | 0.6728 | adasyn |  |
| xapi | deep | paper_cnn_bilstm | 0.7188 | 0.7279 | 0.7344 | 0.7303 |  | 12.0000 |

## Paper Reference Values

| dataset | paper CNN-BiLSTM accuracy | other paper metrics |
|---|---:|---|
| student-mat | 1.0000 | decision_tree_accuracy=0.9620 |
| student-por | 0.9231 | decision_tree_accuracy=0.8923 |
| xapi | 0.8438 | cnn_bilstm_precision=0.8426, cnn_bilstm_recall=0.8521, cnn_bilstm_f1=0.8447 |

## Optuna Smoke Check

- Dataset: student-mat
- Trials: 2
- Epochs per trial: 3
- Best validation Macro-F1: 0.4163
- Full Optuna sweep has not been run.

## Honest Notes

- Nếu kết quả project thấp hơn paper, report giữ nguyên số thật và không copy số từ PDF.
- XGBoost được chạy nếu package có sẵn; nếu thiếu package, dòng skip sẽ nằm trong CSV kết quả.
- Các hình training curve/confusion matrix nằm trong `reports/figures/paper_replication/`.