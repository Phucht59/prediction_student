# Bao cao cap nhat CNN-BiLSTM-XAPI ngay 2026-05-30

## 1. Muc tieu cap nhat

Da thay the mo hinh xAPI cu `HybridXAPI` bang mo hinh `cnn_bilstm_xapi` / `CLS-XAPI` theo huong mot input duy nhat. Toan bo dac trung xAPI sau preprocessing va feature selection duoc dua vao chung mot tensor, khong con tach 4 hanh vi thanh sequence branch va cac dac trung con lai thanh static branch.

Pipeline model hien tai:

`Conv1D -> MaxPooling1D -> Conv1D -> MaxPooling1D -> Reshape -> BiLSTM -> Dropout -> Dense -> Softmax`

Trong code PyTorch, model tra ve logits de dung voi `CrossEntropyLoss`; softmax duoc ap dung khi predict/evaluate.

## 2. Cac file da cap nhat

- `src/models/deep_learning.py`: bo `HybridXAPI`, them `CNNBiLSTMXAPI` va alias class `CLSXAPI`.
- `src/train/train_xapi_deep.py`: doi model mac dinh thanh `cnn_bilstm_xapi`, dung mot input `X_train/X_val/X_test`, cap nhat log/config/prediction theo `input_features`.
- `src/train/train_xapi_cv.py`: doi default CV sang `cnn_bilstm_xapi`.
- `src/train/train_deep.py`: wrapper `--dataset xapi --model auto` tro sang `cnn_bilstm_xapi`.
- `scripts/run_train_deep.py`, `config.yaml`, `README.md`: cap nhat ten model xAPI.
- `src/evaluation/summarize_xapi_results.py`, `src/evaluation/summarize_final_imbalance_deep.py`: tong hop ket qua theo `cnn_bilstm_xapi`.

## 3. Du lieu va tien xu ly

- Dataset: xAPI-Edu-Data.
- So mau: 480.
- Chia train/validation/test: 336/72/72.
- Raw features: 16.
- Processed features sau one-hot/scaling: 72.
- Feature selection: `pearson_chi2`, giu toi da 56 features.
- Target: `Class`, 3 lop:
  - `L -> 0 = Low`
  - `M -> 1 = Middle`
  - `H -> 2 = High`
- Phan bo lop toan bo du lieu: Low 127, Middle 211, High 142.
- Phan bo train truoc can bang: Low 89, Middle 147, High 100.

## 4. Cau hinh train moi nhat

- Model: `cnn_bilstm_xapi`.
- Preset: `default`.
- Conv channels: 16.
- BiLSTM hidden: 16.
- So layer BiLSTM: 1.
- Dense hidden: 64.
- Dropout: 0.1.
- Optimizer: AdamW.
- Learning rate: 0.001.
- Weight decay: 0.0001.
- Epochs toi da: 100.
- Early stopping: validation Macro-F1, patience 15.
- Batch size: 16.
- Seed: 42.

## 5. Ket qua xAPI moi nhat

So lieu lay tu `reports/results/xapi_deep_results.csv`, split `test`, `split_mode=processed`, `model_preset=default`.

| Strategy | Loss weight | Effective | Best epoch | Val Macro-F1 | Test Accuracy | Test Precision Macro | Test Recall Macro | Test Macro-F1 | Test Recall Low | Test F1 Low | PR-AUC Macro |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | none | none | 21 | 0.7755 | 0.7639 | 0.7644 | 0.7814 | 0.7708 | 0.8947 | 0.8293 | 0.7842 |
| none | balanced | none | 9 | 0.7679 | 0.6667 | 0.6884 | 0.7173 | 0.6731 | 1.0000 | 0.7308 | 0.7310 |
| smote | none | smote | 13 | 0.7641 | 0.6667 | 0.6730 | 0.7156 | 0.6708 | 0.9474 | 0.7826 | 0.7629 |
| adasyn | none | adasyn | 9 | 0.6934 | 0.5972 | 0.6070 | 0.6618 | 0.5948 | 0.8947 | 0.7391 | 0.7273 |

## 6. Nhan xet

- Mo hinh moi dung dung yeu cau mot input cho xAPI; khong con duong tach behavior sequence va static features.
- Voi split hien tai, `cnn_bilstm_xapi + none` dat ket qua cao nhat: Test Macro-F1 = 0.7708, Accuracy = 0.7639.
- SMOTE va class weight tang Recall Low, nhung Macro-F1 test thap hon cau hinh khong oversampling.
- ADASYN da duoc chay lai thanh cong, nhung tren cau hinh moi khong tot bang SMOTE/none.
- Ket qua baseline Random Forest truoc do van cao hon deep xAPI moi: Test Macro-F1 = 0.8235 voi `borderline_smote`.

## 7. Artifact moi

- Ket qua raw: `reports/results/xapi_deep_results.csv`
- Tong hop xAPI: `reports/tables/xapi_deep_summary.csv`
- Bang so sanh final: `reports/tables/final_imbalance_deep_comparison.csv`
- Report xAPI tu script tong hop: `report_temp/xapi_experiment_report.md`
- Model da luu: `models/saved/xapi_deep/processed/default/cnn_bilstm_xapi/`
