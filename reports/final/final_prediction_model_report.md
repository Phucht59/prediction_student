# Final Prediction Model Report

## Muc tieu mo hinh

Muc tieu ky thuat cua pipeline la xay dung mo hinh Deep Learning CNN-BiLSTM de du doan thanh tich hoc tap sinh vien theo bai toan phan lop ba muc. Baseline duoc dung de doi chung trong cung dataset, scenario, locked test va preprocessing protocol; baseline khong phai mo hinh chinh cua de tai.

## Dataset su dung

| dataset | scenario final | vai tro |
| --- | --- | --- |
| student-mat | late | ket qua Student chinh voi G1 va G2 |
| student-por | late | ket qua Student chinh voi G1 va G2 |
| student-por | midterm | kich ban Student ho tro voi G1, khong dung G2 |
| xAPI | xapi | mo rong cho du lieu hanh vi hoc tap |

Khong su dung `student-combine`. Locked test chi duoc dung cho danh gia cuoi cung. Threshold tuning cho lop Low duoc thuc hien bang CV/OOF train-pool probabilities, khong tune bang locked test.

## Kien truc CNN-BiLSTM

Mo hinh chinh cho Student la `sequence_cnn_bilstm_only`. Sequence branch dung CNN de trich xuat mau cuc bo tren chuoi diem theo thoi diem, sau do BiLSTM hoc bieu dien hai chieu cua chuoi ngan. Xac suat dau ra duoc danh gia bang Macro F1, Recall Low va F1 Low, trong do Recall Low duoc theo doi rieng de giam bo sot nhom sinh vien rui ro.

Voi xAPI, bien the duoc chon la `gated_fusion_v28`, ket hop sequence branch voi context branch thong qua co che gated fusion. Lua chon nay phu hop hon cho du lieu hanh vi hoc tap vi xAPI co cac bien ngu canh va hanh vi ngoai chuoi.

## Ly do chon model cho Student

`sequence_cnn_bilstm_only` duoc chon cho Student vi cac dot V28, V29 va V30 khong cai thien on dinh so voi old champion. Fusion va cac bien the lon hon khong duoc chon mac dinh khi CV/OOF khong thang sequence-only. Ket qua late cua Student-Mat va Student-Por cho thay mo hinh deep canh tranh tot voi baseline, dac biet o lop Low.

## Ly do chon model cho xAPI

`gated_fusion_v28` duoc chon cho xAPI vi day la ket qua deep tot nhat hien co tren xAPI. Mo hinh chua vuot Random Forest ve Macro F1, nhung Recall Low cao, nen duoc trinh bay nhu huong mo rong cho du lieu hanh vi hoc tap thay vi claim vuot baseline.

## Final deep results

| dataset | scenario | source | model_variant | prediction_mode | Macro F1 | Recall Low | F1 Low | thesis_usage |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| student-mat | late | old_champion | sequence_cnn_bilstm_only | low_f1_tuned | 0.9365 | 0.9615 | 0.8929 | main_student_result |
| student-por | late | old_champion | sequence_cnn_bilstm_only | low_f1_tuned | 0.8783 | 0.9000 | 0.8182 | main_student_result |
| student-por | midterm | old_champion | sequence_cnn_bilstm_only | argmax | 0.8228 | 0.6500 | 0.7429 | supporting_student_scenario |
| xAPI | xapi | v28_best_deep | gated_fusion_v28 | low_f1_tuned | 0.7541 | 0.8846 | 0.8214 | behavior_dataset_extension |

## Baseline comparison

| dataset | scenario | deep model | deep Macro F1 | baseline model | baseline Macro F1 | nhan xet |
| --- | --- | --- | ---: | --- | ---: | --- |
| student-mat | late | sequence_cnn_bilstm_only | 0.9365 | xgboost | 0.9469 | baseline cao hon Macro F1, deep giu Recall Low rat cao |
| student-por | late | sequence_cnn_bilstm_only | 0.8783 | xgboost | 0.8411 | deep cao hon baseline Macro F1 |
| student-por | midterm | sequence_cnn_bilstm_only | 0.8228 | xgboost | 0.7659 | deep cao hon baseline Macro F1 |
| xAPI | xapi | gated_fusion_v28 | 0.7541 | random_forest | 0.8465 | baseline cao hon Macro F1, deep giu Recall Low cao |

## Nhan xet ky thuat

- Deep model thang hoac canh tranh tot o Student datasets, dac biet voi `student-mat late` va `student-por late`.
- xAPI chua vuot baseline ve Macro F1, nhung Recall Low cao nen phu hop de bao cao nhu huong mo rong cho du lieu hanh vi hoc tap.
- Khong claim regression head vi RMSE con cao.
- Khong su dung optimistic/paper-like result selection.
- Khong su dung `student-combine`.
- Khong dung ADASYN truc tiep tren du lieu Student co bien phan loai da label encoding.
- Khong tune threshold bang locked test.
