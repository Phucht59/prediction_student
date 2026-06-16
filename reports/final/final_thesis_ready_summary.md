# Tom tat ket qua mo hinh du doan

Mo hinh chinh cua de tai la CNN-BiLSTM cho bai toan du doan thanh tich hoc tap sinh vien theo ba muc. Qua cac thuc nghiem V28, V29 va V30, cac bien the mo rong khong cai thien on dinh so voi ket qua deep champion da co. Vi vay, mo hinh cuoi cung cho cac tap Student duoc chon la `sequence_cnn_bilstm_only`; rieng xAPI su dung `gated_fusion_v28` nhu mot huong mo rong cho du lieu hanh vi hoc tap.

| Dataset | Scenario | Mo hinh chon | Che do du doan | Macro F1 | Recall Low | F1 Low |
| --- | --- | --- | --- | ---: | ---: | ---: |
| student-mat | late | sequence_cnn_bilstm_only | low_f1_tuned | 0.9365 | 0.9615 | 0.8929 |
| student-por | late | sequence_cnn_bilstm_only | low_f1_tuned | 0.8783 | 0.9000 | 0.8182 |
| student-por | midterm | sequence_cnn_bilstm_only | argmax | 0.8228 | 0.6500 | 0.7429 |
| xAPI | xapi | gated_fusion_v28 | low_f1_tuned | 0.7541 | 0.8846 | 0.8214 |

Ket qua cho thay mo hinh CNN-BiLSTM dat hieu nang tot tren cac tap Student va co kha nang phat hien lop Low voi Recall cao trong cac kich ban quan trong. Voi xAPI, mo hinh deep chua vuot baseline Random Forest ve Macro F1, nhung Recall Low cao cho thay tiem nang ung dung trong bai toan can uu tien phat hien som nhom nguoi hoc co rui ro.

Toan bo qua trinh chon model duoc thuc hien dua tren CV/OOF. Locked test chi duoc su dung de danh gia cuoi cung. Threshold cho lop Low duoc tune bang xac suat OOF tren train-pool, khong tune bang locked test. De tai khong su dung `student-combine`, khong dung ADASYN truc tiep tren du lieu Student co bien phan loai da label encoding, va khong claim regression head do RMSE con cao.
