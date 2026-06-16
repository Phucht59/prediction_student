# Xay dung mo hinh hoc ket hop de du doan thanh tich hoc tap sinh vien

## Mo ta ngan

Du an xay dung pipeline du doan thanh tich hoc tap sinh vien theo ba muc bang mo hinh Deep Learning CNN-BiLSTM. Baseline models duoc dung de doi chung ky thuat, trong khi mo hinh chinh cua de tai la CNN-BiLSTM theo dung de cuong.

## Dataset

| Dataset | Vai tro |
| --- | --- |
| `student-mat` | Du lieu Student Performance mon Toan |
| `student-por` | Du lieu Student Performance mon Tieng Bo Dao Nha |
| `xAPI` | Du lieu hanh vi hoc tap va tuong tac hoc truc tuyen |

Du an khong su dung `student-combine` lam dataset chinh.

## Mo hinh chinh

- Student datasets: `sequence_cnn_bilstm_only`.
- xAPI: `gated_fusion_v28`, ket hop sequence branch va context branch bang gated fusion.
- Sequence branch dung CNN de trich xuat dac trung cuc bo, sau do BiLSTM hoc bieu dien chuoi.

## Xu ly mat can bang

Pipeline thuc nghiem da kiem tra cac chien luoc hop le nhu class weight, SMOTENC, random oversampling va focal loss. Khong dung ADASYN truc tiep tren du lieu Student co bien phan loai da label encoding.

## Final results

| Dataset | Scenario | Model | Prediction mode | Macro F1 | Recall Low | F1 Low |
| --- | --- | --- | --- | ---: | ---: | ---: |
| student-mat | late | sequence_cnn_bilstm_only | low_f1_tuned | 0.9365 | 0.9615 | 0.8929 |
| student-por | late | sequence_cnn_bilstm_only | low_f1_tuned | 0.8783 | 0.9000 | 0.8182 |
| student-por | midterm | sequence_cnn_bilstm_only | argmax | 0.8228 | 0.6500 | 0.7429 |
| xAPI | xapi | gated_fusion_v28 | low_f1_tuned | 0.7541 | 0.8846 | 0.8214 |

## Cach chay pipeline final

Chay test co ban:

```powershell
py -3.10 -m pytest -q
```

Xem ket qua final:

```powershell
Get-Content reports\final\final_prediction_model_report.md
Import-Csv reports\final\final_deep_results_table.csv | Format-Table -AutoSize
```

Neu can tai lap cac thuc nghiem ky thuat, xem cac script trong `scripts/` va cac module trong `src/`. Cac thuc nghiem V28-V30 da duoc archive de tranh nham voi artifact final.

## Bao cao

- `reports/final/final_model_manifest.json`
- `reports/final/final_deep_results_table.csv`
- `reports/final/final_baseline_comparison.csv`
- `reports/final/final_prediction_model_report.md`
- `reports/final/final_thesis_ready_summary.md`

## Ghi chu hoc thuat

- Locked test chi duoc dung cho final evaluation.
- Threshold tuning cho lop Low dung CV/OOF train-pool probabilities, khong dung locked test.
- Baseline chi dung de doi chung, khong phai mo hinh chinh cua de tai.
- Regression head khong duoc claim la ket qua chinh vi RMSE con cao.
- Khong su dung optimistic/paper-like result selection.
