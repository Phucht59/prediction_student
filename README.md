# Dự đoán rủi ro học tập sinh viên

Repository này sử dụng authority Phase8 Hybrid đã đóng băng từ nhánh
`codex/backup-hybrid-phase8-2026-08-17` của `C:\hufit\kltn`.

## Authority đang active

- Một public model class: `src/prediction/model/hybrid.py:Hybrid`.
- Một output nhị phân: logit/xác suất rủi ro, không có head ba lớp.
- UCI dùng adapter MAT + POR kết hợp, target `G3 < 10`.
- OULAD dùng target `Fail/Withdrawn`.
- UCI `S2` và OULAD `FINAL-100` là các view chính; các stage sớm là view hỗ trợ.
- UCI và OULAD fit riêng theo instance; không có joint training.
- Baseline active: Logistic Regression, Decision Tree, Random Forest, SVM và MLP.

Code và cấu hình active bắt đầu từ:

| Thành phần | Vị trí |
|---|---|
| Model, contract, inference | `src/prediction/` |
| Registry/config | `configs/prediction/` |
| Frozen Phase8 evidence | `artifacts/prediction/` |
| Final tables | `reports/prediction/final/` |
| Migration/audit report | `reports/migration/PHASE8_PREDICTION_RESTORE.md` |
| Historical prediction evidence | `artifacts/audit/` và `reports/audit/` |

Raw UCI/OULAD files không nằm trong repository này. Vì vậy adapter kiểm tra
đúng target, mask, record identity và forbidden predictors nhưng không tự tải
dữ liệu hoặc tự tạo nhãn khi thiếu raw input.

Authority Phase8 có frozen prediction/evidence nhưng không kèm trained
checkpoint file. Loader checkpoint trong `src/prediction/training/checkpoints.py`
fail-closed; fixture checkpoint chỉ dùng cho round-trip test, không phải artifact
khoa học để suy luận kết quả.

## Kiểm chứng read-only

```powershell
python project.py prediction status
python project.py prediction registry
python project.py prediction validate
pytest tests/prediction -q
```

Các lệnh trên không train, không HPO và không chạy lại outer evaluation.

## Recommendation và database

`src/recommend_hybrid/` vẫn là tầng downstream. Adapter public hiện nhận
`PredictionResult`; ranking, action policy, EBM và safety router không bị thiết
kế lại trong migration này. Audit hiện hành đánh dấu recommendation là
cần regenerate prediction-derived features và revalidate trước khi gọi là
release hợp lệ.

Database và các artifact lịch sử được giữ nguyên để truy xuất provenance. Chúng
không phải authority dự đoán active sau migration.

## Provenance

- `artifacts/migration/PREDICTION_PHASE8_MIGRATION_MANIFEST.csv`
- `artifacts/migration/MODEL_EQUIVALENCE.json`
- `artifacts/migration/DATA_EQUIVALENCE.json`
- `artifacts/migration/ACTIVE_IMPORT_BOUNDARY_AUDIT.json`
- `artifacts/audit/RECOMMENDATION_PREDICTION_DEPENDENCY_AUDIT.json`

Migration đã xác nhận model source và adapter dữ liệu tương đương định lượng
với authority Phase8 trong fixture deterministic; `C:\hufit\kltn` không bị sửa.
