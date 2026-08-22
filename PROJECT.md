# PROJECT — Hybrid CNN–BiLSTM and Recommendation V

Bản đồ hiện tại của khóa luận. Không dùng tên C0 / V2 / V3 trên tài liệu công khai.

Lịch sử multiclass / `cnn_bilstm_*` / H1: `reports/prediction/historical/pre_phase4/PROJECT_LEGACY.md`.

## Hệ thống

```text
Raw CSV
  → cutoff-safe features
  → Hybrid CNN–BiLSTM
  → p, ngưỡng t, ŷ, H₂(p)
  → Recommendation V
  → RECOMMEND Top-1 / HUMAN_REVIEW Top-3
```

| Module | Việc | Phạm vi |
|---|---|---|
| **Hybrid CNN–BiLSTM** | Dự đoán nguy cơ nhị phân | UCI S0–S2; OULAD 20–100% |
| **Recommendation V** | Xếp hạng 5 hành động hỗ trợ | OULAD 20/35/50/75% only |
| **PostgreSQL `student_db`** | Lưu raw → catalog → prediction → recommendation | Serve, không refit |

Báo cáo kỹ thuật (I/O từng bước, đặc trưng, kiến trúc, kết quả): `reports/BAO_CAO_KY_THUAT_PROJECT.md`.  
Landing page: `README.md`.

## Prediction

| Dataset | Target | Mốc của **một** fitted model |
|---|---|---|
| UCI | `G3 < 10` | S0 → S1 → S2 |
| OULAD | Fail / Withdrawn | 20% → 35% → 50% → 75% → 100% |

- Public name: **Hybrid CNN–BiLSTM** (`model_id = hybrid`, class `Hybrid`)
- Topology: CNN ∥ BiLSTM song song + cổng softmax 3 nhánh + một logit nhị phân
- Shared widths: `d_fuse=128`, `cnn_channels=64`, `bilstm_hidden=128`
- Cùng kiến trúc cho UCI và OULAD. Không mô hình riêng 100%.

Baseline đang dùng: `LR / DT / RF / SVM / MLP`. XGBoost chỉ còn historical.

Đánh giá: robust inner 3×3. Outer **không** mở. Gate nghiên cứu từng ghi `NOT_READY_FOR_FINAL_EVAL` (UCI S0 vs RF); owner chọn Hybrid CNN–BiLSTM làm authority.

Giới hạn: UCI S0 chưa có G1/G2; OULAD 100% độ dài lịch sử liên hệ Withdrawn.

## Recommendation V

Chỉ OULAD 20/35/50/75. Không refit Hybrid. Năm EBM, feasibility cứng, RECOMMEND Top-1 hoặc HUMAN_REVIEW Top-3.

Panel C held-out: NDCG@3 **0.88785**, invalid-action **0**.

## PostgreSQL

Schema live: `raw` (student_mat, student_por, oulad) → `catalog` → `prediction` → `recommendation`.

```powershell
python project.py db status
python project.py db lookup --student 631334 --course CCC --presentation 2014B --stage 20
python project.py db predict --student 631334 --course CCC --presentation 2014B --stage 20
python project.py db recommend --student 631334 --course CCC --presentation 2014B --stage 20
```

## Đọc ở đâu

| Item | Path |
|---|---|
| README | `README.md` |
| Báo cáo kỹ thuật | `reports/BAO_CAO_KY_THUAT_PROJECT.md` |
| Chương 3 | `reports/prediction/final/CHUONG_3.md` |
| Robust mean dự đoán | `reports/prediction/final/FINAL_PREDICTION_MODEL_REPORT.md` |
| Bảng metric Hybrid CNN–BiLSTM | `reports/prediction/final/uci_final.csv`, `oulad_final.csv` |
| Recommendation V | `reports/recommend_hybrid/v3/FINAL_RECOMMENDATION_V3_REPORT.md` |
| Live DB | `database/live/README.md` |
| Code dự đoán | `src/prediction/` |
| Code khuyến nghị | `src/recommend_hybrid/v3/` |
| Current vs historical | `reports/CURRENT_REPORTS.md` |
