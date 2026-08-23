# Hybrid CNN–BiLSTM và module khuyến nghị

Khóa luận: **Hybrid CNN–BiLSTM** dự đoán nguy cơ học tập nhị phân trên **UCI** và **OULAD**; **module khuyến nghị** gắn nút thắt còn kéo dài trên hàng đợi top-K (OULAD 20/35/50/75%).

Hai miền dùng **cùng kiến trúc**. Khác nhau chỉ chiều input, FIT-only preprocessing và trọng số. Tên thí nghiệm không dùng trên tài liệu công khai.

Bản kỹ thuật: [`CHUONG_3.md`](reports/prediction/final/CHUONG_3.md) (phương pháp), [`CHUONG_4.md`](reports/prediction/final/CHUONG_4.md) (thực nghiệm, 18 hình). Notebook vẽ hình: [`CHUONG_4.ipynb`](reports/prediction/final/CHUONG_4.ipynb).

```text
Raw CSV
  → cutoff-safe tensors
  → Hybrid CNN–BiLSTM  (CNN ∥ BiLSTM + cổng 3 nhánh)
  → p, ngưỡng t, ŷ, H₂(p)
  → module khuyến nghị → ACTION / QUEUE / COUNSEL / OUT_OF_BUDGET
  → PostgreSQL: raw → catalog → prediction → recommendation
```

---

## Kết quả khóa (robust inner 3×3)

AP = `sklearn.metrics.average_precision_score`. Outer test **không** dùng. Một checkpoint / miền chấm mọi mốc.

### Hybrid CNN–BiLSTM — UCI (`G3 < 10`, prevalence 0.22)

| Mốc | Acc | AP | Prec | F1 | Rec |
|---|---:|---:|---:|---:|---:|
| S0 | 0.521 | 0.455 | 0.291 | 0.429 | 0.842 |
| S1 | 0.855 | **0.821** | 0.660 | 0.690 | 0.759 |
| S2 | 0.909 | **0.910** | 0.765 | 0.801 | 0.855 |

S0 chưa có điểm — CNN/BiLSTM tắt theo thiết kế. S1/S2 là claim chính: AP 0.821 / 0.910.

### Hybrid CNN–BiLSTM — OULAD (Fail \| Withdrawn)

| Mốc | Acc | AP | Prec | F1 | Rec |
|---|---:|---:|---:|---:|---:|
| 20% | 0.686 | 0.762 | 0.603 | 0.678 | 0.777 |
| 35% | 0.744 | **0.806** | 0.661 | 0.700 | 0.746 |
| 50% | 0.800 | **0.848** | 0.745 | 0.731 | 0.721 |
| 75% | 0.863 | **0.889** | 0.852 | 0.781 | 0.722 |
| 100% | 0.903 | **0.920** | 0.905 | 0.837 | 0.781 |

AP tăng theo cutoff: 0.762 (20%) → **0.920** (100%). 100% không dùng khuyến nghị.

### Đối chiếu cùng tensor (không thay Hybrid)

Cùng protocol, bộ so sánh (LR/DT/RF/SVM/MLP/XGB) không thay Hybrid. Claim chính: UCI S1 **0.821** / S2 **0.910**; OULAD 35–100% **0.806 → 0.920**.

### Module khuyến nghị (persistence, top-K)

Hàng đợi 10% theo Hybrid `p`: Precision@10% 0.923 (20%) → 0.999 (75%). Rec học nút thắt 14 ngày, macro-F1 **0.763** vs luật đuôi 0.677 trên test chia theo sinh viên. Invalid **0**. Không phải nhân quả.

---

## Chạy

```powershell
python project.py final status
python project.py db status
python project.py db load-all
python project.py db predict --student 631334 --course CCC --presentation 2014B --stage 20
python project.py db recommend --student 631334 --course CCC --presentation 2014B --stage 20
```

Kết nối PostgreSQL từ `.env` (`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`).

Chuỗi DB tối giản: `raw` → `catalog` → `prediction` → `recommendation`. Khóa train nằm ở `training.lock`. Không copy `studentVle` vào Postgres.

```powershell
pytest -q tests/prediction tests/recommend_hybrid/serving tests/integration tests/database
```

---

## Bản đồ

| | |
|---|---|
| Chương 1–5 | `reports/prediction/final/CHUONG_1.md` … `CHUONG_5.md` |
| Nhật ký thí nghiệm | `reports/prediction/final/NHAT_KY_THI_NGHIEM.ipynb` |
| Hybrid | `src/prediction/` |
| Module khuyến nghị | `src/recommend_hybrid/serving/` |
| Config | `configs/prediction/hybrid_final.json` |
| DB live | `database/live/` |
| Research / test cũ | `test_lab/` (không thuộc bản phát hành) |
