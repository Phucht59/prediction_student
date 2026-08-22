# Hybrid CNN–BiLSTM và Recommendation V

Khóa luận: **Hybrid CNN–BiLSTM** dự đoán nguy cơ học tập nhị phân trên **UCI** và **OULAD**; **Recommendation V** xếp hành động hỗ trợ trên OULAD (20/35/50/75%).

Hai miền dùng **cùng kiến trúc**. Khác nhau chỉ chiều input, FIT-only preprocessing và trọng số. Tên thí nghiệm không dùng trên tài liệu công khai.

Bản kỹ thuật đầy đủ: [`reports/prediction/final/CHUONG_3.md`](reports/prediction/final/CHUONG_3.md).

```text
Raw CSV
  → cutoff-safe tensors
  → Hybrid CNN–BiLSTM  (CNN ∥ BiLSTM + cổng 3 nhánh)
  → p, ngưỡng t, ŷ, H₂(p)
  → Recommendation V → RECOMMEND Top-1 / HUMAN_REVIEW Top-3
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

S0 chưa có điểm — CNN/BiLSTM tắt. S1/S2 Hybrid đứng đầu AP trên roster phục vụ (LR, DT, RF, SVM, MLP).

### Hybrid CNN–BiLSTM — OULAD (Fail \| Withdrawn)

| Mốc | Acc | AP | Prec | F1 | Rec |
|---|---:|---:|---:|---:|---:|
| 20% | 0.686 | 0.762 | 0.603 | 0.678 | 0.777 |
| 35% | 0.744 | **0.806** | 0.661 | 0.700 | 0.746 |
| 50% | 0.800 | **0.848** | 0.745 | 0.731 | 0.721 |
| 75% | 0.863 | **0.889** | 0.852 | 0.781 | 0.722 |
| 100% | 0.903 | **0.920** | 0.905 | 0.837 | 0.781 |

Hybrid đứng đầu AP serving từ 35% đến 100%; 20% thua LR 0.0008. 100% không dùng khuyến nghị.

### Baseline một-trọng-số trên cùng tensor Hybrid

Cùng đặc trưng với Hybrid, không summary last/mean/max. XGB không phục vụ.

- UCI: Hybrid AP S1 **0.811** / S2 **0.913** (thắng LR). S0 thua RF.
- OULAD vs XGB: 20% thua 0.019; 35–100% hòa (\|Δ AP\| ≤ 0.003).

Không tuyên bố vượt trội kiến trúc trên OULAD với trần này. Mô hình đưa ra vẫn Hybrid CNN–BiLSTM.

### Recommendation V (Panel C, 632 case)

NDCG@3 **0.888** vs B1 0.866 (Δ +0.021, 95% CI [0.014, 0.028]). P@1 0.992. Invalid-action **0**. RECOMMEND 94 / HUMAN_REVIEW 175 / INSUFFICIENT_EVIDENCE 363. Không phải nhân quả.

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
pytest -q tests/prediction tests/recommend_hybrid/v3 tests/integration tests/database
```

---

## Bản đồ

| | |
|---|---|
| Chương 3 | `reports/prediction/final/CHUONG_3.md` |
| Hybrid | `src/prediction/` |
| Recommendation V | `src/recommend_hybrid/v3/` |
| Config | `configs/prediction/hybrid_final.json` |
| DB live | `database/live/` |
| Research / test cũ | `test_lab/` (không thuộc bản phát hành) |
