# Project — Hybrid CNN–BiLSTM và Recommendation V

Bản đồ khóa luận. Tên công khai: **Hybrid CNN–BiLSTM**, **Recommendation V**. Không dùng tên thí nghiệm trên tài liệu công khai.

**Bản kỹ thuật Chương 3 (nguồn sự thật):** [`reports/prediction/final/CHUONG_3.md`](reports/prediction/final/CHUONG_3.md)

Outer test không dùng. AP = `sklearn.metrics.average_precision_score`.

---

## Mô hình cuối đã khóa

| | |
|---|---|
| Dự đoán | Hybrid CNN–BiLSTM, `model_id=hybrid`, một checkpoint UCI (S0–S2), một checkpoint OULAD (20–100%) |
| Khuyến nghị | Recommendation V, chỉ OULAD 20/35/50/75, đọc `PredictionResult` |
| Roster serving | Hybrid, LR, DT, RF, SVM, MLP (không XGB) |
| Lệch lớp | `pos_weight` FIT-only; SMOTE tensor thử, không chọn |

### Hybrid serving 3×3 (số khóa)

UCI AP: S0 0.455 · S1 **0.821** · S2 **0.910**  
OULAD AP: 20% 0.762 · 35% **0.806** · 50% **0.848** · 75% **0.889** · 100% **0.920**

Serving: Hybrid thắng AP UCI S1–S2 và OULAD 35–100%. S0 thua RF. 20% thua LR 0.0008.

### Baseline một-trọng-số cùng tensor

UCI AP Hybrid S1 0.811 / S2 0.913 (thắng LR). S0 thua RF.  
OULAD vs XGB: 20% 0.747 vs 0.766; 35–100% hòa (\|Δ\|≤0.003). Không tuyên bố vượt trội OULAD.

### Recommendation V

Panel C: NDCG@3 0.888 vs B1 0.866; invalid 0. Không nhân quả.

Chi tiết bảng, kiến trúc, rò rỉ, overfit, luồng case: **Chương 3**.

---

## Code

| | |
|---|---|
| Hybrid | `src/prediction/` |
| Recommendation V | `src/recommend_hybrid/v3/` |
| Live DB | `src/database/live_runtime.py`, `database/live/` |
| CLI | `python project.py` |
| Config | `configs/prediction/hybrid_final.json` |
| Artifacts serving | `artifacts/prediction/final/`, `artifacts/recommend_hybrid/v3/` |

Research, HPO, test cũ: `test_lab/` — không phải bản phát hành.

---

## PostgreSQL (`student_db`)

```text
raw (3 dataset) → catalog (enrollment) → prediction (Hybrid) → recommendation (Recommendation V)
training.lock  ← khóa Hybrid + baseline một-trọng-số
```

Không schema Optuna/research trên bản cuối. Không nhét `studentVle` vào DB.

```powershell
python project.py db status
python project.py db load-all
python project.py db predict --student 631334 --course CCC --presentation 2014B --stage 20
python project.py db recommend --student 631334 --course CCC --presentation 2014B --stage 20
```

---

## Kiểm tra

```powershell
python project.py final status
pytest -q tests/prediction tests/recommend_hybrid/v3 tests/integration tests/database
```
