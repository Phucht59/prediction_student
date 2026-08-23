# Project — Hybrid CNN–BiLSTM và module khuyến nghị

Bản đồ khóa luận. Tên công khai: **Hybrid CNN–BiLSTM**, **module khuyến nghị**. Không dùng tên thí nghiệm trên tài liệu công khai.

**Bản kỹ thuật:** [`CHUONG_3.md`](reports/prediction/final/CHUONG_3.md) (thiết kế) và Chương 1–5 trong [`reports/prediction/final/`](reports/prediction/final/) (Ch1 giả thuyết H1–H3, Ch2 công thức, Ch4 thực nghiệm + XAI cổng, Ch5 hạn chế). Nhật ký: `NHAT_KY_THI_NGHIEM.ipynb`. Không xây giao diện.

Outer test không dùng. AP = `sklearn.metrics.average_precision_score`.

---

## Mô hình cuối đã khóa

| | |
|---|---|
| Dự đoán | Hybrid CNN–BiLSTM, `model_id=hybrid`, một checkpoint UCI (S0–S2), một checkpoint OULAD (20–100%) |
| Khuyến nghị | Persistence top-K, chỉ OULAD 20/35/50/75, đọc `PredictionResult` |
| Roster serving | Hybrid, LR, DT, RF, SVM, MLP, XGB |
| Lệch lớp | `pos_weight` FIT-only; SMOTE tensor thử, không chọn |

### Hybrid serving 3×3 (số khóa)

UCI AP: S0 0.455 · S1 **0.821** · S2 **0.910**  
OULAD AP: 20% 0.762 · 35% **0.806** · 50% **0.848** · 75% **0.889** · 100% **0.920**

Hybrid (mô hình chính): UCI AP S1 0.821 / S2 0.910; OULAD AP 0.762 → 0.920 theo cutoff. LR/DT/RF/SVM/MLP/XGB chỉ so sánh.

### Baseline một-trọng-số cùng tensor

Hybrid khóa: UCI S1 **0.821** / S2 **0.910**; OULAD 35–100% **0.806 → 0.920**. Family chỉ đối chiếu.

### Module khuyến nghị

Top-K theo `p`: Precision@10% 0.923–0.999. Rec F1 0.763 vs luật 0.677. Invalid 0. Không nhân quả.

Thiết kế, kiến trúc, giao thức rò rỉ: **Chương 3**. Số liệu, overfit, Panel C, luồng case phục vụ: **Chương 4**.

---

## Code

| | |
|---|---|
| Hybrid | `src/prediction/` |
| Module khuyến nghị | `src/recommend_hybrid/serving/` |
| Live DB | `src/database/live_runtime.py`, `database/live/` |
| CLI | `python project.py` |
| Config | `configs/prediction/hybrid_final.json` |
| Artifacts serving | `artifacts/prediction/final/`, `artifacts/recommend_hybrid/serving/` |

Research, HPO, test cũ: `test_lab/` — không phải bản phát hành.

---

## PostgreSQL (`student_db`)

```text
raw (3 dataset) → catalog (enrollment) → prediction (Hybrid) → recommendation (persistence top-K)
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
pytest -q tests/prediction tests/recommend_hybrid/serving tests/integration tests/database
```
