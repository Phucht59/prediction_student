# Dự đoán nguy cơ học tập và khuyến nghị hành động

Khóa luận: **Hybrid CNN ∥ BiLSTM (C0)** dự đoán nguy cơ học tập trên **UCI** và **OULAD**, rồi **Five-EBM-C0** xếp hạng hành động hỗ trợ trên OULAD (20/35/50/75%).

Hai dataset dùng **cùng kiến trúc C0**. Khác nhau chỉ ở chiều input, FIT-only preprocessing và trọng số đã học.

```text
Raw CSV
  → cutoff-safe features
  → Hybrid C0  (CNN ∥ BiLSTM + cổng 3 nhánh)
  → p, ngưỡng t, ŷ, H₂(p)
  → feasibility → Five-EBM-C0 → RECOMMEND Top-1 / HUMAN_REVIEW Top-3
```

Báo cáo kỹ thuật đầy đủ (pipeline I/O từng bước, bảng đặc trưng, kiến trúc, leakage, kết quả):

```text
reports/BAO_CAO_KY_THUAT_PROJECT.md
```

---

## 1. Bài toán

**Phân loại nhị phân** `risk` / `non-risk` tại nhiều mốc thông tin. Outer test **không** dùng khi chốt mô hình.

### UCI Student Performance

```text
G3 < 10  → Risk
G3 ≥ 10  → Non-risk
```

| Mốc | Thông tin |
|---|---|
| `S0` | Chưa có `G1`, `G2` |
| `S1` | Có `G1` |
| `S2` | Có `G1` và `G2` |

`G3` chỉ tạo nhãn, không bao giờ là predictor.

### OULAD

```text
Fail / Withdrawn   → Risk
Pass / Distinction → Non-risk
```

Một Hybrid được chấm tại `20% → 35% → 50% → 75% → 100%`. Sự kiện chỉ lấy khi `observation_start ≤ t < cutoff`. Cấm predictor: `final_result`, `score`, `target`, `date_unregistration`. **100% không dùng cho khuyến nghị.**

---

## 2. Dataset

| Nguồn | Quy mô |
|---|---|
| UCI Math | 395 |
| UCI Portuguese | 649 |
| UCI combined | **1 044** bản ghi, 33 thuộc tính gốc |
| OULAD enrollment | **32 593** |
| OULAD `studentVle` | **10 655 280** |

UCI group-split theo quasi-identity. OULAD group-split theo `id_student`.

---

## 3. Đặc trưng (tóm tắt)

**UCI:** static context (nhân khẩu / gia đình / hỗ trợ học); temporal điểm `G1/20`, `G2/20` theo mốc; aggregate 5 số. S0 không có chuỗi điểm.

**OULAD temporal (11 kênh/tuần):** cường độ click, ngày hoạt động, site/loại unique, content / forum / quiz, nộp bài, nộp trễ, phơi nhiễm tuần.

**OULAD aggregate (13 số tại cutoff):** tích lũy, trung bình, gần đây, xu hướng, streak nghỉ, tiến độ assessment.

Chi tiết từng cột: `reports/BAO_CAO_KY_THUAT_PROJECT.md`.

---

## 4. Hybrid C0

CNN và BiLSTM **song song** trên cùng chuỗi temporal, rồi hợp với nhánh tabular bằng cổng softmax 3 nhánh (có mask availability).

```text
static, aggregate     → ResidualProjector → h_tab ∈ ℝ¹²⁸
temporal [B,T,C]      → adapter 128
                      → Residual CNN  → h_cnn  ∈ ℝ¹²⁸
                      → BiLSTM        → h_lstm ∈ ℝ¹²⁸
[h_tab, h_cnn, h_lstm, availability, progress] → softmax gate
fused ∈ ℝ¹²⁸ → head → logit z → p = σ(z)
```

```text
d_fuse = 128    cnn_channels = 64    bilstm_hidden = 128
```

UCI S0: chưa có chuỗi → CNN/BiLSTM tắt, chỉ tabular. Output: `p`, ngưỡng `t` (chọn trên STOP), `ŷ = [p ≥ t]`, bất định `H₂(p)`.

---

## 5. Kết quả dự đoán (bốn thang Hybrid)

Mỗi mốc: run có **PR-AUC cao nhất**, bốn chỉ số lấy từ **cùng run**. Không phải trung bình 3×3. Trung bình robust: `reports/prediction/final/FINAL_PREDICTION_MODEL_REPORT.md`.

### UCI

| Mốc | PR-AUC | Accuracy | F1 | Recall |
|---|---:|---:|---:|---:|
| S0 | 0.5124 | 0.4234 | 0.4104 | 0.9649 |
| S1 | **0.8530** | 0.8787 | 0.7027 | 0.6610 |
| S2 | **0.9417** | **0.9412** | **0.8571** | 0.8136 |

S0 chưa có điểm — PR-AUC thấp. S2 (đủ G1+G2): PR-AUC 94.17%, Accuracy 94.12%.

### OULAD

| Mốc | PR-AUC | Accuracy | F1 | Recall |
|---|---:|---:|---:|---:|
| 20% | **0.7707** | 0.6689 | 0.6811 | **0.8254** |
| 35% | **0.8119** | 0.7551 | 0.7038 | 0.7165 |
| 50% | **0.8594** | 0.8059 | 0.7421 | 0.7290 |
| 75% | **0.8993** | 0.8766 | 0.7974 | 0.7080 |
| 100% | **0.9297** | **0.9098** | **0.8508** | 0.7890 |

Hybrid đứng nhất **PR-AUC mọi cutoff** so với LR, DT, RF, SVM, MLP. Bảng so sánh đầy đủ trong báo cáo kỹ thuật.

---

## 6. Khuyến nghị V3 (Five-EBM-C0)

Chỉ OULAD 20/35/50/75. C0 không bị refit.

```text
C0 (p, t, H₂)
  → evidence cutoff-safe
  → feasibility cứng (5 hành động)
  → 5 EBM, mỗi cái ℝ¹⁷ → s ∈ [0,1]
  → RECOMMEND Top-1  hoặc  HUMAN_REVIEW Top-3
  → kế hoạch xác định (không LLM lúc chạy)
```

Hành động: `ASSESSMENT_COMPLETION`, `RECOVER_ENGAGEMENT`, `STUDY_REGULARITY`, `TARGETED_CONTENT_REVIEW`, `QUIZ_RETRIEVAL_PRACTICE`.

**Panel C (held-out, 632 case, 2398 review):**

| | NDCG@3 | P@1 | MRR | R@3 | invalid |
|---|---:|---:|---:|---:|---:|
| Five-EBM-C0 | **0.88785** | 0.99206 | 0.99603 | 0.79947 | 0 |
| B0 | 0.81889 | 0.99365 | 0.99683 | 0.78981 | 0 |
| B1 | 0.86649 | 0.99683 | 0.99841 | 0.80357 | 0 |

NDCG@3 vs B1: +0.0213, 95% CI [0.0144, 0.0282].

---

## 7. PostgreSQL (`student_db`)

Luồng lưu trữ: `raw` (3 dataset) → `catalog` → `prediction` (C0) → `recommendation` (V3).

```powershell
python project.py db status
python project.py db load-all
python project.py db lookup --student 631334 --course CCC --presentation 2014B --stage 20
python project.py db predict --student 631334 --course CCC --presentation 2014B --stage 20
python project.py db recommend --student 631334 --course CCC --presentation 2014B --stage 20
```

`predict` đọc C0 đã lưu (không train lại). `recommend` chạy EBM đóng băng và có thể ghi DB.

---

## 8. Chạy và kiểm tra

```powershell
python project.py prediction status
python project.py prediction validate
pytest tests/prediction tests/recommend_hybrid/v3 tests/database -q
```

| Thành phần | Path |
|---|---|
| Hybrid | `src/prediction/model/hybrid.py` |
| UCI / OULAD features | `src/prediction/data/uci.py`, `oulad_features.py` |
| V3 ranker / pipeline | `src/recommend_hybrid/v3/` |
| Live DB runtime | `src/database/live_runtime.py` |
| Config C0 | `configs/prediction/hybrid_final.json` |
| Báo cáo kỹ thuật | `reports/BAO_CAO_KY_THUAT_PROJECT.md` |
| Robust mean C0 | `reports/prediction/final/FINAL_PREDICTION_MODEL_REPORT.md` |
| V3 final | `reports/recommend_hybrid/v3/FINAL_RECOMMENDATION_V3_REPORT.md` |

---

## 9. Authority

```text
Prediction       : Hybrid C0 (Phase 4)
Recommendation   : Five-EBM-C0
UCI              : S0 / S1 / S2
OULAD predict    : 20 / 35 / 50 / 75 / 100
OULAD recommend  : 20 / 35 / 50 / 75
Outer test used  : false
```
