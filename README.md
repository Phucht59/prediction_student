# Dự đoán nguy cơ học tập và khuyến nghị hành động

Khóa luận: **Hybrid CNN–BiLSTM** dự đoán nguy cơ học tập trên **UCI** và **OULAD**, rồi **Recommendation V** xếp hạng hành động hỗ trợ trên OULAD (20/35/50/75%).

Hai dataset dùng **cùng kiến trúc Hybrid CNN–BiLSTM**. Khác nhau chỉ ở chiều input, FIT-only preprocessing và trọng số đã học.

```text
Raw CSV
  → cutoff-safe features
  → Hybrid CNN–BiLSTM  (CNN ∥ BiLSTM + cổng 3 nhánh)
  → p, ngưỡng t, ŷ, H₂(p)
  → feasibility → Recommendation V → RECOMMEND Top-1 / HUMAN_REVIEW Top-3
```

Chương 3 (đề xuất phương pháp, số liệu khóa): `reports/prediction/final/CHUONG_3.md`.  
Báo cáo kỹ thuật pipeline: `reports/BAO_CAO_KY_THUAT_PROJECT.md`.

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

Một Hybrid CNN–BiLSTM được chấm tại `20% → 35% → 50% → 75% → 100%`. Sự kiện chỉ lấy khi `observation_start ≤ t < cutoff`. Cấm predictor: `final_result`, `score`, `target`, `date_unregistration`. **100% không dùng cho khuyến nghị.**

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

## 4. Hybrid CNN–BiLSTM (mô hình dự đoán cuối)

Tên công khai **Hybrid CNN–BiLSTM**. Một class PyTorch, hai checkpoint (UCI / OULAD). CNN và BiLSTM **song song** trên cùng chuỗi temporal, hợp với nhánh tabular bằng cổng softmax 3 nhánh (mask availability).

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

## 5. Kết quả dự đoán (robust 3×3)

Bốn chỉ số: **AP** (`average_precision_score`), **Accuracy**, **F1**, **Recall** — trung bình 3 fold × 3 seed. Không lấy run đẹp nhất. Chi tiết: `reports/prediction/final/FINAL_PREDICTION_MODEL_REPORT.md`. Chương 3: `reports/prediction/final/CHUONG_3.md`.

### UCI

| Mốc | AP | Accuracy | F1 | Recall |
|---|---:|---:|---:|---:|
| S0 | 0.4547 | 0.5213 | 0.4291 | 0.8421 |
| S1 | **0.8214** | 0.8553 | 0.6899 | 0.7587 |
| S2 | **0.9101** | **0.9094** | **0.8010** | 0.8545 |

S0 chưa có `G1`/`G2` — AP thấp, không tuyên bố thắng. S1/S2 Hybrid đứng đầu AP so với LR, DT, RF, SVM, MLP.

### OULAD

| Mốc | AP | Accuracy | F1 | Recall |
|---|---:|---:|---:|---:|
| 20% | 0.7624 | 0.6854 | 0.6781 | 0.7769 |
| 35% | **0.8058** | 0.7456 | 0.7001 | 0.7464 |
| 50% | **0.8483** | 0.8006 | 0.7306 | 0.7207 |
| 75% | **0.8885** | 0.8627 | 0.7807 | 0.7221 |
| 100% | **0.9204** | **0.9088** | **0.8372** | 0.7807 |

Hybrid đứng nhất AP từ 35% đến 100% trên roster serving. 20% thua LR 0.0008. 100% không dùng cho khuyến nghị.

---

## 6. Recommendation V

Chỉ OULAD 20/35/50/75. Hybrid CNN–BiLSTM không bị refit.

```text
Hybrid CNN–BiLSTM (p, t, H₂)
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
| Recommendation V | **0.88785** | 0.99206 | 0.99603 | 0.79947 | 0 |
| B0 | 0.81889 | 0.99365 | 0.99683 | 0.78981 | 0 |
| B1 | 0.86649 | 0.99683 | 0.99841 | 0.80357 | 0 |

NDCG@3 vs B1: +0.0213, 95% CI [0.0144, 0.0282].

---

## 7. PostgreSQL (`student_db`)

Luồng lưu trữ: `raw` (3 dataset) → `catalog` → `prediction` (Hybrid CNN–BiLSTM) → `recommendation` (Recommendation V).

```powershell
python project.py db status
python project.py db load-all
python project.py db lookup --student 631334 --course CCC --presentation 2014B --stage 20
python project.py db predict --student 631334 --course CCC --presentation 2014B --stage 20
python project.py db recommend --student 631334 --course CCC --presentation 2014B --stage 20
```

`predict` đọc xác suất đã lưu (không train lại). `recommend` chạy Recommendation V đóng băng và có thể ghi DB.

---

## 8. Chạy và kiểm tra

```powershell
python project.py prediction status
python project.py prediction validate
pytest tests/prediction tests/recommend_hybrid/v3 tests/database -q
```

| Thành phần | Path |
|---|---|
| Bản đồ project | `PROJECT.md` |
| Hybrid CNN–BiLSTM | `src/prediction/model/hybrid.py` |
| UCI / OULAD features | `src/prediction/data/uci.py`, `oulad_features.py` |
| Recommendation V | `src/recommend_hybrid/v3/` |
| Live DB runtime | `src/database/live_runtime.py` |
| Config | `configs/prediction/hybrid_final.json` |
| Chương 3 khóa luận | `reports/prediction/final/CHUONG_3.md` |
| Báo cáo kỹ thuật | `reports/BAO_CAO_KY_THUAT_PROJECT.md` |
| Robust mean dự đoán | `reports/prediction/final/FINAL_PREDICTION_MODEL_REPORT.md` |
| Lệch lớp và so sánh công bằng | `reports/prediction/final/IMBALANCE_AND_FAIRNESS.md` |
| Recommendation V final | `reports/recommend_hybrid/v3/FINAL_RECOMMENDATION_V3_REPORT.md` |

---

## 9. Authority

```text
Prediction       : Hybrid CNN–BiLSTM
Recommendation   : Recommendation V
UCI              : S0 / S1 / S2
OULAD predict    : 20 / 35 / 50 / 75 / 100
OULAD recommend  : 20 / 35 / 50 / 75
Outer test used  : false
```
