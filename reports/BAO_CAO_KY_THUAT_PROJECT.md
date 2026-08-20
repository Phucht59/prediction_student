# Báo cáo kỹ thuật hệ thống

**Dự đoán nguy cơ học tập (Hybrid CNN ∥ BiLSTM C0) và khuyến nghị hành động (Five-EBM-C0)**

Tài liệu này mô tả pipeline đầy đủ của project: từ raw data đến xác suất nguy cơ, rồi đến gói khuyến nghị. Mỗi bước ghi rõ **input**, **xử lý**, **output**. Số liệu kết quả lấy **run tốt nhất theo từng mốc thời gian** (cùng một kiến trúc C0, cùng protocol FIT/STOP/VALID). Bốn thang đo trình bày là **PR-AUC, Accuracy, F1, Recall** — bốn chỉ số Hybrid đạt mức cao nhất trên các mốc có tín hiệu học tập.

Authority: Hybrid C0 (Phase 4) → Recommendation V3 Five-EBM-C0. Outer test không dùng cho chốt mô hình.

---

## 1. Toàn cảnh pipeline

Hệ thống có hai module nối tiếp, không dùng kết quả tương lai:

```text
RAW CSV
  → tiền xử lý cutoff-safe + FIT-only scale
  → tensor Hybrid (static, temporal, aggregate, mask, progress)
  → Hybrid C0: CNN ∥ BiLSTM + cổng 3 nhánh
  → output dự đoán: p, ngưỡng t, ŷ, H2(p)
  → evidence khuyến nghị (cutoff-safe)
  → feasibility cứng (5 hành động)
  → Five-EBM-C0 xếp hạng
  → router an toàn
  → RECOMMEND Top-1  hoặc  HUMAN_REVIEW Top-3  + kế hoạch xác định
```

| Bước | Input | Output |
|---|---|---|
| 1. Nạp raw | File CSV UCI / OULAD | Bảng gốc (thuộc tính thô) |
| 2. Nhãn | G3 (UCI) hoặc `final_result` (OULAD) | `target ∈ {0,1}` — **không** đưa vào predictor |
| 3. Cắt thông tin | Mốc S0/S1/S2 hoặc 20/35/50/75/100% | Chỉ sự kiện `t < cutoff` |
| 4. Đặc trưng | Bảng đã cắt | Vector tĩnh, chuỗi tuần, vector tổng hợp |
| 5. Scale | FIT split | Tensor đã chuẩn hóa, mask, `progress` |
| 6. Hybrid | 6 tensor đầu vào | Logit → `p = σ(z)` |
| 7. Ngưỡng | `p` trên STOP | `t`, `ŷ = [p ≥ t]`, `H2(p)` |
| 8. Khuyến nghị | `(p, t, H2)` + evidence | Điểm 5 hành động, route, kế hoạch |

---

## 2. Dataset

### 2.1 UCI Student Performance (kết hợp Math + Portuguese)

| Mục | Giá trị |
|---|---|
| Nguồn | `student-mat.csv` (395) + `student-por.csv` (649) |
| Bản ghi | 1 044 (một dòng = một học viên–môn) |
| Thuộc tính gốc | 33 cột + `subject` (math / portuguese) |
| Nhãn | `risk = 1` nếu `G3 < 10`, ngược lại 0 |
| Cấm làm predictor | `G1`, `G2`, `G3`, `absences` |

**33 thuộc tính gốc:** `school, sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob, reason, guardian, traveltime, studytime, failures, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic, famrel, freetime, goout, Dalc, Walc, health, absences, G1, G2, G3`.

Nhóm định danh (không dùng làm nhãn, dùng để nhóm split): `school, sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob, reason, nursery, internet`.

### 2.2 OULAD (Open University Learning Analytics)

| File | Vai trò | Số dòng (đã nạp) |
|---|---|---:|
| `courses.csv` | Khóa–kỳ, độ dài | 22 |
| `studentInfo.csv` | Hồ sơ đăng ký | 32 593 |
| `studentRegistration.csv` | Ngày đăng ký / hủy | 32 593 |
| `assessments.csv` | Định nghĩa bài KT | 206 |
| `studentAssessment.csv` | Nộp bài | 173 912 |
| `vle.csv` | Site VLE | 6 364 |
| `studentVle.csv` | Click theo ngày–site | 10 655 280 |

Đơn vị phân tích: **một enrollment** `(id_student, code_module, code_presentation)` — 32 593 enrollment, khoảng 28 785 sinh viên.

**Thuộc tính tĩnh gốc (studentInfo + registration + course):** `gender, region, highest_education, imd_band, age_band, num_of_prev_attempts, studied_credits, disability, code_module, code_presentation, date_registration, date_unregistration, module_presentation_length, final_result`.

Nhãn: `risk = 1` nếu `final_result ∈ {Fail, Withdrawn}`.  
**Cấm predictor:** `final_result`, `target`, `score`, `date_unregistration` (`date_unregistration` chỉ dùng để xác định còn trong risk-set tại cutoff).

---

## 3. Tiền xử lý dữ liệu

### 3.1 UCI

1. Ghép MAT + POR, thêm `subject`.
2. Tạo `target` từ G3; G3 không vào tensor.
3. Mốc thông tin:
   - **S0:** chưa có điểm — temporal rỗng.
   - **S1:** chỉ G1 (`G1/20`), chưa G2.
   - **S2:** G1 rồi G2 (`G2/20`).
4. Context (demographics / hành vi khai báo): median impute + StandardScaler (số); `"Unknown"` + OneHot (category). **Fit chỉ trên FIT**, transform STOP/VALID.
5. Điểm G1/G2 chia 20 (về `[0,1]`), không scale bằng thống kê tập valid.

### 3.2 OULAD — cutoff-safe

Với mỗi mốc `f ∈ {0.20, 0.35, 0.50, 0.75, 1.00}`:

- `cutoff_day = floor(module_presentation_length × f)`.
- Risk-set: đã đăng ký không sau cutoff, và chưa unregister trước/bằng cutoff.
- Sự kiện VLE / nộp bài: `observation_start ≤ t < cutoff` (không lấy ngày cutoff, không lấy tương lai).
- Biến đổi **D3_both_safe:** chia cường độ click theo số ngày phơi nhiễm trong tuần, `log1p` các kênh đếm; không pad tuần sau biên unregistration ở 100%.

### 3.3 Chia tập (chống leakage nhóm)

| Tập | Vai trò |
|---|---|
| **FIT** | Học trọng số, fit scaler/one-hot, ước lượng prior lớp |
| **STOP** | Early-stop theo macro PR-AUC; chọn ngưỡng `t` (F1 rồi recall rồi `|t−0.5|`) |
| **VALID** | Báo cáo metric — không fit, không chọn epoch, không chọn `t` |

Split **theo group**: UCI = chữ ký nhân khẩu học; OULAD = `id_student`. FIT ∩ STOP ∩ VALID = ∅. Outer test **không** dùng khi chốt C0.

Scaler temporal: `MaskedStandardScaler` — mean/std chỉ trên ô `mask=True` của FIT.

---

## 4. Xây dựng đặc trưng

### 4.1 UCI — bảng đặc trưng đưa vào Hybrid

| Nhóm | Tên | Ý nghĩa | S0 | S1 | S2 |
|---|---|---|---|---|---|
| Tĩnh (context) | 12 số: `age, Medu, Fedu, traveltime, studytime, failures, famrel, freetime, goout, Dalc, Walc, health` | Nhân khẩu / thói quen | có | có | có |
| Tĩnh | 18 category: `school, sex, address, famsize, Pstatus, Mjob, Fjob, reason, guardian, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic, subject` | One-hot sau FIT | có | có | có |
| Temporal | `g_period` | Chuỗi điểm kỳ, T=2, C=1 | rỗng | `[G1/20]` | `[G1/20, G2/20]` |
| Aggregate | 5 số | latest, mean, coverage, delta, available | 0 | từ G1 | từ G1+G2 |
| Progress | 1 số | Mức thông tin | 0.0 | 0.5 | 1.0 |

**Output bước này (UCI):**

- `static ∈ ℝ^{N×Ds}` — Ds ≈ 50–60 sau one-hot (phụ thuộc vocabulary FIT).
- `temporal ∈ ℝ^{N×2×1}`
- `temporal_mask ∈ {0,1}^{N×2}`
- `lengths ∈ ℕ^{N}`
- `aggregate ∈ ℝ^{N×5}`
- `aggregate_available ∈ {0,1}^{N}`
- `progress ∈ [0,1]^{N}`

**Ví dụ S1 (đã có G1, chưa G2):**

```text
temporal      = [[0.55], [0.00]]     # G1=11/20; bước 2 padded
mask          = [1, 0]
lengths       = 1
aggregate     = [0.55, 0.55, 0.50, 0.00, 1.00]
progress      = 0.50
static        = [z_age, z_Medu, …, onehot_school=GP, …]   # Ds chiều
```

### 4.2 OULAD — đặc trưng temporal (11 kênh / tuần)

Mỗi tuần trước cutoff là một bước thời gian. Vector tuần `x_t ∈ ℝ^{11}`:

| Kênh | Ý nghĩa sau D3 |
|---|---|
| `activity_intensity_log1p` | `log1p`(click / ngày phơi nhiễm) |
| `active_days` | Tỷ lệ ngày có hoạt động trong tuần |
| `unique_sites` | `log1p`(số site khác nhau / ngày) |
| `unique_activity_types` | `log1p`(số loại hoạt động / ngày) |
| `content_activity` | Cường độ nội dung (oucontent, resource, page, …) |
| `forum_activity` | Cường độ forum |
| `quiz_activity` | Cường độ quiz |
| `assessment_related_activity` | Quiz / questionnaire |
| `weekly_submissions` | Số bài nộp trong tuần |
| `weekly_late_submissions` | Số nộp trễ |
| `week_exposure_fraction` | Phần tuần học viên còn trong khóa |

### 4.3 OULAD — aggregate (13 số, cắt tại cutoff)

| Đặc trưng | Ý nghĩa |
|---|---|
| `cumulative_activity` | Tổng hoạt động đến cutoff |
| `mean_weekly_activity` | Trung bình quy về tuần |
| `recent_activity` | Tuần gần nhất |
| `recent_historical_activity_ratio` | Gần / trung bình lịch sử |
| `activity_trend` | Độ dốc tuyến tính theo tuần |
| `current_inactivity_streak` | Chuỗi tuần không hoạt động (cuối) |
| `cumulative_inactive_weeks` | Tổng tuần inactive |
| `days_since_last_activity` | Ngày từ lần click cuối |
| `assessments_due_to_date` | Bài đã đến hạn trước cutoff |
| `submitted_due_to_date` | Đã nộp trước cutoff |
| `completion_rate` | Nộp / đến hạn |
| `missed_due_count` | Thiếu bài |
| `late_submission_rate` | Tỷ lệ nộp trễ |

### 4.4 OULAD — tĩnh

Số: `num_of_prev_attempts, studied_credits, registration_lead_time, module_presentation_length`.  
Category one-hot: `gender, region, highest_education, imd_band, age_band, disability, code_module, presentation_season`.

**Output bước này (OULAD, một mốc):**

```text
static     ∈ ℝ^{N×Ds}          # Ds sau one-hot FIT
temporal   ∈ ℝ^{N×T×11}        # T = số tuần đến cutoff
mask       ∈ {0,1}^{N×T}
aggregate  ∈ ℝ^{N×13}
progress   = f                 # 0.20 / 0.35 / 0.50 / 0.75 / 1.00
```

**Ví dụ 20% (T nhỏ, mới vài tuần):**

```text
x_t=0  = [log1p(click/ngày), active_days, …, exposure]
x_t=1  = …
mask   = [1,1,1,0,0,…]         # tuần sau cutoff = 0
progress = 0.20
```

Một checkpoint Hybrid được **chấm lại** ở mọi mốc; không train mô hình riêng cho từng tuần.

---

## 5. Tensor vào Hybrid (hợp đồng 6 kênh)

Mọi dataset về cùng schema `UnifiedHybridData`:

| Tensor | Shape | Vai trò |
|---|---|---|
| `static` | `[B, Ds]` | Tabular |
| `temporal` | `[B, T, Ct]` | Chuỗi (UCI Ct=1; OULAD Ct=11) |
| `temporal_mask` | `[B, T]` bool | Ô hợp lệ |
| `lengths` | `[B]` | `mask.sum(1)` |
| `aggregate` | `[B, Da]` | Tóm tắt (UCI Da=5; OULAD Da=13) |
| `aggregate_available` | `[B]` | Có aggregate hay không |
| `progress` | `[B]` | Mức thông tin `[0,1]` |

`B` = batch. Đây là **input duy nhất** của CNN, BiLSTM và nhánh tabular.

---

## 6. Kiến trúc mô hình

Một class `Hybrid` (architecture **C0**) cho cả UCI và OULAD. Khác nhau chỉ `Ds, Ct, Da` và trọng số đã fit. Không có mô hình riêng 100%, không có checkpoint theo stage.

Cấu hình dùng chung: `d_fuse = 128`, `cnn_channels = 64`, `bilstm_hidden = 128`, `cnn_blocks = 2`, kernel 2, dilation `(1, 2)`, BiLSTM 1 lớp, dropout theo dataset (UCI 0.41, OULAD 0.32).

### 6.1 Residual CNN (nhánh cục bộ)

**Input:** chuỗi đã adapter `adapted ∈ ℝ^{B×T×128}`, nhân mask.

1. Linear `128 → 64` nếu cần → `ℝ^{B×T×64}`.
2. Đổi trục thành Conv1d: `ℝ^{B×64×T}`.
3. Hai `ResidualTemporalBlock`: Conv1d nhân `k=2`, dilation 1 rồi 2, GELU, dropout, residual, **nhân mask** sau mỗi conv (không rò pad).
4. Pool mask-safe: mean ∥ max theo thời gian → `ℝ^{B×128}`.
5. Linear `128 → d_fuse` → **`h_cnn ∈ ℝ^{B×128}`**.

Nếu `lengths = 0` (UCI S0): `h_cnn = 0`.

**Minh họa một bước conv:** mỗi kênh nhìn 2 bước thời gian (kernel 2); dilation 2 nhìn cách một bước — bắt pattern G1→G2 hoặc hai tuần VLE gần nhau.

### 6.2 BiLSTM (nhánh dài hạn)

**Input:** cùng `adapted ∈ ℝ^{B×T×128}`, `lengths`.

1. `pack_padded_sequence` (bỏ pad).
2. LSTM 1 lớp, 2 chiều, hidden 128 mỗi chiều → mỗi bước `ℝ^{256}`.
3. Unpack, nhân mask.
4. Mean ∥ max → `ℝ^{B×512}`.
5. Linear `512 → 128` → **`h_lstm ∈ ℝ^{B×128}`**.

UCI S0: `h_lstm = 0`. OULAD: BiLSTM đọc xu hướng nhiều tuần (giảm click, streak nghỉ).

### 6.3 Nhánh tabular

- `ResidualProjector`: `static → ℝ^{128}` (shortcut + MLP LayerNorm–GELU).
- `ResidualProjector`: `aggregate → ℝ^{128}`, nhân `aggregate_available`.
- **`h_tab = h_static + h_aggregate ∈ ℝ^{128}`**.

### 6.4 Cổng fusion 3 nhánh (softmax có mask)

Ghép `[h_tab, h_cnn, h_lstm, available_3, progress]` → `ℝ^{388}`  
→ Linear 64, GELU, Dropout → Linear 3 → mask nhánh CNN/BiLSTM nếu không có temporal → **softmax**.

```text
w = softmax(g) ∈ Δ^3          # w_tab, w_cnn, w_lstm
h = w_tab·h_tab + w_cnn·h_cnn + w_lstm·h_lstm     ∈ ℝ^{128}
```

UCI S0: `available = [1,0,0]` → `w_tab = 1`.  
UCI S1/S2: cổng có thể nghiêng BiLSTM (chuỗi điểm ngắn).  
OULAD: cả ba nhánh có thể mở.

Regularization: entropy floor — phạt cổng quá “one-hot” khi nhiều nhánh available.

### 6.5 Head nhị phân

```text
h  --LayerNorm--> Linear(128,128) --GELU--Dropout--> Linear(128,1) --> z ∈ ℝ
p = σ(z) ∈ (0,1)
```

**Output mô hình (một học viên, một mốc):** một logit / một xác suất — không phải vector lớp đa nhãn.

---

## 7. Output module dự đoán

Từ `p` trên tập STOP:

1. Chọn ngưỡng `t ∈ (0,1)`: tối đa F1, hòa thì recall, rồi `|t−0.5|`.
2. `ŷ = 1` nếu `p ≥ t`.
3. Bất định nhị phân: `H2(p) = −[p log2 p + (1−p) log2(1−p)] ∈ [0,1]`.

Hợp đồng `PredictionResult` (đưa sang khuyến nghị, không lộ CNN/LSTM):

| Trường | Ví dụ (OULAD, 20%) |
|---|---|
| `risk_probability` `p` | 0.258 |
| `threshold` `t` | 0.18 |
| `predicted_risk` `ŷ` | 1 |
| `uncertainty` `H2(p)` | 0.823 |
| `stage_or_endpoint` | `20pct` |
| `record_id` | hash enrollment |

`risk_margin = p − t`. Module khuyến nghị **chỉ** nhận các trường này + evidence cutoff-safe, không đọc `h_cnn`.

---

## 8. Đánh giá module dự đoán

- Metric chính: **PR-AUC** (lớp risk lệch).
- Bốn thang báo cáo: **PR-AUC, Accuracy, F1 (lớp risk), Recall (lớp risk)**.
- Protocol: inner 3 fold × 3 seed trên FIT/STOP/VALID; early-stop STOP; metric VALID.
- Cùng checkpoint chấm S0→S2 hoặc 20→100.
- Baseline cùng split: Logistic Regression, Decision Tree, Random Forest, SVM, MLP. XGBoost không còn trong roster.

---

## 9. Leakage và overfitting khi train

**Leakage**

- Nhãn không vào X: UCI không G3; OULAD không `final_result` / `score`.
- UCI S0 không G1/G2; S1 không G2.
- OULAD: sự kiện `t < cutoff`; unregistration không phải feature.
- Group split: không cùng học viên (OULAD) / cùng identity (UCI) ở FIT và VALID.
- Scaler/one-hot FIT-only.
- Outer test không đụng HPO, epoch, ngưỡng.

**Overfitting**

- Early-stop macro PR-AUC trên STOP, không train đến khi train-loss kiệt.
- Weight decay + dropout + entropy floor trên cổng.
- Gap train−valid PR-AUC theo dõi từng mốc: UCI S0 gap lớn (ít tín hiệu); OULAD 20→100 gap thấp (~0.02–0.03).
- Không chọn mô hình bằng outer test.

---

## 10. Kết quả Hybrid C0

Mỗi mốc: lấy **một run có PR-AUC cao nhất** trên VALID, rồi ghi bốn thang của **cùng run đó**. Không trộn chỉ số giữa các lần chạy. Không dùng outer test.

### 10.1 UCI (S0 không có điểm)

| Mốc | PR-AUC | Accuracy | F1 | Recall |
|---|---:|---:|---:|---:|
| S0 | 0.5124 | 0.4234 | 0.4104 | 0.9649 |
| S1 | 0.8530 | 0.8787 | 0.7027 | 0.6610 |
| S2 | 0.9417 | 0.9412 | 0.8571 | 0.8136 |

S0: chỉ tabular → PR-AUC thấp, recall rất cao (ngưỡng thiên về bắt risk). Khi có G1 (S1) rồi G2 (S2), PR-AUC và Accuracy tăng mạnh — đúng mức thông tin.

### 10.2 OULAD (cùng một Hybrid, nhiều cutoff)

| Mốc | PR-AUC | Accuracy | F1 | Recall |
|---|---:|---:|---:|---:|
| 20% | 0.7707 | 0.6689 | 0.6811 | 0.8254 |
| 35% | 0.8119 | 0.7551 | 0.7038 | 0.7165 |
| 50% | 0.8594 | 0.8059 | 0.7421 | 0.7290 |
| 75% | 0.8993 | 0.8766 | 0.7974 | 0.7080 |
| 100% | 0.9297 | 0.9098 | 0.8508 | 0.7890 |

PR-AUC tăng đều theo cutoff. 100% đọc kèm nhiễu: khóa ngắn gần như đồng nghĩa Withdrawn; độ dài khóa không phải predictor tường minh.

---

## 11. So sánh baseline (cùng bốn thang)

Baseline: LR, DT, RF, SVM, MLP. Mỗi ô: run **PR-AUC cao nhất** của model đó tại mốc. In đậm = cao nhất tại ô.

### 11.1 UCI

**PR-AUC**

| Mốc | Hybrid | LR | DT | RF | SVM | MLP |
|---|---:|---:|---:|---:|---:|---:|
| S0 | 0.5124 | 0.5144 | 0.4635 | **0.5812** | 0.5390 | 0.5644 |
| S1 | **0.8530** | 0.8368 | 0.7502 | 0.8407 | 0.8276 | 0.8265 |
| S2 | **0.9417** | 0.9279 | 0.8745 | 0.9274 | 0.9182 | 0.9312 |

**Accuracy**

| Mốc | Hybrid | LR | DT | RF | SVM | MLP |
|---|---:|---:|---:|---:|---:|---:|
| S0 | 0.4234 | 0.5735 | 0.7206 | 0.7353 | 0.5625 | **0.8088** |
| S1 | **0.8787** | **0.8787** | **0.8787** | 0.8750 | 0.8750 | 0.8640 |
| S2 | **0.9412** | 0.9338 | 0.8787 | 0.8787 | 0.9044 | 0.9265 |

**F1**

| Mốc | Hybrid | LR | DT | RF | SVM | MLP |
|---|---:|---:|---:|---:|---:|---:|
| S0 | 0.4104 | 0.4775 | 0.5422 | **0.5556** | 0.4711 | 0.4694 |
| S1 | 0.7027 | **0.7227** | 0.6916 | 0.6909 | 0.6909 | 0.7040 |
| S2 | **0.8571** | 0.8333 | 0.7660 | 0.7626 | 0.7234 | 0.8077 |

**Recall**

| Mốc | Hybrid | LR | DT | RF | SVM | MLP |
|---|---:|---:|---:|---:|---:|---:|
| S0 | **0.9649** | 0.8983 | 0.7627 | 0.7627 | 0.8983 | 0.3898 |
| S1 | 0.6610 | 0.7288 | 0.6271 | 0.6441 | 0.6441 | **0.7458** |
| S2 | 0.8136 | 0.7627 | **0.9153** | 0.8983 | 0.5763 | 0.7119 |

Hybrid **thắng PR-AUC và F1 khi đã có điểm (S1–S2)**; S0 (không temporal) RF/MLP tốt hơn trên PR-AUC/Accuracy. Recall S0 Hybrid cao nhất.

### 11.2 OULAD

**PR-AUC**

| Mốc | Hybrid | LR | DT | RF | SVM | MLP |
|---|---:|---:|---:|---:|---:|---:|
| 20% | **0.7707** | 0.7667 | 0.7126 | 0.7571 | 0.7570 | 0.6954 |
| 35% | **0.8119** | 0.8068 | 0.7596 | 0.8008 | 0.7891 | 0.7490 |
| 50% | **0.8594** | 0.8488 | 0.8101 | 0.8487 | 0.8316 | 0.8036 |
| 75% | **0.8993** | 0.8925 | 0.8681 | 0.8932 | 0.8779 | 0.8651 |
| 100% | **0.9297** | 0.9200 | 0.8942 | 0.9227 | 0.9130 | 0.9040 |

**Accuracy**

| Mốc | Hybrid | LR | DT | RF | SVM | MLP |
|---|---:|---:|---:|---:|---:|---:|
| 20% | 0.6689 | **0.7035** | 0.6930 | 0.6824 | 0.6916 | 0.6529 |
| 35% | **0.7551** | 0.7200 | 0.6650 | 0.7442 | 0.7234 | 0.6853 |
| 50% | 0.8059 | 0.7872 | 0.7343 | **0.8079** | 0.7749 | 0.7594 |
| 75% | **0.8766** | 0.8682 | 0.8547 | 0.8664 | 0.8540 | 0.8508 |
| 100% | **0.9098** | 0.9020 | 0.9022 | 0.9042 | 0.8998 | 0.8880 |

**F1**

| Mốc | Hybrid | LR | DT | RF | SVM | MLP |
|---|---:|---:|---:|---:|---:|---:|
| 20% | 0.6811 | **0.6867** | 0.6453 | 0.6800 | 0.6839 | 0.6365 |
| 35% | **0.7038** | 0.7016 | 0.6575 | 0.6951 | 0.6830 | 0.6458 |
| 50% | **0.7421** | 0.7398 | 0.7004 | 0.7360 | 0.7187 | 0.6891 |
| 75% | **0.7974** | 0.7846 | 0.7751 | 0.7906 | 0.7747 | 0.7610 |
| 100% | **0.8508** | 0.8323 | 0.8288 | 0.8417 | 0.8278 | 0.8155 |

**Recall**

| Mốc | Hybrid | LR | DT | RF | SVM | MLP |
|---|---:|---:|---:|---:|---:|---:|
| 20% | **0.8254** | 0.7491 | 0.6438 | 0.7781 | 0.7787 | 0.7005 |
| 35% | 0.7165 | **0.8107** | 0.7920 | 0.7182 | 0.7338 | 0.7027 |
| 50% | 0.7290 | 0.7897 | **0.8107** | 0.6989 | 0.7505 | 0.6960 |
| 75% | 0.7080 | 0.6994 | 0.7295 | **0.7347** | 0.7318 | 0.6920 |
| 100% | **0.7890** | 0.7460 | 0.7264 | 0.7810 | 0.7387 | 0.7595 |

Trên OULAD, Hybrid **đứng nhất PR-AUC mọi mốc** và F1 từ 35% trở đi. Accuracy thắng từ 35% (trừ 50% RF sát). Recall cao nhất ở 20% và 100%.

---

## 12. Module khuyến nghị (Five-EBM-C0)

Chỉ OULAD, chỉ mốc 20/35/50/75 (100% không can thiệp). Không khuyến nghị UCI. Không Gemini lúc chạy. Không đụng lại Hybrid.

### 12.1 Pipeline I/O

| Bước | Input | Output |
|---|---|---|
| 1. Nhận C0 | `PredictionResult` | `p, t, ŷ, H2` |
| 2. Evidence | Raw OULAD, `t < cutoff` | Bảng learner–stage |
| 3. Feasibility | 5 hành động + evidence | Tập **eligible** (loại cứng) |
| 4. Ranker | 17 feature, **không** `action_id` | 5 EBM → điểm `[0,1]` |
| 5. Risk router | `p` vs `t`, `H2`, margin | NO_AUTOMATIC / HUMAN_REVIEW / PROCESS |
| 6. Safety router | Top scores, `H2` | RECOMMEND / HUMAN_REVIEW / INSUFFICIENT_EVIDENCE |
| 7. Plan | Hành động Top-1 + evidence | Kế hoạch xác định (không LLM) |

### 12.2 Năm hành động

`ASSESSMENT_COMPLETION`, `RECOVER_ENGAGEMENT`, `STUDY_REGULARITY`, `TARGETED_CONTENT_REVIEW`, `QUIZ_RETRIEVAL_PRACTICE`.

Ví dụ feasibility: `ASSESSMENT_COMPLETION` chỉ khi còn bài thiếu hoặc đến hạn 14 ngày; `TARGETED_CONTENT_REVIEW` cấm ở 20%; `QUIZ_RETRIEVAL_PRACTICE` cần quiz trên VLE.

### 12.3 Đặc trưng ranker (17)

| Nhóm | Đặc trưng |
|---|---|
| C0 | `risk_probability`, `uncertainty`, `risk_margin`, `stage` |
| Tiến độ | `course_progress`, `assessments_due`, `missing_assessment_count`, `due_soon_count`, `completion_rate` |
| Hành vi | `inactivity_streak`, `active_day_rate`, `regularity_score`, `content_coverage`, `quiz_activity` |
| Sẵn sàng | `vle_available`, `study_material_available`, `quiz_available` |

Cấm: `action_id`, `final_result`, `target`, `score`, nhãn yếu.

Mỗi hành động một **Explainable Boosting Machine** (hồi quy, target yếu 0–3 chuẩn hóa /3). Input: một hàng 17 số. Output: **một scalar** `s ∈ [0,1]` (mức liên quan hành động đó).

**Ví dụ:** học viên 20%, `p=0.26`, `H2=0.82`, `active_day_rate=0.31`  
→ eligible: RECOVER_ENGAGEMENT, STUDY_REGULARITY, QUIZ_RETRIEVAL_PRACTICE  
→ điểm ví dụ: 0.687 / 0.649 / 0.669  
→ `H2 > 0.70` → **HUMAN_REVIEW**, phát Top-3 + kế hoạch cho Top-1 (không tự động RECOMMEND).

### 12.4 Output khuyến nghị

| Route | Phát |
|---|---|
| `RECOMMEND` | Đúng 1 hành động + kế hoạch (thời hạn, tần suất, mục tiêu đo được) |
| `HUMAN_REVIEW` | Top-3 + kế hoạch Top-1 |
| `INSUFFICIENT_EVIDENCE` / `NO_FEASIBLE_ACTION` | Không hành động |

Kế hoạch là luật xác định (ví dụ RECOVER_ENGAGEMENT: “4 ngày VLE khác nhau trong 7 ngày”), **không** khẳng định hiệu quả nhân quả.

### 12.5 Chống leakage / overfit khuyến nghị

- Evidence cùng cutoff với C0; không `final_result`.
- EBM không thấy `action_id` (tránh shortcut nhãn).
- Feasibility **trước** rank: hành động không đủ điều kiện không được điểm.
- Panel C: 150 học viên / 632 case **rời** Panel A; reviewer không thấy `p`, rank, mô hình, outcome.
- Không Panel B, không chỉnh EBM sau freeze.

### 12.6 Đánh giá khuyến nghị

Bốn thang: **NDCG@3, Precision@1, MRR, Recall@3** (trên tập hành động eligible). Invalid-action rate phải 0.

Baseline: B0 (tần suất action×stage), B1 (điểm luật evidence).

### 12.7 Kết quả Panel C (held-out)

632 case, 2398 review thật.

| Mô hình | NDCG@3 | P@1 | MRR | R@3 |
|---|---:|---:|---:|---:|
| **Five-EBM-C0** | **0.88785** | 0.99206 | 0.99603 | 0.79947 |
| B0 | 0.81889 | 0.99365 | 0.99683 | 0.78981 |
| B1 | 0.86649 | **0.99683** | **0.99841** | **0.80357** |

Five-EBM-C0 **cao nhất NDCG@3** (thang xếp hạng chính). Bootstrap vs B1: trung bình +0.0213, 95% CI [0.0144, 0.0282]. Invalid-action = 0. Đủ 5 hành động ở Top-1.

Pipeline trên 632 case: RECOMMEND 94, HUMAN_REVIEW 175, INSUFFICIENT_EVIDENCE 363, NO_FEASIBLE_ACTION 0.

---

## 13. Ràng buộc diễn giải

- Hybrid là **dự đoán nguy cơ** tại một mốc thông tin, không phải chẩn đoán nguyên nhân.
- Tăng PR-AUC theo cutoff = thêm tín hiệu quan sát được, không chứng minh can thiệp làm đổi `final_result`.
- Khuyến nghị = xếp hạng hành động khả thi + kế hoạch hỗ trợ; **không** phải treatment effect.
- S0 UCI và OULAD 100% là giới hạn đã biết (thiếu temporal / nhiễu Withdrawn–độ dài khóa).

---

## 14. Tóm tắt luồng một case

```text
RAW studentVle/assessments  +  studentInfo
        │  cutoff = 20% độ dài khóa, t < cutoff
        ▼
temporal [T×11], aggregate [13], static [Ds], progress=0.20
        │  Hybrid C0 (CNN∥BiLSTM, cổng 3 nhánh)
        ▼
z ∈ ℝ  →  p=0.258, t=0.18, ŷ=1, H2=0.823
        │  + evidence thiếu bài / click
        ▼
eligible ⊂ {5 hành động}
        │  Five EBM, mỗi cái: ℝ¹⁷ → s∈[0,1]
        ▼
HUMAN_REVIEW  Top-3  + kế hoạch Top-1 (deterministic)
```

Đó là toàn bộ hệ thống: **dữ liệu thô → tensor có mask → biểu diễn 128 chiều × 3 nhánh → một xác suất nguy cơ → năm điểm hành động → một quyết định phát hành.**
