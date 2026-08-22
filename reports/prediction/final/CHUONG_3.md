# Chương 3. Đề xuất phương pháp: Hybrid CNN–BiLSTM và Recommendation V

Tài liệu này là bản kỹ thuật khóa luận cho **mô hình cuối**. Tên công khai: **Hybrid CNN–BiLSTM** (dự đoán) và **Recommendation V** (khuyến nghị). Không dùng mã thí nghiệm trên văn bản này.

Đánh giá dự đoán: trung bình robust inner 3 fold × 3 seed (`42`, `1201`, `2026`). Tập outer **không** dùng để chọn mô hình. Nguồn số: `reports/prediction/final/FINAL_PREDICTION_MODEL_REPORT.md`, `uci_final.csv`, `oulad_final.csv`.

---

## 3.1. Phân tích dữ liệu đầu vào

### 3.1.1. Hai bộ dữ liệu

**UCI Student Performance (Cortez & Silva, 2008).** Gộp Mathematics (395 dòng) và Portuguese (649 dòng) thành **1 044 bản ghi**, 33 cột gốc (CSV phân tách `;`). Nhãn nhị phân: `risk = 1` khi `G3 < 10`, tỷ lệ risk **0.220** (230/1044). `G3` trung bình 11.34. Nhóm tách: quasi-identity 13 trường (`school`, `sex`, `age`, `address`, `famsize`, `Pstatus`, `Medu`, `Fedu`, `Mjob`, `Fjob`, `reason`, `nursery`, `internet`) → 662 `global_student_group`; 366 nhóm xuất hiện ở cả hai môn.

Bản chất **tĩnh theo học kỳ**: mỗi bản ghi là một (học sinh, môn). Thông tin tăng theo mốc điểm: S0 chưa có `G1`/`G2`; S1 có `G1`; S2 có `G1` rồi `G2`. Chuỗi temporal dài tối đa **T = 2**.

**OULAD (Kuzilek, Hlosta & Zdrahal, 2017).** 32 593 enrollment, 28 785 sinh viên. `studentVle` 10 655 280 dòng click. Nhãn: `risk = 1` nếu `final_result ∈ {Fail, Withdrawn}`. Prevalence toàn cohort ≈ 0.528; tại 100% còn 22 522 bản ghi đủ điều kiện, prevalence 0.317 vì phần lớn Withdrawn đã rời trước cuối khóa (94 Withdrawn còn lại ở 100%).

Bản chất **tương tác theo thời gian**: mỗi enrollment là chuỗi tuần VLE + aggregate tại cutoff. Năm mốc: 20 / 35 / 50 / 75 / 100% chiều dài `module_presentation_length`. Sự kiện chỉ lấy khi `observation_start ≤ event_time < cutoff`.

Hai miền **không gộp**. Cùng class `Hybrid`, cùng topology; khác chiều input và checkpoint.

### 3.1.2. Thuộc tính — Spearman với nhãn risk (UCI, n = 1044)

| Thuộc tính | Spearman vs `G3<10` | Vào mô hình? |
|---|---:|---|
| `G3` | −0.722 | **Không** — chỉ tạo nhãn |
| `G2` | −0.675 | Temporal, chỉ S2 |
| `G1` | −0.628 | Temporal, S1 và S2 |
| `failures` | +0.376 | Static numeric |
| `age` | +0.128 | Static |
| `Fedu` / `studytime` / `goout` / `Medu` | ~±0.11 | Static |
| `absences` | +0.052 | **Cấm** (có thể đồng thời với kết quả) |

`G1`/`G2` mạnh nhưng **không** đưa vào nhánh static/aggregate của Hybrid CNN–BiLSTM — chỉ chuỗi temporal có mask, tránh “điểm đã biết” tràn tabular. `failures` là tín hiệu static mạnh nhất còn lại.

**OULAD — kênh được khóa trong code** (`src/prediction/data/oulad.py`):

Temporal 11 kênh/tuần: `activity_intensity_log1p`, `active_days`, `unique_sites`, `unique_activity_types`, `content_activity`, `forum_activity`, `quiz_activity`, `assessment_related_activity`, `weekly_submissions`, `weekly_late_submissions`, `week_exposure_fraction`.

Aggregate 13 số tại cutoff: `cumulative_activity`, `mean_weekly_activity`, `recent_activity`, `recent_historical_activity_ratio`, `activity_trend`, `current_inactivity_streak`, `cumulative_inactive_weeks`, `days_since_last_activity`, `assessments_due_to_date`, `submitted_due_to_date`, `completion_rate`, `missed_due_count`, `late_submission_rate`.

Context tĩnh: categorical `gender`, `region`, `highest_education`, `imd_band`, `age_band`, `disability`, `code_module`, `presentation_season`; numeric `num_of_prev_attempts`, `studied_credits`, `registration_lead_time`, `module_presentation_length`.

Cấm predictor: `final_result`, `score`, `date_unregistration`.

---

## 3.2. Tiền xử lý và đặc trưng

### 3.2.1. Thu thập

File gốc: `data/raw/student-mat.csv`, `student-por.csv`; OULAD `studentInfo`, `studentRegistration`, `studentVle`, `assessments`, `studentAssessment`, `courses`, `vle`. SHA-256 từng file khóa trong protocol. Không lấy dump ngoài repo.

### 3.2.2. Làm sạch và tái cấu trúc thời gian

UCI: bắt buộc 395+649 dòng; thêm `subject`; `target` từ `G3` một lần.

OULAD: join enrollment–registration–course. Log VLE gom **theo tuần** trước cutoff. Assessment chỉ tính hạn/`date_submitted` **trước** cutoff. Withdrawal sau cutoff không thành đặc trưng. 100%: 94 Withdrawn còn lại — không dùng độ dài chuỗi làm proxy Withdrawn cho cảnh báo sớm.

### 3.2.3. Tensor thống nhất

Một dataclass `UnifiedHybridData` cho cả hai miền:

`static [N, Ds]`, `temporal [N, T, C]`, `temporal_mask [N, T]`, `lengths [N]`, `aggregate [N, Da]`, `aggregate_available [N]`, `progress [N]`, `target`, `record_id`, `group_id`.

UCI: `T=2`, `C` điểm chuẩn hóa `/20` và delta; S0 mask toàn 0; aggregate tắt (`aggregate_available=0`). OULAD: `T` pad theo mốc dài nhất (tới 39 tuần ở 100%), 11 kênh, 13 aggregate. Chuẩn hóa **FIT-only** (`ContextPreprocessor`, `MaskedStandardScaler`). Cùng class `Hybrid` nhận tensor này; khác nhau chỉ `Ds`, `C`, `Da` và trọng số.

### 3.2.4. Chia dữ liệu — outer không để chọn mô hình

- Group-split: UCI `global_student_group`, OULAD `id_student`. Không cùng người ở hai phía split.
- Outer 3 fold; **fold 0 outer là firewall** lúc phát triển — không tune, không chọn kiến trúc.
- Inner 3 fold trên phần còn lại: FIT / STOP / VALID. STOP: early-stop và chọn ngưỡng `t`. VALID: báo cáo. Seed split 42; seed train 42, 1201, 2026.
- Outer test **không** dùng khi chốt Hybrid CNN–BiLSTM (`outer_test_used_for_phase4_finalization: false`).

---

## 3.3. Hybrid CNN–BiLSTM

### 3.3.1. Kiến trúc dùng chung

Code: `src/prediction/model/hybrid.py`. `model_id = hybrid`. Một checkpoint UCI chấm S0–S2; một checkpoint OULAD chấm 20–100%. Không mô hình riêng 100%.

```
static     → ResidualProjector → h_tab ∈ ℝ^128   (cộng nhánh aggregate nếu available)
temporal   → Linear+LN 128
           → Residual CNN (64 kênh, 2 block, kernel 2, dilation 1 rồi 2)
             → masked mean-max → h_cnn ∈ ℝ^128
           → BiLSTM (hidden 128, 1 lớp, hai chiều)
             → masked pool → h_lstm ∈ ℝ^128
[h_tab, h_cnn, h_lstm, availability 3 bit, progress] → softmax 3 nhánh (mask nhánh CNN/LSTM khi T=0)
fused ∈ ℝ^128 → LayerNorm → Linear 128 → GELU → Dropout → Linear 1 → logit z
p = σ(z)
```

S0 / 20% rất sớm: `lengths=0` → CNN và BiLSTM **tắt**, chỉ tabular. Cấu hình chung: `d_fuse=128`, `cnn_channels=64`, `bilstm_hidden=128`. Khác dataset: `lr`, `dropout`, `batch_size`, `pos_weight_multiplier` (quy mô), **không** đổi topology.

Output serving: `p`, ngưỡng `t` (STOP), `ŷ = [p ≥ t]`, bất định `H₂(p)`. Hợp đồng `PredictionResult` — Recommendation V không đọc CNN/LSTM.

### 3.3.2. Huấn luyện

- Mất mát: BCE with logits, **`pos_weight = (n_neg/n_pos) × hệ số FIT`** (UCI hệ số 1.183, OULAD 0.779). Đây là xử lý lệch lớp **cost-sensitive**, cùng công thức mọi fold. SMOTE/ADASYN trên tensor Hybrid: đã thử FIT-only, không chọn (nội suy chuỗi không tạo điểm/VLE thật).
- Tối ưu: AdamW. UCI `lr = 8.61×10⁻⁵`, `weight_decay = 3.29×10⁻³`, `dropout = 0.406`, `batch = 32`. OULAD `lr = 1.18×10⁻⁴`, `weight_decay = 7.11×10⁻⁴`, `dropout = 0.320`, `batch = 128`.
- Early-stop trên STOP (macro AP). Ngưỡng: lưới STOP, xếp F1 rồi recall rồi `|t−0.5|`, áp VALID.
- Seed 42, 1201, 2026. AMP GPU khi train nghiên cứu; serving đọc xác suất đã lưu.

### 3.3.3. Rò rỉ và quá khớp

**Rò rỉ đã chặn:** `G3` không phải predictor; `G1`/`G2` không vào static Hybrid; `absences` cấm; OULAD cấm `final_result`, `score`, `date_unregistration`; sự kiện `event_time < cutoff`; scaler FIT-only; group-disjoint; VALID không chọn `t`; outer không chọn mô hình.

**Quá khớp:** dropout + weight decay + early-stop STOP; 3 seed; S0/20% báo như hạn chế (không “thắng”); ablation temporal shuffle khi debug kiến trúc. Không SMOTE lên STOP/VALID.

### 3.3.4. Kết quả — AP, F1, Recall, Accuracy (robust 3×3)

Lệch lớp: AP (`sklearn.average_precision_score`) là chỉ số xếp hạng; F1/Recall/Accuracy tại `t` STOP. Không dùng R²/RMSE.

**UCI Combined (prevalence 0.220)**

| Mốc | Accuracy | AP | F1 | Recall |
|---|---:|---:|---:|---:|
| S0 | 0.5213 | 0.4547 | 0.4291 | 0.8421 |
| S1 | 0.8553 | **0.8214** | 0.6899 | 0.7587 |
| S2 | 0.9094 | **0.9101** | 0.8010 | 0.8545 |

S0 chưa có điểm: AP 0.45 — hạn chế, không phải thắng. S1→S2: AP +0.089 khi thêm `G2`.

**OULAD**

| Mốc | Accuracy | AP | F1 | Recall | n đủ điều kiện |
|---|---:|---:|---:|---:|---:|
| 20% | 0.6854 | 0.7624 | 0.6781 | 0.7769 | 26 697 |
| 35% | 0.7456 | 0.8058 | 0.7001 | 0.7464 | 25 606 |
| 50% | 0.8006 | 0.8483 | 0.7306 | 0.7207 | 24 599 |
| 75% | 0.8627 | 0.8885 | 0.7807 | 0.7221 | 23 159 |
| 100% | 0.9088 | 0.9204 | 0.8372 | 0.7807 | 22 522 |

AP tăng đều theo cutoff (+0.158 từ 20% đến 100%). 100% không dùng cho khuyến nghị.

### 3.3.5. So sánh baseline cùng protocol serving

Roster phục vụ: Hybrid CNN–BiLSTM, LR, DT, RF, SVM, MLP. **Một checkpoint Hybrid** chấm mọi mốc. XGBoost không nằm roster serving.

**UCI — AP**

| Mô hình | S0 | S1 | S2 |
|---|---:|---:|---:|
| Hybrid CNN–BiLSTM | 0.4547 | **0.8214** | **0.9101** |
| LR | 0.4754 | 0.7794 | 0.8812 |
| RF | **0.4995** | 0.7895 | 0.9072 |
| SVM | 0.4970 | 0.7936 | 0.8866 |
| DT | 0.4169 | 0.7330 | 0.8547 |
| MLP | 0.4486 | 0.7595 | 0.8778 |

S1 và S2 Hybrid đứng đầu AP. Macro UCI thua RF vì **S0** (RF 0.4995 vs 0.4547) — đúng khi chưa có chuỗi điểm.

**OULAD — AP**

| Mô hình | 20 | 35 | 50 | 75 | 100 |
|---|---:|---:|---:|---:|---:|
| Hybrid CNN–BiLSTM | 0.7624 | **0.8058** | **0.8483** | **0.8885** | **0.9204** |
| LR | **0.7632** | 0.7986 | 0.8399 | 0.8828 | 0.9114 |
| RF | 0.7522 | 0.7940 | 0.8402 | 0.8847 | 0.9154 |
| SVM | 0.7534 | 0.7835 | 0.8257 | 0.8723 | 0.9018 |
| DT | 0.7084 | 0.7548 | 0.7954 | 0.8530 | 0.8862 |
| MLP | 0.6799 | 0.7388 | 0.7998 | 0.8556 | 0.8964 |

Hybrid đứng nhất AP từ 35% đến 100%; 20% thua LR 0.0008.

---

## 3.4. Recommendation V

### 3.4.1. Luồng

```
Hybrid CNN–BiLSTM → PredictionResult (p, t, ŷ, H₂)
  → chỉ OULAD 20/35/50/75 (100% bị từ chối)
  → bộ định tuyến risk quanh t và H₂
  → feasibility cứng 5 hành động
  → năm EBM, mỗi cái ℝ¹⁷ → s ∈ [0,1]
  → RECOMMEND Top-1  hoặc  HUMAN_REVIEW Top-3
  → kế hoạch xác định, không LLM lúc chạy
```

Code: `src/recommend_hybrid/v3/`. Không refit Hybrid. Không đọc class CNN/LSTM.

### 3.4.2. Hành động và an toàn

Năm hành động: `ASSESSMENT_COMPLETION`, `RECOVER_ENGAGEMENT`, `STUDY_REGULARITY`, `TARGETED_CONTENT_REVIEW`, `QUIZ_RETRIEVAL_PRACTICE`.

Bốn trạng thái: `RECOMMEND`, `HUMAN_REVIEW`, `INSUFFICIENT_EVIDENCE`, `NO_FEASIBLE_ACTION`. `TARGETED_CONTENT_REVIEW` bị chặn quá sớm (20%). Invalid-action trên Panel C: **0**.

Panel C held-out: 632 case, 150 sinh viên, 2398 review. NDCG@3 **0.88785** vs B1 0.86649; hiệu +0.0213, 95% CI [0.0144, 0.0282]. P@1 0.992. Trên 632 case: RECOMMEND 94, HUMAN_REVIEW 175, INSUFFICIENT_EVIDENCE 363, NO_FEASIBLE_ACTION 0.

### 3.4.3. Ràng buộc diễn giải

`p` là nguy cơ nhị phân **tại một mốc thông tin**, không phải nguyên nhân trượt. Recommendation V xếp **hành động khả thi** theo relevance đã học, không phải ATE. Cấm viết: can thiệp này làm tăng `G3` hay đổi `final_result`. Gemini chỉ weak label lúc xây ranking, không gán Risk, không chọn kiến trúc Hybrid.

---

## 3.5. Luồng hệ thống

### 3.5.1. Một trường hợp OULAD 20%

Sinh viên `id_student`, môn `CCC`, kỳ `2014B`, cutoff = 20% × `module_presentation_length`.

1. Đọc `studentInfo` + `studentRegistration` + `courses` → static.  
2. `studentVle` tuần có `event_time < cutoff` → `temporal` 11 kênh, mask.  
3. Assessment hạn trước cutoff → 13 aggregate.  
4. Checkpoint Hybrid CNN–BiLSTM OULAD → `p`, `H₂(p)`. `t` lấy từ STOP đã khóa. `ŷ = [p ≥ t]`.  
5. Recommendation V: nếu `p` đủ margin và H₂ thấp và còn hành động khả thi → EBM ranking → `RECOMMEND` Top-1; nếu bất định cao → `HUMAN_REVIEW` Top-3; nếu thiếu bằng chứng VLE/assessment → `INSUFFICIENT_EVIDENCE`.  
6. PostgreSQL: `raw` → `catalog` → `prediction` → `recommendation`. `python project.py db predict|recommend` không train lại.

UCI S2 tương tự: static context + chuỗi `(G1/20, G2/20)` → một checkpoint UCI → `p`; **không** gọi Recommendation V (module chỉ OULAD).

### 3.5.2. Giới hạn chương

**Đã kiểm soát:** nhãn nhị phân tường minh; cấm G3/`final_result`/`score`; FIT-only scale; group-split; STOP-only `t`; outer không chọn mô hình; một kiến trúc hai miền; lệch lớp bằng `pos_weight` (SMOTE tensor: thử, không chọn); Recommendation V không nhân quả.

**Cần nói rõ khi viết học thuật:** S0/20% Hybrid không vượt RF/LR — thiếu chuỗi. Siêu tham số dataset-specific (lr, dropout, batch) là ngoại lệ quy mô, không phải hai kiến trúc. AP UCI và AP OULAD không so trực tiếp (khác prevalence). 100% OULAD không phải cảnh báo sớm.
