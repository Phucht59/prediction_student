# Chương 3. Đề xuất phương pháp: Hybrid CNN–BiLSTM và Recommendation V

Tài liệu này là bản kỹ thuật khóa luận cho **mô hình cuối**. Tên công khai: **Hybrid CNN–BiLSTM** (dự đoán) và **Recommendation V** (khuyến nghị). Không dùng mã thí nghiệm trên văn bản này.

Đánh giá dự đoán: trung bình robust inner 3 fold × 3 seed (`42`, `1201`, `2026`). Tập outer **không** dùng để chọn mô hình. Nguồn số: `uci_final.csv`, `oulad_final.csv`. AP là `sklearn.metrics.average_precision_score` — **không** gọi AP là PR-AUC (PR-AUC hình thang khác công thức).

---

## 3.1. Phân tích dữ liệu đầu vào

### 3.1.1. Hai bộ dữ liệu

**UCI Student Performance (Cortez & Silva, 2008).** Gộp Mathematics (395 dòng) và Portuguese (649 dòng) thành **1 044 bản ghi**, 33 cột gốc (CSV phân tách `;`). Nhãn nhị phân: `risk = 1` khi `G3 < 10`, tỷ lệ risk **0.220** (230/1044). `G3` trung bình 11.34. Nhóm tách: quasi-identity 13 trường (`school`, `sex`, `age`, `address`, `famsize`, `Pstatus`, `Medu`, `Fedu`, `Mjob`, `Fjob`, `reason`, `nursery`, `internet`) → 662 `global_student_group`; 366 nhóm xuất hiện ở cả hai môn.

Bản chất **tĩnh theo học kỳ**: mỗi bản ghi là một (học sinh, môn). Thông tin tăng theo mốc điểm: S0 chưa có `G1`/`G2`; S1 có `G1`; S2 có `G1` rồi `G2`. Chuỗi temporal dài tối đa **T = 2**. Không trộn hai môn thành một chuỗi thời gian — `subject` là context tĩnh.

**OULAD (Kuzilek, Hlosta & Zdrahal, 2017).** 32 593 enrollment, 28 785 sinh viên. `studentVle` 10 655 280 dòng click. Nhãn: `risk = 1` nếu `final_result ∈ {Fail, Withdrawn}`. Prevalence toàn cohort ≈ 0.528; tại 100% còn 22 522 bản ghi đủ điều kiện, prevalence 0.317 vì phần lớn Withdrawn đã rời trước cuối khóa (94 Withdrawn còn lại ở 100%).

Bản chất **tương tác theo thời gian**: mỗi enrollment là chuỗi tuần VLE + aggregate tại cutoff. Năm mốc: 20 / 35 / 50 / 75 / 100% chiều dài `module_presentation_length`. Sự kiện chỉ lấy khi `observation_start ≤ event_time < cutoff`. Số bản ghi còn đủ điều kiện (chưa unregistration trước cutoff): 20% 26 697; 35% 25 606; 50% 24 599; 75% 23 159; 100% 22 522.

Hai miền **không gộp**. Cùng class `Hybrid`, cùng topology; khác chiều input và checkpoint. AP UCI và AP OULAD không so trực tiếp vì khác prevalence và khác nguồn tín hiệu.

SHA-256 file gốc (khóa protocol):

| File | SHA-256 |
|---|---|
| `student-mat.csv` | `e47f9ee2…ef5dec80` |
| `student-por.csv` | `a7594a11…63fb3f` |
| `studentInfo.csv` | `7e6f3e47…99b0d6` |
| `studentVle.csv` | `52668253…b937f0` |
| `studentAssessment.csv` | `fd532078…510a6e` |
| `assessments.csv` | `8cc738fb…5a046f1` |
| `studentRegistration.csv` | `0d326762…e90170` |
| `courses.csv` | `4f16eee7…737a75` |
| `vle.csv` | `d1b28303…fe87e9` |

Hash đủ 64 ký tự nằm trong protocol khóa; bảng trên chỉ để định danh nguồn in-repo. Không lấy dump ngoài repo.

### 3.1.2. Thuộc tính — Spearman với nhãn risk (UCI, n = 1044)

Spearman trên 1 044 dòng gộp, nhãn `G3 < 10`. Đây là mô tả dữ liệu thô, **không** phải thứ tự importance sau FIT-scale.

| Thuộc tính | Spearman | p | Vào mô hình? |
|---|---:|---:|---|
| `G3` | −0.722 | < 10⁻¹⁶⁰ | **Không** — chỉ tạo nhãn |
| `G2` | −0.675 | < 10⁻¹³⁹ | Temporal, chỉ S2 |
| `G1` | −0.628 | < 10⁻¹¹⁴ | Temporal, S1 và S2 |
| `failures` | +0.376 | 2.0×10⁻³⁶ | Static numeric |
| `age` | +0.128 | 3.5×10⁻⁵ | Static |
| `Fedu` | −0.111 | 3.5×10⁻⁴ | Static |
| `studytime` | −0.110 | 3.5×10⁻⁴ | Static |
| `goout` | +0.110 | 3.8×10⁻⁴ | Static |
| `Medu` | −0.109 | 4.4×10⁻⁴ | Static |
| `Dalc` | +0.097 | 1.8×10⁻³ | Static |
| `Walc` | +0.072 | 2.0×10⁻² | Static |
| `freetime` | +0.068 | 2.8×10⁻² | Static |
| `absences` | +0.052 | 0.091 | **Cấm** (có thể đồng thời với kết quả) |
| `health` / `traveltime` / `famrel` | \|ρ\| < 0.04 | > 0.25 | Static, tín hiệu yếu |

`G1`/`G2` mạnh nhưng **không** đưa vào nhánh static/aggregate của Hybrid CNN–BiLSTM — chỉ chuỗi temporal có mask, tránh “điểm đã biết” tràn tabular. `failures` là tín hiệu static mạnh nhất còn lại. `absences` cấm dù Spearman yếu: không phải vì |ρ| nhỏ, mà vì đồng thời với kết quả.

Context tĩnh UCI còn lại (không hiện Spearman ở bảng): categorical `school`, `sex`, `address`, `famsize`, `Pstatus`, `Mjob`, `Fjob`, `reason`, `guardian`, `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`, `romantic`, `subject`; numeric thêm `traveltime`, `famrel`, `freetime`, `health`.

**OULAD — kênh được khóa trong code** (`src/prediction/data/oulad.py`):

Temporal **11 kênh/tuần**: `activity_intensity_log1p`, `active_days`, `unique_sites`, `unique_activity_types`, `content_activity`, `forum_activity`, `quiz_activity`, `assessment_related_activity`, `weekly_submissions`, `weekly_late_submissions`, `week_exposure_fraction`.

Aggregate **13 số tại cutoff**: `cumulative_activity`, `mean_weekly_activity`, `recent_activity`, `recent_historical_activity_ratio`, `activity_trend`, `current_inactivity_streak`, `cumulative_inactive_weeks`, `days_since_last_activity`, `assessments_due_to_date`, `submitted_due_to_date`, `completion_rate`, `missed_due_count`, `late_submission_rate`.

Context tĩnh: categorical `gender`, `region`, `highest_education`, `imd_band`, `age_band`, `disability`, `code_module`, `presentation_season`; numeric `num_of_prev_attempts`, `studied_credits`, `registration_lead_time`, `module_presentation_length`.

Cấm predictor: `final_result`, `score`, `date_unregistration`. Không dùng độ dài chuỗi làm cột — nhưng tại 100% độ dài quan sát vẫn **liên đới** Withdrawn (mục 3.3.3).

---

## 3.2. Tiền xử lý và đặc trưng

### 3.2.1. Thu thập

File gốc: `data/raw/student-mat.csv`, `student-por.csv`; OULAD `studentInfo`, `studentRegistration`, `studentVle`, `assessments`, `studentAssessment`, `courses`, `vle`. SHA-256 từng file khóa trong protocol (mục 3.1.1). Không lấy dump ngoài repo.

### 3.2.2. Làm sạch và tái cấu trúc thời gian

UCI: bắt buộc 395+649 dòng; thêm `subject`; `target` từ `G3` một lần. `record_id` = hash ổn định (`subject`, chỉ số dòng nguồn, chữ ký identity). `global_student_group` = hash 13 trường quasi-identity — một người học cả toán và Bồ Đào Nha không được nằm hai phía split.

OULAD: join enrollment–registration–course. Log VLE gom **theo tuần** trước cutoff. Assessment chỉ tính hạn/`date_submitted` **trước** cutoff. Withdrawal sau cutoff không thành đặc trưng. 100%: 94 Withdrawn còn lại — không dùng độ dài chuỗi làm proxy Withdrawn cho cảnh báo sớm.

Quy tắc cutoff OULAD: `observation_start ≤ event_time < cutoff`. `cutoff = fraction × module_presentation_length`. Enrollment đã `date_unregistration < cutoff` bị loại khỏi mốc đó (không còn là đối tượng cảnh báo).

### 3.2.3. Tensor thống nhất

Một dataclass `UnifiedHybridData` cho cả hai miền (`src/prediction/data/common.py`):

`static [N, Ds]`, `temporal [N, T, C]`, `temporal_mask [N, T]`, `lengths [N]`, `aggregate [N, Da]`, `aggregate_available [N]`, `progress [N]`, `target`, `record_id`, `group_id`.

Ràng buộc: `lengths = sum(mask)`; ô temporal bị mask phải ≈ 0; `target ∈ {0,1}`; `progress ∈ [0,1]`; `aggregate_available ∈ {0,1}`.

| Miền | T | C | Da | progress | Ghi chú |
|---|---:|---:|---:|---|---|
| UCI S0 | 2 | điểm/20 | tắt | 0.00 | mask toàn 0; CNN/BiLSTM tắt |
| UCI S1 | 2 | điểm/20 | tắt | 0.50 | chỉ bước G1 |
| UCI S2 | 2 | điểm/20 | tắt | 1.00 | G1 rồi G2 |
| OULAD 20–100% | pad tới mốc dài nhất (tới ~39 tuần ở 100%) | 11 | 13 | 0.20 … 1.00 | aggregate luôn available |

Chuẩn hóa **FIT-only**: UCI/OULAD context one-hot + scale trên FIT; temporal `MaskedStandardScaler` (chỉ ô mask=1); aggregate mean/std FIT. STOP/VALID/outer **không** refit scaler. Cùng class `Hybrid` nhận tensor này; khác nhau chỉ `Ds`, `C`, `Da` và trọng số.

### 3.2.4. Chia dữ liệu — outer không để chọn mô hình

- Group-split: UCI `global_student_group`, OULAD `id_student`. Không cùng người ở hai phía split.
- Outer 3 fold; **fold 0 outer là firewall** lúc phát triển — không tune, không chọn kiến trúc.
- Inner 3 fold trên phần còn lại: FIT / STOP / VALID. STOP: early-stop và chọn ngưỡng `t`. VALID: báo cáo. Seed split 42; seed train 42, 1201, 2026.
- Outer test **không** dùng khi chốt Hybrid CNN–BiLSTM (`outer_test_used_for_phase4_finalization: false`).

Hash split khóa: inner UCI `ad8f44e5…e02ae8`, inner OULAD `8559efcf…72650c`. Không regenerate split khi viết chương.

---

## 3.3. Hybrid CNN–BiLSTM

### 3.3.1. Kiến trúc dùng chung

Code: `src/prediction/model/hybrid.py`. `model_id = hybrid`. Một checkpoint UCI chấm S0–S2; một checkpoint OULAD chấm 20–100%. Không mô hình riêng 100%. Cấu hình serving: `d_fuse=128`, `cnn_channels=64`, `cnn_blocks=2`, `kernel=2`, `dilation=(1,2)`, `bilstm_hidden=128`, 1 lớp hai chiều, fusion softmax 3 nhánh.

```
static     → ResidualProjector → h_static ∈ ℝ^128
aggregate  → ResidualProjector → h_agg ∈ ℝ^128
             h_tab = h_static + 1[agg] · h_agg
temporal   → Linear+LN 128, nhân mask
           → Residual CNN (64 kênh, 2 block, kernel 2, dilation 1 rồi 2)
             → masked mean-max → Linear → h_cnn ∈ ℝ^128
           → BiLSTM (hidden 128, 1 lớp, hai chiều)
             → masked pool (4×hidden) → Linear → h_lstm ∈ ℝ^128
nếu lengths=0: h_cnn = h_lstm = 0   (S0 / chuỗi rỗng)

g = softmax( Gate([h_tab, h_cnn, h_lstm, a_tab=1, a_cnn, a_lstm, progress]) )
     với a_cnn = a_lstm = 1[lengths>0]; logit nhánh tắt = −∞
h = g_tab h_tab + g_cnn h_cnn + g_lstm h_lstm
z = Head(LN(h))     Head: LN → Linear 128 → GELU → Dropout → Linear 1
p = σ(z)
```

S0 / 20% rất sớm: `lengths=0` → CNN và BiLSTM **tắt**, chỉ tabular. Khác dataset: `lr`, `dropout`, `batch_size`, `pos_weight_multiplier`, hệ số entropy-floor (UCI 0.002, OULAD 0.005) — **không** đổi topology. Entropy-floor phạt cổng quá chắc khi nhiều nhánh available, không phải loss chính.

Output serving: `p`, ngưỡng `t` (STOP), `ŷ = [p ≥ t]`, bất định `H₂(p) = −p log p − (1−p) log(1−p)`. Hợp đồng `PredictionResult` — Recommendation V không đọc CNN/LSTM, không đọc `g`.

### 3.3.2. Huấn luyện

Mất mát chính: BCE with logits, **cost-sensitive**

```
pos_weight_FIT = (n_neg / n_pos)_FIT × hệ_số
UCI hệ_số = 1.183    OULAD hệ_số = 0.779
```

Cùng công thức mọi fold; hệ số FIT-only. Đây là xử lý lệch lớp đã chọn. SMOTE/ADASYN trên tensor Hybrid: thử FIT-only, **không chọn** (nội suy chuỗi không tạo điểm/VLE thật; UCI screen SMOTE hạ F1, S1 recall ~0.49). Focal+SMOTE tăng phương sai fold. Kết quả âm giữ làm bằng chứng, không giấu.

Tối ưu AdamW, một bộ siêu tham số / miền (không một bộ / mốc):

| | lr | weight_decay | dropout | batch |
|---|---:|---:|---:|---:|
| UCI | 8.61×10⁻⁵ | 3.29×10⁻³ | 0.406 | 32 |
| OULAD | 1.18×10⁻⁴ | 7.11×10⁻⁴ | 0.320 | 128 |

Thuật toán một fold / một seed:

1. Fit scaler + `pos_weight` trên FIT.
2. Train trên FIT, early-stop macro AP trên STOP (không phải VALID).
3. Lưới ngưỡng `t` trên STOP: xếp **F1**, rồi recall, rồi `|t − 0.5|`.
4. Áp `t` đã chọn lên VALID — VALID không chọn `t`.
5. Lặp 3 inner fold × 3 seed; báo **trung bình 9 số**, không lấy run đẹp nhất.

AMP GPU khi train nghiên cứu; serving đọc xác suất đã lưu. Không joint-train hai miền.

### 3.3.3. Rò rỉ và quá khớp

**Rò rỉ đã chặn:** `G3` không phải predictor; `G1`/`G2` không vào static Hybrid; `absences` cấm; OULAD cấm `final_result`, `score`, `date_unregistration`; sự kiện `event_time < cutoff`; scaler FIT-only; group-disjoint; VALID không chọn `t`; outer không chọn mô hình. Audit khóa: `g3_in_predictors=false`, `s0_has_g1g2=false`.

**100% OULAD — confounder độ dài, không phải future-event leak.** Withdrawn trung bình 9.17 tuần quan sát; Pass / Distinction / Fail ≈ 37 tuần. `length` vs Withdrawn: AP 0.991, ROC-AUC 0.993. Tỷ lệ Withdrawn trong lịch sử ngắn ≈ 0.999. Hybrid **không** nhận cột length; vẫn phải đọc 100% như hạn chế diễn giải, không phải cảnh báo sớm.

**Quá khớp** (9 run/mốc; khe = AP_FIT − AP_VALID). Ngưỡng lớp: HIGH nếu khe ≥ 0.10 hoặc std ≥ 0.05; MODERATE nếu khe ≥ 0.04 hoặc std ≥ 0.02; còn lại LOW.

| Mốc | AP VALID | AP std | AP FIT | khe | lớp |
|---|---:|---:|---:|---:|---|
| UCI S0 | 0.4547 | 0.043 | 0.5801 | **0.1254** | HIGH |
| UCI S1 | 0.8214 | 0.034 | 0.8566 | 0.0352 | MODERATE |
| UCI S2 | 0.9101 | 0.022 | 0.9304 | 0.0203 | MODERATE |
| OULAD 20% | 0.7624 | 0.007 | 0.7963 | 0.0339 | LOW |
| OULAD 35% | 0.8058 | 0.004 | 0.8371 | 0.0312 | LOW |
| OULAD 50% | 0.8483 | 0.007 | 0.8722 | 0.0238 | LOW |
| OULAD 75% | 0.8885 | 0.008 | 0.9088 | 0.0203 | LOW |
| OULAD 100% | 0.9204 | 0.006 | 0.9359 | 0.0155 | LOW |

Dropout + weight decay + early-stop STOP; 3 seed. S0/20% báo như hạn chế (không “thắng”). Không SMOTE lên STOP/VALID.

### 3.3.4. Kết quả — AP, F1, Precision, Recall, Accuracy (robust 3×3)

Lệch lớp: AP là chỉ số xếp hạng, không cần ngưỡng. Accuracy / Precision / Recall / F1 tại `t` STOP. F1 là trung hòa điều hòa của Precision và Recall — **một `t` không tối đa đồng thời cả ba**. Không dùng R²/RMSE.

**UCI Combined (prevalence 0.220)**

| Mốc | Accuracy | AP | Precision | F1 | Recall |
|---|---:|---:|---:|---:|---:|
| S0 | 0.5213 | 0.4547 | 0.2911 | 0.4291 | 0.8421 |
| S1 | 0.8553 | **0.8214** | 0.6604 | 0.6899 | 0.7587 |
| S2 | 0.9094 | **0.9101** | 0.7654 | 0.8010 | 0.8545 |

S0 chưa có điểm: AP 0.45, Precision 0.29 — hạn chế, không phải thắng. Recall S0 cao vì `t` lệch về bắt risk (prevalence 0.22). S0→S1: AP +0.367 khi có `G1`. S1→S2: AP +0.089 khi thêm `G2`. ECE Hybrid: S0 0.254, S1 0.129, S2 0.117.

**OULAD**

| Mốc | Accuracy | AP | Precision | F1 | Recall | n |
|---|---:|---:|---:|---:|---:|---:|
| 20% | 0.6862 | 0.7624 | 0.6033 | 0.6781 | 0.7769 | 26 697 |
| 35% | 0.7435 | **0.8058** | 0.6613 | 0.7001 | 0.7464 | 25 606 |
| 50% | 0.8001 | **0.8483** | 0.7445 | 0.7306 | 0.7207 | 24 599 |
| 75% | 0.8628 | **0.8885** | 0.8516 | 0.7807 | 0.7221 | 23 159 |
| 100% | 0.9034 | **0.9204** | 0.9048 | 0.8372 | 0.7807 | 22 522 |

AP tăng đều theo cutoff (+0.158 từ 20% đến 100%). Precision tăng mạnh hơn Recall: 20% P=0.60 → 100% P=0.90; Recall dao động 0.72–0.78. ECE giảm 0.069 (20%) → 0.020 (100%). 100% không dùng cho khuyến nghị.

Macro OULAD 5 mốc: AP 0.8451, Acc 0.7992, F1 0.7453, Rec 0.7493. Macro early (20–75): AP 0.8262.

### 3.3.5. So sánh baseline cùng protocol serving

Roster phục vụ: Hybrid CNN–BiLSTM, LR, DT, RF, SVM, MLP. **Một checkpoint Hybrid** chấm mọi mốc. XGBoost không nằm roster serving. Số dưới là trung bình 3×3 (`uci_final.csv`, `oulad_final.csv`) — cùng protocol FIT/STOP/VALID, không phải run đẹp nhất.

**UCI — AP**

| Mô hình | S0 | S1 | S2 |
|---|---:|---:|---:|
| Hybrid CNN–BiLSTM | 0.4547 | **0.8214** | **0.9101** |
| LR | 0.4754 | 0.7794 | 0.8812 |
| RF | **0.4995** | 0.7895 | 0.9072 |
| SVM | 0.4970 | 0.7936 | 0.8866 |
| DT | 0.4169 | 0.7330 | 0.8547 |
| MLP | 0.4486 | 0.7595 | 0.8778 |

**UCI — F1 tại t STOP**

| Mô hình | S0 | S1 | S2 |
|---|---:|---:|---:|
| Hybrid CNN–BiLSTM | 0.4291 | 0.6899 | **0.8010** |
| LR | 0.4703 | 0.6999 | 0.7988 |
| RF | 0.5073 | 0.7043 | 0.7888 |
| DT | **0.5214** | **0.7068** | 0.7544 |
| SVM | 0.4795 | 0.6807 | 0.7463 |
| MLP | 0.4506 | 0.6614 | 0.7728 |

S1 và S2 Hybrid đứng đầu AP. Macro UCI thua RF vì **S0** (RF AP 0.4995 vs 0.4547) — đúng khi chưa có chuỗi điểm: CNN/BiLSTM tắt, tabular UCI không có lợi thế inductive. F1 S2 Hybrid 0.8010 ≥ LR 0.7988 ≥ RF 0.7888. F1 S0 Hybrid thấp vì Precision 0.29 (bắt nhiều dương tính giả khi chưa có điểm).

**OULAD — AP**

| Mô hình | 20 | 35 | 50 | 75 | 100 |
|---|---:|---:|---:|---:|---:|
| Hybrid CNN–BiLSTM | 0.7624 | **0.8058** | **0.8483** | **0.8885** | **0.9204** |
| LR | **0.7632** | 0.7986 | 0.8399 | 0.8828 | 0.9114 |
| RF | 0.7522 | 0.7940 | 0.8402 | 0.8847 | 0.9154 |
| SVM | 0.7534 | 0.7835 | 0.8257 | 0.8723 | 0.9018 |
| DT | 0.7084 | 0.7548 | 0.7954 | 0.8530 | 0.8862 |
| MLP | 0.6799 | 0.7388 | 0.7998 | 0.8556 | 0.8964 |

**OULAD — F1 tại t STOP**

| Mô hình | 20 | 35 | 50 | 75 | 100 |
|---|---:|---:|---:|---:|---:|
| Hybrid CNN–BiLSTM | 0.6781 | **0.7001** | **0.7306** | **0.7807** | **0.8372** |
| LR | **0.6824** | 0.6930 | 0.7300 | 0.7756 | 0.8211 |
| RF | 0.6732 | 0.6904 | 0.7253 | 0.7794 | 0.8295 |
| SVM | 0.6759 | 0.6746 | 0.7111 | 0.7669 | 0.8176 |
| DT | 0.6413 | 0.6605 | 0.6904 | 0.7618 | 0.8138 |
| MLP | 0.6163 | 0.6356 | 0.6823 | 0.7465 | 0.8068 |

Hybrid đứng nhất AP từ 35% đến 100%; 20% thua LR 0.0008. F1 cùng hình: 20% thua LR 0.004, từ 35% trở đi Hybrid ≥ mọi baseline serving. Accuracy 100% Hybrid 0.9034 vs RF 0.9010 vs LR 0.8962.

Bảng 3.3.5 là so sánh **serving** (Phase 4, roster không có XGB). Bảng 3.3.6 là đối sánh kiến trúc trên **cùng tensor Hybrid**.

### 3.3.6. Một-trọng-số trên cùng tensor Hybrid

Cùng rule với Hybrid: **một estimator / family** chấm mọi mốc; ngưỡng `t` STOP theo mốc. Đặc trưng = tensor Hybrid (static + aggregate + temporal có mask + progress). **Không** last/mean/max/std/slope. Optuna warm-macro AP (40 trial UCI / 28 OULAD; RF OULAD khóa sau 15 trial vì TPE lặp 800 cây). Trung bình 3×3. Outer không dùng. Số Hybrid **cùng run** với baseline (khác bảng serving 3.3.4–3.3.5). Nguồn: `PARITY_ONE_WEIGHT.md`.

XGB/CatBoost **không** phục vụ; chỉ là trần tensor-parity.

**UCI — AP**

| Mô hình | S0 | S1 | S2 |
|---|---:|---:|---:|
| Hybrid CNN–BiLSTM | 0.4559 | **0.8110** | **0.9132** |
| LR | 0.4234 | 0.7687 | 0.8955 |
| RF | **0.4796** | 0.7080 | 0.8494 |
| XGB | 0.4503 | 0.7278 | 0.8469 |
| SVM | 0.4299 | 0.7390 | 0.7593 |
| CatBoost | 0.4463 | 0.7090 | 0.8199 |
| DT | 0.4298 | 0.6700 | 0.7667 |
| MLP | 0.4337 | 0.5381 | 0.6460 |

**OULAD — AP**

| Mô hình | 20 | 35 | 50 | 75 | 100 |
|---|---:|---:|---:|---:|---:|
| Hybrid CNN–BiLSTM | 0.7469 | **0.8054** | **0.8524** | 0.8929 | **0.9190** |
| XGB | **0.7661** | 0.8027 | 0.8512 | **0.8935** | 0.9187 |
| CatBoost | 0.7618 | 0.7988 | 0.8476 | 0.8900 | 0.9143 |
| LR | 0.7556 | 0.7958 | 0.8415 | 0.8853 | 0.9166 |
| RF | 0.7536 | 0.7920 | 0.8448 | 0.8877 | 0.9129 |
| SVM | 0.7291 | 0.7832 | 0.8250 | 0.8767 | 0.9050 |
| MLP | 0.7025 | 0.7539 | 0.8052 | 0.8645 | 0.9068 |
| DT | 0.7049 | 0.7459 | 0.7984 | 0.8476 | 0.8745 |

UCI: Hybrid dẫn AP ở S1 (+0.042 vs LR) và S2 (+0.018 vs LR). S0 thua RF 0.480 vs 0.456 — không có chuỗi điểm.

OULAD: 20% thua XGB 0.019. Từ 35% đến 100%, |Δ AP| so với XGB ≤ 0.003 (hòa trong nhiễu 3×3). F1 75–100% Hybrid 0.792 / 0.837 ≈ XGB 0.789 / 0.836. **Không** viết vượt trội kiến trúc trên OULAD với trần này. Mô hình phục vụ vẫn Hybrid CNN–BiLSTM.

---

## 3.4. Recommendation V

### 3.4.1. Luồng

```
Hybrid CNN–BiLSTM → PredictionResult (p, t, ŷ, H₂)
  → chỉ OULAD 20/35/50/75 (100% bị từ chối)
  → định tuyến risk quanh t và H₂
  → feasibility cứng 5 hành động
  → năm EBM, mỗi cái ℝ¹⁷ → s ∈ [0,1]
  → an toàn Top-1 (điểm, margin)
  → RECOMMEND Top-1  hoặc  HUMAN_REVIEW Top-3
  → kế hoạch xác định, không LLM lúc chạy
```

Code: `src/recommend_hybrid/` (implementation), hợp đồng công khai **Recommendation V**. Không refit Hybrid. Không đọc class CNN/LSTM. Adapter chỉ lấy `PredictionResult.recommendation_features()`.

Định tuyến risk (`risk_router`): nếu `p < t` → không tự động (INSUFFICIENT_EVIDENCE, lý do dưới ngưỡng serving). Nếu `H₂(p) > 0.70` hoặc `(p − t) < 0.05` → HUMAN_REVIEW. Còn lại vào feasibility + EBM.

An toàn sau ranking: Top-1 `s < 0.45` → INSUFFICIENT_EVIDENCE; `H₂ > 0.85` hoặc margin Top-1−Top-2 quá nhỏ → HUMAN_REVIEW; không thì RECOMMEND.

### 3.4.2. Hành động, feasibility, 17 đặc trưng EBM

Năm hành động: `ASSESSMENT_COMPLETION`, `RECOVER_ENGAGEMENT`, `STUDY_REGULARITY`, `TARGETED_CONTENT_REVIEW`, `QUIZ_RETRIEVAL_PRACTICE`.

Ngưỡng feasibility cứng (không học):

| Hành động | Điều kiện eligible | Chặn |
|---|---|---|
| ASSESSMENT_COMPLETION | `missing>0` hoặc `due_soon>0` | không còn gap |
| RECOVER_ENGAGEMENT | `active_day_rate < 0.5` và có VLE | engagement đã đủ |
| STUDY_REGULARITY | `regularity < 0.8` hoặc `active_day_rate < 0.8` | đã đều |
| TARGETED_CONTENT_REVIEW | `content_coverage < 0.8`, **không** ở 20% | quá sớm / coverage đủ |
| QUIZ_RETRIEVAL_PRACTICE | `quiz_available` | không có quiz |

Bốn trạng thái: `RECOMMEND`, `HUMAN_REVIEW`, `INSUFFICIENT_EVIDENCE`, `NO_FEASIBLE_ACTION`. Invalid-action trên Panel C: **0**.

Mười bảy cột EBM (cấm `action_id`, `final_result`, weak-label): `risk_probability`, `uncertainty`, `risk_margin`, `course_progress`, `inactivity_streak`, `active_day_rate`, `assessments_due`, `regularity_score`, `content_coverage`, `quiz_activity`, `missing_assessment_count`, `due_soon_count`, `completion_rate`, `vle_available`, `study_material_available`, `quiz_available`, `stage`. Mỗi hành động một EBM riêng — không một mô hình đa lớp dùng `action_id`.

Kế hoạch sau Top-1 là template xác định (thời hạn, tần suất, điều kiện an toàn). Gemini **không** sinh câu lúc serving.

### 3.4.3. Panel C held-out

632 case, 150 sinh viên, 2398 review (Gemini chỉ weak label lúc xây ranking; prompt đóng băng; không gán Risk, không chọn kiến trúc Hybrid). Panel C **không** dùng để tune.

| Mô hình | NDCG@3 | P@1 | MRR | R@3 | invalid |
|---|---:|---:|---:|---:|---:|
| Recommendation V | **0.88785** | 0.99206 | 0.99603 | 0.79947 | 0 |
| B1 rule score | 0.86649 | 0.99683 | 0.99841 | 0.80357 | 0 |
| B0 action+stage | 0.81889 | 0.99365 | 0.99683 | 0.78981 | 0 |

Δ NDCG@3 vs B1: +0.0213, bootstrap 2000, 95% CI [0.0144, 0.0282], P(Δ>0)=1.00.

Trên 632 case: RECOMMEND 94 (14.9%), HUMAN_REVIEW 175 (27.7%), INSUFFICIENT_EVIDENCE 363 (57.4%), NO_FEASIBLE_ACTION 0. Top-1 đủ năm hành động (RECOVER_ENGAGEMENT 111, QUIZ 64, CONTENT 36, REGULARITY 31, ASSESSMENT 27) — không collapsed một action.

### 3.4.4. Ràng buộc diễn giải

`p` là nguy cơ nhị phân **tại một mốc thông tin**, không phải nguyên nhân trượt. Recommendation V xếp **hành động khả thi** theo relevance đã học, không phải ATE. Cấm viết: can thiệp này làm tăng `G3` hay đổi `final_result`. Gemini chỉ weak label lúc xây ranking.

---

## 3.5. Luồng hệ thống

### 3.5.1. Một trường hợp OULAD 20%

Sinh viên `id_student`, môn `CCC`, kỳ `2014B`, cutoff = 20% × `module_presentation_length`.

1. Đọc `studentInfo` + `studentRegistration` + `courses` → static (one-hot + scale FIT đã khóa).
2. `studentVle` tuần có `event_time < cutoff` → `temporal` 11 kênh, mask; tuần sau cutoff không tồn tại trên tensor.
3. Assessment hạn trước cutoff → 13 aggregate; `aggregate_available=1`.
4. Checkpoint Hybrid CNN–BiLSTM OULAD (cùng trọng số cho 20–100%) → logit `z` → `p=σ(z)`, `H₂(p)`. `t` lấy từ STOP đã khóa của mốc 20%. `ŷ = [p ≥ t]`.
5. `100%` bị từ chối trước Recommendation V. Ở 20%: nếu `p < t` → INSUFFICIENT_EVIDENCE. Nếu margin/`H₂` xấu → HUMAN_REVIEW. Nếu `TARGETED_CONTENT_REVIEW` — ineligible vì quá sớm.
6. EBM trên 17 cột → điểm 5 hành động khả thi → an toàn Top-1. RECOMMEND kèm kế hoạch 3–14 ngày; HUMAN_REVIEW kèm Top-3.
7. PostgreSQL: `raw` → `catalog` → `prediction` → `recommendation`. `python project.py db predict|recommend` không train lại.

UCI S2 tương tự: static context + chuỗi `(G1/20, G2/20)` → một checkpoint UCI → `p`; **không** gọi Recommendation V (module chỉ OULAD).

### 3.5.2. Giới hạn chương

**Đã kiểm soát:** nhãn nhị phân tường minh; cấm G3/`final_result`/`score`; FIT-only scale; group-split; STOP-only `t`; outer không chọn mô hình; một kiến trúc hai miền; lệch lớp bằng `pos_weight` (SMOTE tensor: thử, không chọn); Recommendation V không nhân quả; invalid-action Panel C = 0.

**Cần nói rõ khi viết học thuật:**

- S0/20% Hybrid không vượt RF/LR — thiếu chuỗi; CNN/BiLSTM tắt khi `T=0`.
- Siêu tham số dataset-specific (lr, dropout, batch) là ngoại lệ quy mô, không phải hai kiến trúc.
- AP UCI và AP OULAD không so trực tiếp (khác prevalence).
- 100% OULAD không phải cảnh báo sớm (confounder length–Withdrawn AP 0.991).
- Precision/Recall/F1 tại một `t` không đồng thời cực đại; AP mới là chỉ số xếp hạng.
- Cổng Phase 4 “thắng cả hai miền” từng trả NOT_READY; mô hình cuối là lựa chọn của chủ đề, không phải rewrite cổng đó, và không bịa outer test.
- Trên tensor-parity một-trọng-số: Hybrid thắng AP UCI S1/S2; OULAD từ 35% hòa XGB (|Δ|≤0.003), 20% thua XGB. Không tuyên bố vượt trội OULAD.
- Recommendation V xếp hành động khả thi, không ước lượng tác động lên `final_result`.
