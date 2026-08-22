# Chương 3. Phân tích và thiết kế hệ thống

Đối tượng nghiên cứu chính của đề tài là mô hình **Hybrid CNN–BiLSTM**, dùng để dự đoán nguy cơ học tập nhị phân trên hai miền dữ liệu độc lập UCI và OULAD, kết hợp **Recommendation V** để xếp hạng hành động hỗ trợ trên OULAD. Các mô hình học máy cổ điển (hồi quy logistic, cây quyết định, rừng ngẫu nhiên, SVM, mạng perceptron đa lớp) chỉ đóng vai trò bộ so sánh cùng giao thức, không phải mô hình khóa của khóa luận.

Chương này trình bày phân tích dữ liệu đầu vào, quy trình tiền xử lý, kiến trúc mô hình đề xuất, cấu hình và quy trình huấn luyện, cách đóng gói phục vụ, và thiết kế Recommendation V. **Toàn bộ số liệu hiệu suất, biểu đồ thực nghiệm và nhận xét kết quả được trình bày ở Chương 4.**

---

## 3.1. Phân tích dữ liệu đầu vào

Trước khi xây dựng mô hình, việc phân tích và tìm hiểu sâu về dữ liệu đầu vào là một bước vô cùng quan trọng. Bước này giúp đảm bảo các đặc trưng được lựa chọn là phù hợp, không rò rỉ nhãn hay thông tin tương lai, và có ý nghĩa đối với bài toán cảnh báo sớm.

### 3.1.1. Mô tả bộ dữ liệu

Đề tài sử dụng **hai bộ dữ liệu độc lập**, không gộp thành một tập huấn luyện chung. Cùng một class `Hybrid` nhận tensor thống nhất; sự khác nhau giữa hai miền nằm ở chiều đầu vào, bộ scaler FIT-only và trọng số đã học.

**UCI Student Performance (Cortez & Silva, 2008).** Đề tài gộp hai file gốc Mathematics (395 dòng) và Portuguese (649 dòng) thành **1 044 bản ghi**, 33 cột gốc, định dạng CSV phân tách bằng dấu `;`. Mỗi bản ghi là một cặp (học sinh, môn) trong một học kỳ. Bản chất dữ liệu là **tĩnh theo học kỳ**: chuỗi điểm tối đa **T = 2** (G1 rồi G2).

Nhãn nhị phân được tạo một lần từ điểm cuối kỳ:

- `risk = 1` khi `G3 < 10`
- `risk = 0` khi `G3 ≥ 10`

Tỷ lệ lớp nguy cơ trên toàn bộ 1 044 bản ghi là **0.220** (230/1044). `G3` trung bình 11.34. Để tách nhóm khi chia fold, 13 trường quasi-identity (`school`, `sex`, `age`, `address`, `famsize`, `Pstatus`, `Medu`, `Fedu`, `Mjob`, `Fjob`, `reason`, `nursery`, `internet`) tạo **662** `global_student_group`; 366 nhóm xuất hiện ở cả hai môn.

Thông tin điểm giữa kỳ được đưa vào theo mốc, không phải lúc nào cũng có đủ:

- S0: chưa có `G1`/`G2`
- S1: đã có `G1`
- S2: đã có `G1` rồi `G2`

Chi tiết các nhóm thuộc tính UCI sau khi xác định vai trò trong mô hình được mô tả trong bảng dưới đây:

| Tên thuộc tính | Mô tả | Vai trò trong mô hình |
|---|---|---|
| `G3` | Điểm cuối kỳ, thang 0–20 | **Chỉ tạo nhãn** `risk = [G3 < 10]`. Không đưa vào predictor. |
| `G1`, `G2` | Điểm kỳ 1 / kỳ 2, thang 0–20 | Chuỗi temporal (chia 20). S1 chỉ có `G1`; S2 có `G1` rồi `G2`. View UCI còn điền 5 cột aggregate tóm tắt điểm **đã quan sát** tại mốc đó; S0 tắt aggregate. Không đưa `G1`/`G2` vào context tĩnh. |
| `failures` | Số lần trượt môn trước đó | Static numeric — tín hiệu nền mạnh nhất còn lại sau khi cấm điểm cuối kỳ. |
| `age`, `Medu`, `Fedu`, `traveltime`, `studytime`, `famrel`, `freetime`, `goout`, `Dalc`, `Walc`, `health` | Tuổi, học vấn phụ huynh, thời gian đi học/học, quan hệ gia đình, thời gian rảnh, đi chơi, rượu, sức khỏe | Static numeric — ngữ cảnh nền. |
| `school`, `sex`, `address`, `famsize`, `Pstatus`, `Mjob`, `Fjob`, `reason`, `guardian`, `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`, `romantic`, `subject` | Trường, giới, chỗ ở, gia đình, nghề phụ huynh, lý do chọn trường, hỗ trợ, hoạt động, môn | Static categorical — one-hot trên FIT. |
| `absences` | Số buổi vắng | **Cấm** — có thể đồng thời với kết quả, không phải đặc trưng cảnh báo sớm. |

**Bảng 3.1.** Mô tả các thuộc tính UCI sau khi xác định vai trò trong Hybrid CNN–BiLSTM.

**OULAD (Kuzilek, Hlosta & Zdrahal, 2017).** 32 593 enrollment, 28 785 sinh viên. Nhật ký `studentVle` gồm 10 655 280 dòng click. Bản chất dữ liệu là **tương tác theo thời gian**: mỗi enrollment là chuỗi tuần VLE cộng đặc trưng gộp tại cutoff.

Nhãn nhị phân:

- `risk = 1` nếu `final_result ∈ {Fail, Withdrawn}`
- `risk = 0` nếu Pass hoặc Distinction

Năm mốc cutoff: 20 / 35 / 50 / 75 / 100% chiều dài `module_presentation_length`. Sự kiện chỉ được lấy khi `observation_start ≤ event_time < cutoff`. Số bản ghi còn đủ điều kiện: 20% 26 697; 35% 25 606; 50% 24 599; 75% 23 159; 100% 22 522.

| Tên thuộc tính / kênh | Mô tả | Vai trò trong mô hình |
|---|---|---|
| `final_result` | Pass / Distinction / Fail / Withdrawn | **Chỉ tạo nhãn**. Không đưa vào predictor. |
| `score`, `date_unregistration` | Điểm bài kiểm tra; ngày hủy đăng ký | **Cấm** làm predictor. Enrollment có `date_unregistration < cutoff` bị loại khỏi mốc đó. |
| 11 kênh temporal / tuần | Cường độ hoạt động, ngày hoạt động, số site, loại hoạt động, content / forum / quiz, nộp bài, nộp trễ, tỷ lệ phơi nhiễm tuần | Chuỗi CNN ∥ BiLSTM, pad tới tuần dài nhất tại cutoff, có mask. |
| 13 số aggregate tại cutoff | Hoạt động cộng dồn, trung bình tuần, gần đây, xu hướng, chuỗi nghỉ, hạn / nộp / hoàn thành / trễ | Nhánh tabular aggregate; `aggregate_available = 1` khi có thống kê cutoff. |
| Context tĩnh | `gender`, `region`, `highest_education`, `imd_band`, `age_band`, `disability`, `code_module`, `presentation_season`; numeric `num_of_prev_attempts`, `studied_credits`, `registration_lead_time`, `module_presentation_length` | Static — one-hot + scale trên FIT. |
| Độ dài chuỗi quan sát | Số tuần đã thấy tới cutoff | **Không** dùng làm cột predictor. Tại 100% độ dài quan sát vẫn liên đới Withdrawn — diễn giải ở Chương 4. |

**Bảng 3.2.** Mô tả các kênh OULAD sau khi xác định vai trò trong Hybrid CNN–BiLSTM.

SHA-256 từng file gốc được khóa trong protocol (in-repo, không dump ngoài):

| File | SHA-256 (rút gọn) |
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

**Bảng 3.3.** Chữ ký SHA-256 (rút gọn) của các file dữ liệu gốc.

### 3.1.2. Phân tích tương quan và lựa chọn thuộc tính

Để hiểu rõ hơn mối quan hệ giữa các biến nền tảng và nhãn nguy cơ trên UCI, hệ số Spearman với nhãn `G3 < 10` (n = 1 044) được tính trên dữ liệu thô. Đây là mô tả dữ liệu trước khi FIT-scale, không phải thứ tự importance sau khi huấn luyện.

| Thuộc tính | Spearman | p | Đưa vào Hybrid? |
|---|---:|---:|---|
| `G3` | −0.722 | < 10⁻¹⁶⁰ | **Không** — chỉ tạo nhãn |
| `G2` | −0.675 | < 10⁻¹³⁹ | Temporal (và tóm tắt aggregate cùng mốc), chỉ S2 |
| `G1` | −0.628 | < 10⁻¹¹⁴ | Temporal (và tóm tắt aggregate cùng mốc), S1 và S2 |
| `failures` | +0.376 | 2.0×10⁻³⁶ | Static numeric |
| `age` | +0.128 | 3.5×10⁻⁵ | Static |
| `Fedu` / `studytime` / `goout` / `Medu` | ~±0.11 | < 10⁻³ | Static |
| `absences` | +0.052 | 0.091 | **Cấm** (đồng thời với kết quả) |

**Bảng 3.4.** Spearman với nhãn `G3 < 10` trên UCI (n = 1 044).

Dựa trên phân tích:

- **Tương quan nghịch với nhãn nguy cơ:** `G1`, `G2`, `G3` đều cho thấy tương quan nghịch mạnh: điểm càng cao thì xác suất `G3 < 10` càng thấp. `G3` bị loại khỏi predictor vì chính nó tạo nhãn. `G1`/`G2` được giữ nhưng **chỉ khi mốc đã quan sát được**, dưới dạng chuỗi temporal có mask (và tóm tắt aggregate cùng mốc), không đưa vào vector context tĩnh.
- **Tương quan thuận với nhãn nguy cơ:** `failures` là tín hiệu static mạnh nhất còn lại (+0.376). `age` và một số biến nền (`Fedu`, `studytime`, `goout`, `Medu`) có tương quan yếu hơn nhưng vẫn mang ngữ cảnh.
- **Tín hiệu bị cấm dù |ρ| nhỏ:** `absences` có |ρ| = 0.052 (p = 0.091). Việc cấm không dựa trên Spearman yếu, mà vì số buổi vắng có thể đồng thời với kết quả, không hợp với cảnh báo sớm.

**Lý do lựa chọn thuộc tính:**

Mặc dù một số tương quan static không quá mạnh, chúng vẫn cho thấy sự tồn tại của mối liên hệ giữa hoàn cảnh học tập và nguy cơ. Trong bài toán cảnh báo sớm, việc đưa vào các biến ngữ cảnh (exogenous) có liên quan cung cấp thêm thông tin cho nhánh tabular, trong khi tín hiệu mạnh `G1`/`G2` (UCI) và nhật ký VLE (OULAD) được để cho CNN và BiLSTM khai thác theo thời gian.

Do đó, đề tài quyết định:

- Giữ toàn bộ context tĩnh hợp lệ (categorical + numeric ở mục 3.1.1), trừ các trường đã cấm.
- Đưa `G1`/`G2` vào chuỗi temporal đúng mốc; không dùng `G3` làm đầu vào.
- Trên OULAD, khóa 11 kênh temporal / tuần và 13 số aggregate tại cutoff như đã liệt kê — không dùng `final_result`, `score`, `date_unregistration` hay độ dài chuỗi làm predictor.

---

## 3.2. Quy trình tiền xử lý dữ liệu

Để xây dựng một mô hình học sâu hiệu quả, việc chuẩn bị một bộ tensor sạch, cutoff-safe và được chuẩn hóa đúng cách là bước nền tảng. Quy trình tiền xử lý được thực hiện có hệ thống, từ file CSV gốc đến `UnifiedHybridData` sẵn sàng cho Hybrid CNN–BiLSTM.

### 3.2.1. Thu thập và khám phá dữ liệu

Quá trình tiền xử lý bắt đầu bằng việc đọc dữ liệu gốc trong `data/raw/`.

- **UCI:** bắt buộc đúng 395 dòng Mathematics + 649 dòng Portuguese; thêm cột `subject` (`math` / `portuguese`).
- **OULAD:** join `studentInfo`–`studentRegistration`–`courses`; log VLE và assessment được đọc riêng theo từng cutoff.

Các bước khảo sát ban đầu cho thấy:

- UCI đang ở dạng bản ghi học kỳ, chưa phải chuỗi tuần.
- OULAD `studentVle` ở dạng sự kiện click; phải gom theo tuần **trước** cutoff.
- Nhãn chỉ được tạo một lần từ `G3` hoặc `final_result`; các trường đó không được đưa vào predictor.

### 3.2.2. Tái cấu trúc và xử lý rò rỉ thời gian

Sau khi đọc file gốc, dữ liệu trải qua các bước biến đổi để có cấu trúc phù hợp cho mô hình chuỗi có mask, đồng thời loại rò rỉ thời gian.

**UCI:**

- Chuyển đổi nhãn: `target` từ `G3` một lần theo quy tắc `G3 < 10`.
- Tái cấu trúc định danh: `record_id` = hash ổn định (`subject`, chỉ số dòng, chữ ký identity); `global_student_group` = hash 13 trường quasi-identity.
- Tái cấu trúc chuỗi: bước 0 = `G1/20` (nếu S1/S2), bước 1 = `G2/20` (nếu S2); S0 mask toàn 0.
- Tóm tắt aggregate cùng mốc (5 chiều): S0 tắt (`aggregate_available = 0`); S1/S2 bật trên điểm **đã có** tại mốc đó. Đây không phải thông tin tương lai.
- Context tĩnh: one-hot / scale các cột categorical và numeric đã chọn; `G1`, `G2`, `G3`, `absences` không vào nhánh static.

**OULAD:**

- Gom VLE **theo tuần** với điều kiện `event_time < cutoff`.
- Assessment chỉ tính hạn / `date_submitted` **trước** cutoff.
- Enrollment có `date_unregistration < cutoff` bị loại khỏi mốc đó.
- 100%: 94 Withdrawn còn lại sau lọc — không dùng độ dài chuỗi làm proxy Withdrawn cho cảnh báo sớm.

Kết quả của giai đoạn này là hai bộ tensor cùng schema `UnifiedHybridData`, mỗi miền một scaler FIT-only, sẵn sàng cho kiến trúc chung.

### 3.2.3. Chuẩn hóa và phân chia dữ liệu

**Tensor thống nhất** (`UnifiedHybridData`):

`static [N, Ds]`, `temporal [N, T, C]`, `temporal_mask [N, T]`, `lengths [N]`, `aggregate [N, Da]`, `aggregate_available [N]`, `progress [N]`, `target`, `record_id`, `group_id`.

Ràng buộc: `lengths = sum(mask)`; ô temporal bị mask ≈ 0; `target ∈ {0,1}`; `progress ∈ [0,1]`.

| Miền | T | C | Da | progress |
|---|---:|---:|---:|---|
| UCI S0 / S1 / S2 | 2 | 1 (điểm/20) | 5 (tắt tại S0) | 0.00 / 0.50 / 1.00 |
| OULAD 20–100% | pad tới mốc dài nhất (tới ~39 tuần) | 11 | 13 | 0.20 … 1.00 |

**Bảng 3.5.** Kích thước tensor thống nhất theo miền.

**Chuẩn hóa dữ liệu (FIT-only):**

- Context tĩnh: one-hot + scale **chỉ trên FIT**.
- Temporal: `MaskedStandardScaler` — chỉ ô `mask = 1`.
- Aggregate: mean / std FIT, chỉ hàng `aggregate_available = 1`.
- STOP / VALID / outer **không** refit scaler. Đối tượng scaler sau khi fit được lưu kèm checkpoint để inference dùng đúng thống kê FIT.

**Phân chia biến và fold:**

- Biến độc lập: bộ tensor trên; biến phụ thuộc: `target` nhị phân.
- Group-split: UCI theo `global_student_group`, OULAD theo `id_student`.
- Outer 3 fold tồn tại nhưng **fold 0 outer là firewall** — không tune, không chọn kiến trúc, không chốt mô hình.
- Inner 3 fold trên phần còn lại: FIT / STOP / VALID. Seed split 42; seed train 42, 1201, 2026.

Hash split khóa: inner UCI `ad8f44e5…e02ae8`, inner OULAD `8559efcf…72650c`.

### 3.2.4. Kiến trúc mô hình đề xuất

Mô hình này kết hợp sức mạnh của nhánh tabular (static + aggregate) để mã hóa ngữ cảnh và thống kê tại cutoff, module CNN để trích mẫu cục bộ trên chuỗi có mask, và module BiLSTM để nắm bắt phụ thuộc hai chiều theo thời gian. Khác với kiến trúc CNN → BiLSTM nối tiếp thuần túy, Hybrid CNN–BiLSTM chạy **CNN song song với BiLSTM**, rồi trộn với nhánh tabular qua **cổng softmax 3 nhánh có mask**.

Kiến trúc tổng thể được minh họa ở **Hình 3.1** (CNN song song BiLSTM + nhánh tabular, cổng softmax 3 nhánh có mask). Sơ đồ ASCII dưới đây trùng nội dung hình:

```text
static ── ResidualProjector ─┐
aggregate ─ ResidualProjector ─┴── h_tab
                                      │
temporal ─ Linear+LN ─┬─ ResidualCNN ── h_cnn ─┐
  (× mask)            └─ BiLSTM ────── h_lstm ─┼─ cổng softmax 3 nhánh
                                               │    (nhánh tắt → logit −∞)
                                               └─ h = Σ g_i h_i
                                                    │
                                                    Head → logit z → p = σ(z)
```

![Hình 3.1. Kiến trúc Hybrid CNN–BiLSTM](figures/architecture_hybrid.png)

**Hình 3.1.** Hybrid CNN–BiLSTM: ResidualProjector tabular, CNN ∥ BiLSTM, cổng softmax có mask, head logit nhị phân.

Một class `Hybrid` (`src/prediction/model/hybrid.py`), `model_id = hybrid`, `display_name = "Hybrid CNN-BiLSTM"`. Một checkpoint UCI chấm S0–S2; một checkpoint OULAD chấm 20–100%. Không huấn luyện mô hình riêng cho mốc 100%.

Khi `lengths = 0` (S0 / chuỗi rỗng): CNN và BiLSTM **tắt**, chỉ tabular. Đây là hành vi thiết kế, không phải trường hợp lỗi.

### 3.2.5. Module CNN

Module này có nhiệm vụ như một bộ trích xuất đặc trưng tự động trên chuỗi temporal đã được adapter về 128 chiều và nhân mask.

- **Adapter temporal:** `Linear + LayerNorm` đưa kênh temporal về `d_fuse = 128`, sau đó nhân mask để ô không hợp lệ không đóng góp.
- **Chiếu kênh:** `Linear` xuống `cnn_channels = 64`.
- **ResidualCNNBranch:** hai block dư, kernel 2, dilation 1 rồi 2, dropout theo cấu hình miền. Mỗi block pad đối xứng theo dilation, hai lớp `Conv1d`, GELU, Dropout, cộng residual, rồi nhân mask.
- **Gộp:** masked mean–max → `Linear` → `h_cnn ∈ ℝ^128`.
- Nếu không có bước temporal hợp lệ (`lengths = 0`): `h_cnn = 0`.

### 3.2.6. Module Bi-LSTM

Dữ liệu chuỗi sau adapter 128 chiều (cùng nhánh với CNN) được đưa vào module BiLSTM để mô hình hóa phụ thuộc thời gian hai chiều.

- **Lớp BiLSTM** (`nn.LSTM` với `bidirectional=True`): hidden 128, **một** lớp, hai chiều. Chuỗi được `pack_padded_sequence` theo `lengths` để không học pad.
- **Gộp:** masked mean–max trên đầu ra 256 chiều (128 xuôi + 128 ngược) tạo vector 512 chiều, rồi `Linear` → `h_lstm ∈ ℝ^128`.
- **Dropout** nằm trên projector / head / cổng theo `dropout` miền, không xếp lớp BiLSTM thứ hai — khác với một số kiến trúc hai lớp BiLSTM xếp chồng.
- Cùng quy tắc tắt khi `lengths = 0`: `h_lstm = 0`.

Việc dùng kiến trúc hai chiều cho phép tại mỗi bước thời gian học thông tin từ cả quá khứ (hướng xuôi) và ngữ cảnh tương lai trong **cửa sổ đã quan sát** (hướng ngược trên chuỗi đã cắt tại cutoff), không nhìn sự kiện sau cutoff.

### 3.2.7. Module fusion và đầu ra

Đây là module cuối cùng của mô hình, có nhiệm vụ tổng hợp các đặc trưng bậc cao từ ba nhánh và đưa ra xác suất nguy cơ.

**Nhánh tabular (trước cổng):**

- `ResidualProjector` trên `static` → `h_static ∈ ℝ^128`.
- `ResidualProjector` trên `aggregate` → nhân `aggregate_available` → cộng vào `h_static` tạo `h_tab`.
- Projector: shortcut tuyến tính + nhánh sâu (Linear → LayerNorm → GELU → Dropout → Linear), cộng residual, LayerNorm.

**Cổng softmax 3 nhánh:**

Đầu vào cổng: `[h_tab, h_cnn, h_lstm, a_tab = 1, a_cnn, a_lstm, progress]`.

- `a_cnn = a_lstm = 1[lengths > 0]`.
- Logit nhánh tắt được gán −∞, rồi softmax 3 nhánh — nhánh không available nhận khối lượng 0.
- `h = g_tab h_tab + g_cnn h_cnn + g_lstm h_lstm`.
- Entropy-floor (hệ số UCI 0.002, OULAD 0.005) phạt cổng quá chắc khi nhiều nhánh available; đây là số hạng phụ, không thay BCE.

**Head (tương đương khối fully connected của mô hình hồi quy):**

LayerNorm → Linear 128 → GELU → Dropout → Linear 1 → logit `z`, `p = σ(z)`.

Output serving: `p`, ngưỡng `t` (chọn trên STOP, mục 3.3), `ŷ = [p ≥ t]`, bất định `H₂(p)`. Hợp đồng `PredictionResult` — Recommendation V không đọc vector CNN/LSTM nội bộ.

### 3.2.8. Cấu hình chi tiết mô hình

Bảng dưới đây trình bày cấu trúc từng khối của mô hình được triển khai trong thực tế, với các thông số khóa trong `configs/prediction/hybrid_final.json` và class `Hybrid`.

| # | Khối | Thiết lập | Ghi chú |
|---|---|---|---|
| 1 | ResidualProjector (static) | vào `Ds` → 128 | Shortcut + residual GELU |
| 2 | ResidualProjector (aggregate) | vào `Da` → 128, × `aggregate_available` | Cộng vào `h_tab` |
| 3 | Temporal adapter | Linear `C` → 128 + LayerNorm | Nhân mask |
| 4 | CNN projection | Linear 128 → 64 | |
| 5 | ResidualCNNBranch | 2 block, kernel 2, dilation (1, 2), 64 kênh | Mask-safe |
| 6 | CNN out | Linear 128 → 128 | Sau masked mean–max |
| 7 | BiLSTM | hidden 128, 1 lớp, bidirectional | `pack_padded_sequence` |
| 8 | LSTM out | Linear 512 → 128 | 4 × hidden sau mean–max |
| 9 | Cổng | Linear 388 → 64 → GELU → Dropout → 3 | 128×3 + 3 availability + progress |
| 10 | Head | LN → 128 → GELU → Dropout → 1 | Logit nhị phân |
| | Tham số (checkpoint OULAD serving) | **482 116** | Cùng topology UCI; khác `Ds`/`C`/`Da` |

**Bảng 3.6.** Cấu hình chi tiết Hybrid CNN–BiLSTM.

Phân tích chi tiết từ bảng:

- **Mask và availability:** CNN/BiLSTM nhận mask từng bước; cổng nhận cờ available theo nhánh. Khi `T = 0`, hai nhánh temporal bị tắt hoàn toàn.
- **Tham số:** Phần lớn dung lượng nằm ở projector residual, BiLSTM và cổng/head. Checkpoint OULAD serving có 482 116 tham số — đủ nhỏ để inference trên GPU Turing 6 GB, đồng thời đủ lớn hơn một hồi quy tuyến tính trên cùng tensor.
- **Một topology, hai miền:** `d_fuse`, CNN, BiLSTM, fusion không đổi. Khác dataset chỉ `lr`, `dropout`, `batch_size`, `pos_weight_multiplier` và chiều đầu vào (mục 3.3.1).
- Bảng này khẳng định lại kiến trúc lai CNN ∥ BiLSTM + tabular đã mô tả và cho thấy cách các module được hiện thực hóa bằng PyTorch.

### 3.2.9. Công thức toán của mô hình (thiết kế)

Các ký hiệu và công thức BCE, AP, \(H_2(p)\), softmax cổng được viết đầy đủ ở **Chương 2 mục 2.6**. Tóm tắt vận hành:

- Huấn luyện: BCE with logits, `pos_weight` chỉ trên FIT.
- Suy diễn: \(p=\sigma(z)\), \(\hat y=\mathbf{1}[p\ge t]\), \(t\) chọn trên STOP.
- Cổng: nhánh `lengths=0` bị mask \(-\infty\) trước softmax; \(h=\sum g_k h_k\).
- Recommendation V chỉ nhận \((p,t,\hat y,H_2)\), không nhận \(h_{\mathrm{cnn}}\).

---

## 3.3. Quy trình huấn luyện mô hình

Quy trình huấn luyện được thiết kế để một checkpoint / miền chấm mọi mốc thông tin, early-stop trên STOP, và không nhìn outer test khi chốt mô hình.

### 3.3.1. Cấu hình huấn luyện

- **Thiết Kế Kiến Trúc Mô Hình**
  - Thiết kế Hybrid CNN–BiLSTM: ResidualProjector tabular; CNN song song với BiLSTM; cổng softmax 3 nhánh có mask.
  - Một class `Hybrid` cho cả UCI và OULAD; khác nhau chỉ chiều input, scaler FIT-only và trọng số.
  - Tắt CNN/BiLSTM khi `lengths = 0` (S0 / chuỗi rỗng).
  - Không huấn luyện mô hình riêng cho OULAD 100%.
- **Lựa Chọn Hàm Tối Ưu Và Mất Mát**
  - Chọn AdamW cho việc học ổn định trên hai miền khác quy mô.
  - Dùng Binary Cross-Entropy with logits, cost-sensitive: `pos_weight_FIT = (n_neg / n_pos)_FIT × hệ_số`.
  - UCI hệ số 1.183; OULAD hệ số 0.779. Cùng công thức mọi fold; hệ số và `(n_neg / n_pos)` chỉ tính trên FIT.
  - SMOTE / ADASYN trên tensor: thử FIT-only như phương án cân bằng lớp, **không chọn** — nội suy không tạo G1/G2 hay tuần VLE thật.
  - Entropy-floor trên cổng là số hạng phụ (UCI 0.002, OULAD 0.005), không thay BCE.
- **Thiết Lập Tham Số Huấn Luyện**
  - UCI: `lr = 8.61×10⁻⁵`, `weight_decay = 3.29×10⁻³`, `dropout = 0.406`, `batch = 32`.
  - OULAD: `lr = 1.18×10⁻⁴`, `weight_decay = 7.11×10⁻⁴`, `dropout = 0.320`, `batch = 128`.
  - Seed train: 42, 1201, 2026. Seed split: 42.
  - Early-stop trên **STOP macro AP** (`sklearn.metrics.average_precision_score`).
- **Cấu Hình Hardware Và Environment**
  - Tự phát hiện CUDA; AMP FP16 + GradScaler trên RTX 2060 (Turing; TF32 tắt vì không phải Ampere).
  - DataLoader shuffle trên FIT; `eval()` trên STOP/VALID để tắt dropout.
  - Serving đọc xác suất đã lưu, không train lại.
- **Chiến Lược Lưu Trữ Mô Hình**
  - Theo dõi STOP macro AP làm metric lưu checkpoint.
  - Lưu `state_dict` theo fold, kèm scaler FIT-only.
  - Ngưỡng `t`: lưới trên STOP, xếp F1 rồi recall rồi `|t − 0.5|`. VALID không chọn `t`.
  - Outer test không tham gia lưu hay chọn kiến trúc.

### 3.3.2. Quy trình huấn luyện

- **Khởi Tạo Phân Chia**
  - Group-disjoint inner 3 fold trên phần không thuộc outer firewall.
  - Mỗi fold tách FIT / STOP / VALID theo nhóm: UCI `global_student_group`, OULAD `id_student`.
  - Hash split khóa như mục 3.2.3.
- **Vòng Lặp Huấn Luyện Chính**
  - Với mỗi cặp (fold, seed), khởi tạo mô hình mới từ đầu theo `HybridConfig`.
  - Fit scaler và `pos_weight` **chỉ trên FIT**.
  - Tạo DataLoader riêng cho FIT (train) và STOP (early-stop).
- **Quy Trình Một Epoch**
  - Giai đoạn Training: `train()` , bật gradient, lặp batch FIT: chuyển tensor lên GPU, forward, BCE with logits (có `pos_weight`), backward, bước AdamW.
  - Giai đoạn STOP: `eval()`, tắt gradient, tính macro AP trên STOP; lưu checkpoint nếu cải thiện; early-stop theo số epoch chờ đã cấu hình.
- **Đánh Giá Và Giám Sát (giao thức, chưa phải bảng số)**
  - Chỉ số chính khi dừng: AP trên STOP.
  - Acc / Precision / F1 / Recall tại `t` đã chọn trên STOP, áp lên VALID.
  - F1 là trung hòa điều hòa của Precision và Recall — một `t` không tối đa đồng thời cả ba.
  - Lặp 3 fold × 3 seed; báo cáo **trung bình 9 số** ở Chương 4. Không lấy run đẹp nhất, không chọn fold theo VALID.
- **Lưu Trữ Phục Vụ**
  - Checkpoint UCI (S0–S2) và OULAD (20–100%).
  - OOF 20–75% cho Recommendation V: 3 inner fold, seed 42.
  - Không joint-train hai miền.
  - Không mở outer khi chốt mô hình.
- **Validation Và Inference Sau Huấn Luyện**
  - Load `state_dict` + scaler FIT; `eval()`.
  - Forward cho ra `p`; so với `t` fold/mốc đã khóa.
  - Gói `PredictionResult` (`p`, `t`, `ŷ`, `H₂`) cho Recommendation V.

---

## 3.4. Triển khai dự đoán Hybrid CNN–BiLSTM

Để đưa mô hình đã huấn luyện vào vận hành trên dữ liệu enrollment, đề tài đóng gói Hybrid CNN–BiLSTM thành chuỗi lệnh PostgreSQL (`raw` → `catalog` → `prediction`), không huấn luyện lại tại lúc phục vụ.

### 3.4.1. Đóng gói mô hình đã huấn luyện

- **Tải Và Khởi Tạo Mô Hình**
  - Khởi tạo kiến trúc `Hybrid` với `HybridConfig` (`d_fuse = 128`, CNN 64 kênh, BiLSTM hidden 128).
  - Nạp trọng số đã khóa (`state_dict`); `eval()` — tắt dropout.
  - Tự phát hiện CUDA/CPU; bản phục vụ DB đọc hàng xác suất đã materialize, không bắt buộc forward GPU từng request.
- **Tải Scaler Và Tiền Xử Lý**
  - Scaler FIT-only đi kèm miền (UCI / OULAD).
  - Không refit min/max hay mean/std khi `python project.py db predict`.
  - Cutoff và mask được áp dụng giống lúc huấn luyện.
- **Phụ Thuộc**
  - PyTorch cho kiến trúc và (nếu cần) forward.
  - scikit-learn / pandas / numpy cho tiền xử lý đã khóa.
  - psycopg2 cho PostgreSQL.
  - python-dotenv đọc `.env` (`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`).

### 3.4.2. Cấu trúc lệnh phục vụ

Chuỗi PostgreSQL: `raw` → `catalog` → `prediction` → `recommendation`.

```text
Base: python project.py db

predict:
  python project.py db predict --student 631334 --course CCC --presentation 2014B --stage 20

recommend:
  python project.py db recommend --student 631334 --course CCC --presentation 2014B --stage 20
```

- **Đầu vào predict:** định danh sinh viên / môn / kỳ / mốc (20, 35, 50, 75, 100).
- **Đầu ra predict:** `p`, `t`, `ŷ`, `H₂(p)`, `enrollment_id`, `prediction_id`.
- Recommendation V nhận đúng `PredictionResult` đó (mục 3.5), không đọc CNN/LSTM.

### 3.4.3. Đặc điểm kỹ thuật

- **Đặc Tính Vận Hành**
  - Inference phục vụ đọc hàng `prediction.prediction` (OOF đã materialize).
  - `t` lấy theo mốc đã khóa trên STOP, không chỉnh trên VALID hay outer.
  - Clickstream `studentVle` không copy vào Postgres (giữ DB tối giản; tensor VLE được dựng lúc materialize).
- **Phạm Vi Recommendation**
  - Recommendation V chỉ OULAD 20 / 35 / 50 / 75.
  - 100% bị từ chối trước module khuyến nghị — không phải cảnh báo sớm.
- **An Toàn Giao Thức**
  - Không train lại khi predict/recommend.
  - Schema `optuna_hs_v2` / `research` không thuộc bản phục vụ.

### 3.4.4. Ví dụ sử dụng

Yêu cầu: sinh viên `631334`, môn `CCC`, kỳ `2014B`, mốc 20%.

```text
python project.py db predict --student 631334 --course CCC --presentation 2014B --stage 20
```

Phản hồi gồm: `p`, `t`, `ŷ`, `H₂(p)`, `enrollment_id`, `prediction_id`. Recommendation V nhận đúng `PredictionResult` đó. Số liệu một lần chạy cụ thể được ghi ở Chương 4. Đề tài **không** xây giao diện người dùng; phạm vi là mô hình và đánh giá.

---

## 3.5. Xây dựng Recommendation V

Recommendation V là lớp xếp hạng hành động hỗ trợ, đọc **chỉ** `PredictionResult` của Hybrid CNN–BiLSTM. Module này không ước lượng hiệu ứng nhân quả lên `final_result`, không refit Hybrid, và không gọi LLM lúc chạy.

### 3.5.1. Kiến trúc

Công nghệ stack:

- **Đầu vào:** `PredictionResult` (`p`, `t`, `ŷ`, `H₂`) + bằng chứng cutoff-safe (VLE / assessment).
- **Định tuyến rủi ro:** so `p` với `t` đã khóa; trần bất định và biên mỏng.
- **Feasibility:** luật cứng, không học, năm hành động.
- **Xếp hạng:** năm EBM (`interpret`), mỗi hành động một mô hình, đầu vào ℝ¹⁷ → `s ∈ [0, 1]`.
- **Kế hoạch:** template xác định (thời hạn, tần suất, điều kiện an toàn).
- **Lưu trữ:** PostgreSQL `recommendation.recommendation` + `recommendation_item`.
- **LLM:** Gemini chỉ dùng lúc xây weak label ranking; không gán Risk, không chọn kiến trúc Hybrid, không chạy lúc serving.

Luồng:

```text
Hybrid CNN–BiLSTM → PredictionResult (p, t, ŷ, H₂)
  → chỉ OULAD 20 / 35 / 50 / 75
  → định tuyến quanh t và H₂
  → feasibility cứng 5 hành động
  → năm EBM, mỗi cái ℝ¹⁷ → s ∈ [0, 1]
  → RECOMMEND Top-1 hoặc HUMAN_REVIEW Top-3
  → kế hoạch xác định, không LLM lúc chạy
```

Code phục vụ: `src/recommend_hybrid/v3/`. Không refit Hybrid.

### 3.5.2. Các chức năng chính

**Năm hành động chuẩn:**

`ASSESSMENT_COMPLETION`, `RECOVER_ENGAGEMENT`, `STUDY_REGULARITY`, `TARGETED_CONTENT_REVIEW`, `QUIZ_RETRIEVAL_PRACTICE`.

**Định tuyến risk (trước feasibility):**

- `p < t` → không tự động (`INSUFFICIENT_EVIDENCE` / không phát hành Top-1).
- `H₂(p) > 0.70` hoặc `(p − t) < 0.05` → `HUMAN_REVIEW`.
- Còn lại vào feasibility + EBM.

**Feasibility cứng (không học):**

| Hành động | Eligible khi | Chặn |
|---|---|---|
| ASSESSMENT_COMPLETION | `missing > 0` hoặc `due_soon > 0` | không còn gap |
| RECOVER_ENGAGEMENT | `active_day_rate < 0.5` và có VLE | engagement đã đủ / không VLE |
| STUDY_REGULARITY | `regularity < 0.8` hoặc `active_day_rate < 0.8` | đã đều / thiếu bằng chứng |
| TARGETED_CONTENT_REVIEW | `content_coverage < 0.8`, không ở 20% | quá sớm / coverage đủ / không tài liệu |
| QUIZ_RETRIEVAL_PRACTICE | `quiz_available` | không có quiz |

**Bảng 3.7.** Luật feasibility cứng của Recommendation V.

**Bốn trạng thái phát hành:** `RECOMMEND`, `HUMAN_REVIEW`, `INSUFFICIENT_EVIDENCE`, `NO_FEASIBLE_ACTION`.

**Mười bảy cột EBM** (cấm `action_id`, `final_result`, weak-label): `risk_probability`, `uncertainty`, `risk_margin`, `course_progress`, `inactivity_streak`, `active_day_rate`, `assessments_due`, `regularity_score`, `content_coverage`, `quiz_activity`, `missing_assessment_count`, `due_soon_count`, `completion_rate`, `vle_available`, `study_material_available`, `quiz_available`, `stage`. Mỗi hành động một EBM riêng.

**An toàn sau xếp hạng:** nếu Top-1 `s` thấp hơn ngưỡng, chuyển `INSUFFICIENT_EVIDENCE`; nếu biên Top-1/Top-2 quá mỏng hoặc bất định còn cao, chuyển `HUMAN_REVIEW`.

**Kế hoạch sau Top-1:** template xác định (ví dụ: nộp bài trước hạn 24 giờ; 15–20 phút VLE mỗi ngày khi phục hồi tương tác). Gemini không tham gia bước này lúc serving.

### 3.5.3. Tích hợp dữ liệu và giới hạn thiết kế

PostgreSQL: `catalog.enrollment` → `prediction.prediction` → `recommendation.recommendation` + `recommendation_item`. Bảng `training.lock` lưu JSON khóa Hybrid CNN–BiLSTM và bộ so sánh một-trọng-số.

Giới hạn chương (thiết kế, không phải bảng hiệu suất):

- Nhãn nhị phân tường minh; cấm G3 / `final_result` / `score` làm predictor.
- FIT-only scale; group-split; STOP-only `t`; outer không chọn mô hình.
- Một kiến trúc hai miền; lệch lớp bằng `pos_weight` FIT-only.
- Recommendation V xếp hành động khả thi, không ước lượng ATE lên `final_result`.
- S0 / 20% là mốc thiếu chuỗi (CNN/BiLSTM tắt khi `T = 0`) — không dùng làm claim chính của kiến trúc lai.
- AP UCI và AP OULAD không so trực tiếp với nhau (khác prevalence, khác sinh dữ liệu).
- 100% OULAD không phải mốc cảnh báo sớm; Recommendation V không nhận 100%.
