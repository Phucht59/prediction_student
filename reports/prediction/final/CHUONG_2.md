# Chương 2. Cơ sở lý thuyết

## 2.1. Cơ sở lý thuyết về cảnh báo sớm nguy cơ học tập

### 2.1.1. Định nghĩa bài toán

Cảnh báo sớm (early warning) trong Educational Data Mining là bài toán **phân lớp nhị phân có thứ tự thời gian**: tại một mốc cutoff, mô hình xếp hạng xác suất sinh viên thuộc lớp nguy cơ, chỉ dùng thông tin đã quan sát **trước** cutoff.

- **Lớp dương (nguy cơ):** UCI — `G3 < 10` (thang 0–20). OULAD — `final_result ∈ {Fail, Withdrawn}`.
- **Lớp âm:** UCI — `G3 ≥ 10`. OULAD — Pass hoặc Distinction.
- **Không phải hồi quy điểm** và **không phải dự báo chuỗi nồng độ liên tục** như bài PM2.5. Đầu ra là logit nhị phân `z`, xác suất `p = σ(z)`.

Nếu predictor chứa `G3`, `final_result`, `score` hoặc `date_unregistration` thì bài toán không còn là cảnh báo sớm.

### 2.1.2. Lệch lớp và hệ quả chọn chỉ số

Lớp dương thường là thiểu số. Trên UCI, prevalence toàn tập là 0.220. Một mô hình hằng âm đạt accuracy ≈ 0.78 mà AP thấp.

- **Accuracy:** tỷ lệ đoán đúng hai lớp; dễ bị thống trị bởi lớp âm.
- **ROC-AUC:** đối xử đối xứng hai lớp; có thể cao khi precision lớp dương vẫn kém.
- **AP (average precision):** diện tích dưới đường precision–recall theo thứ tự `p` giảm dần. Đây là chỉ số **xếp hạng lớp dương**.

Khóa luận lấy AP làm chỉ số chính. Precision, Recall, F1 chỉ báo cáo tại **một** ngưỡng `t` đã chọn trên STOP.

### 2.1.3. Rò rỉ thời gian và rò rỉ nhãn

- **Rò rỉ nhãn:** dùng chính biến tạo nhãn (hoặc biến đồng thời với nhãn) làm đầu vào. Ví dụ: `G3` → nhãn UCI; `final_result` → nhãn OULAD; `absences` có thể đồng thời với kết quả học kỳ.
- **Rò rỉ tương lai:** sự kiện OULAD có `event_time ≥ cutoff`; enrollment đã hủy trước cutoff nhưng vẫn được giữ trong mẫu “cảnh báo sớm”.
- **Rò rỉ nhóm:** cùng một học sinh (hai môn UCI, nhiều enrollment OULAD) nằm cả train lẫn VALID.

Quy tắc OULAD: `observation_start ≤ event_time < cutoff`. Enrollment có `date_unregistration < cutoff` bị loại khỏi mốc đó.

### 2.1.4. Mốc thông tin, không phải mô hình riêng từng mốc

UCI có ba **view** của cùng một bản ghi: S0 (chưa G1/G2), S1 (có G1), S2 (có G1 rồi G2). OULAD có năm view: 20 / 35 / 50 / 75 / 100% chiều dài môn. Đây là trạng thái thông tin, không phải năm mô hình độc lập. Một checkpoint / miền phải chấm được mọi mốc; khi chuỗi rỗng, nhánh temporal phải tắt chứ không học nhiễu pad.

## 2.2. Hai bộ dữ liệu dùng trong khóa luận

### 2.2.1. UCI Student Performance

Cortez và Silva (2008) công bố hai file CSV phân tách `;`: Mathematics (395 dòng) và Portuguese (649 dòng), 33 cột gốc. Mỗi dòng là một cặp (học sinh, môn) trong một học kỳ.

- Điểm `G1`, `G2`, `G3` thang 0–20.
- Khóa luận gộp hai môn thành 1 044 bản ghi, thêm cột `subject`.
- Nhóm fold: 13 trường quasi-identity tạo `global_student_group` (662 nhóm); 366 nhóm xuất hiện ở cả hai môn — nếu chia fold theo dòng thì cùng học sinh có thể vừa train vừa VALID.

Chuỗi tối đa T = 2. Phù hợp để kiểm tra hành vi “tắt CNN/BiLSTM khi chưa có điểm” (S0).

### 2.2.2. OULAD

Kuzilek, Hlosta và Zdrahal (2017): Open University Learning Analytics Dataset.

- 32 593 enrollment, 28 785 sinh viên.
- `studentVle`: 10 655 280 sự kiện click (site, date, sum_click).
- `final_result`: Pass, Distinction, Fail, Withdrawn.
- `module_presentation_length` khác nhau theo môn–kỳ; cutoff là tỷ lệ chiều dài này, không phải số tuần cố định.

Tại 100%, mẫu còn lại đã loại nhiều enrollment rút sớm; prevalence lớp dương giảm so với 20%. AP tại 100% **không** diễn giải như cảnh báo sớm.

Hai miền **không** gộp train. AP UCI và AP OULAD **không** so trực tiếp (khác prevalence, khác sinh dữ liệu).

## 2.3. Tổng quan về khai phá dữ liệu

Khai phá dữ liệu là quá trình tìm mẫu có ích từ dữ liệu lớn, kết hợp thống kê, học máy và quản trị dữ liệu. Các nhiệm vụ điển hình:

- **Phân lớp (Classification):** gán đối tượng vào một lớp đã định nghĩa.
- **Hồi quy (Regression):** dự đoán giá trị số liên tục.
- **Phân cụm (Clustering):** nhóm không nhãn.
- **Luật kết hợp (Association rules):** quan hệ đồng xuất hiện.

Đề tài thuộc **phân lớp nhị phân** (nguy cơ / không nguy cơ), có ràng buộc thời gian cutoff. Không phải hồi quy `G3` hay hồi quy nồng độ.

Quy trình: thu thập CSV gốc → làm sạch và cấm trường rò rỉ → dựng tensor có mask → huấn luyện Hybrid → đánh giá AP inner → Recommendation V đọc `PredictionResult`.

## 2.4. Chuỗi có mask và cutoff — khác dự báo chuỗi hồi quy

Dữ liệu chuỗi thời gian cổ điển là quan sát cách đều, mục tiêu là giá trị tương lai của chính chuỗi (ví dụ PM2.5 ngày t+1). Bài khóa luận khác ở ba điểm:

- Mục tiêu là **nhãn cuối kỳ / cuối môn**, không phải giá trị bước kế tiếp của chuỗi điểm hay click.
- Độ dài chuỗi **thay đổi theo cutoff** và theo sinh viên; phải có `temporal_mask` và `lengths`.
- Hướng “tương lai” của BiLSTM chỉ nằm **trong cửa sổ đã quan sát**. Không được đọc tuần sau cutoff.

Do đó Hybrid không phải CNN→LSTM nối tiếp trên cửa sổ trượt cố định như một số khóa luận dự báo PM2.5. CNN và BiLSTM chạy **song song** trên cùng chuỗi đã mask, rồi trộn với tabular.

## 2.5. Các kỹ thuật tiền xử lý dữ liệu

Tiền xử lý chuyển CSV thô thành tensor `UnifiedHybridData` thống nhất hai miền.

- **Xử lý thiếu / pad:** ô temporal không hợp lệ = 0 và `mask = 0`. Không nội suy G1/G2 giả ở S0. Không nội suy tuần VLE sau cutoff.
- **Chuẩn hóa FIT-only:**
  - Context tĩnh: one-hot + StandardScaler, fit trên FIT.
  - Temporal: `MaskedStandardScaler` — chỉ ô `mask = 1`.
  - Aggregate: mean/std FIT, chỉ hàng `aggregate_available = 1`.
  - STOP, VALID, outer **không** refit.
- **Cân bằng lớp lúc huấn luyện:** `pos_weight` = `(n_neg / n_pos)_FIT × λ`. λ UCI = 1.183; λ OULAD = 0.779. SMOTE/ADASYN trên tensor được thử rồi **không chọn**: nội suy không tạo G1/G2 hay tuần VLE thật.
- **Chia tập group-disjoint:** UCI theo `global_student_group`, OULAD theo `id_student`. Outer fold 0 chỉ để loại ID, không đánh giá khi khóa.

Công thức `pos_weight` trên FIT:

\[
\texttt{pos\_weight}_{\mathrm{FIT}} = \frac{n_{\mathrm{neg}}}{n_{\mathrm{pos}}}\Big|_{\mathrm{FIT}} \times \lambda.
\]

## 2.6. Các mô hình học sâu dùng trong đề tài

Đề tài dùng kiến trúc lai CNN ∥ BiLSTM + nhánh tabular + cổng softmax, không dùng CNN→LSTM nối tiếp thuần túy.

**Mạng nơ-ron tích chập 1D (CNN).** CNN học bộ lọc trượt trên dữ liệu có cấu trúc lưới. Với chuỗi, bộ lọc 1D trượt theo thời gian. Trong Hybrid:

- Adapter `Linear + LayerNorm` đưa kênh temporal về `d_fuse = 128`, nhân mask.
- Hai block dư, **kernel 2**, dilation 1 rồi 2, 64 kênh, GELU, Dropout, pad đối xứng, nhân mask sau mỗi block.
- Gộp masked mean–max rồi chiếu về 128 chiều → `h_cnn`.
- Khi `lengths = 0`: `h_cnn = 0`.

**LSTM và BiLSTM.** LSTM dùng cổng quên / vào / ra và trạng thái ô để giảm vanishing gradient so với RNN. BiLSTM chạy hai hướng trên chuỗi đã cắt. Hybrid dùng **một** lớp BiLSTM, hidden 128, `pack_padded_sequence` theo `lengths`. Gộp masked mean–max (đầu ra 256 chiều → 512 sau mean–max) rồi Linear → `h_lstm ∈ ℝ^{128}`. Hướng ngược không nhìn quá cutoff.

**Cổng fusion 3 nhánh.** Tabular: ResidualProjector trên static và aggregate (`aggregate` nhân `aggregate_available`) → `h_tab`. Logit cổng nhận `[h_tab; h_cnn; h_lstm; a_tab=1; a_cnn; a_lstm; progress]`. Nhánh tắt: logit = −∞ trước softmax. Biểu diễn trộn:

\[
h = g_{\mathrm{tab}} h_{\mathrm{tab}} + g_{\mathrm{cnn}} h_{\mathrm{cnn}} + g_{\mathrm{lstm}} h_{\mathrm{lstm}}.
\]

Head: LayerNorm → Linear 128 → GELU → Dropout → Linear 1 → logit `z`.

**Bộ so sánh tabular (không phải mô hình khóa):** LR, DT, RF, SVM, MLP, XGB trên cùng đặc trưng `static + aggregate + last/mean/max temporal + progress`, cùng FIT/STOP/VALID. Chúng không nhìn thứ tự thô của chuỗi; đó là trần inductive của bảng tóm tắt, dùng để đối chiếu chứ không thay Hybrid.

## 2.7. Huấn luyện và đánh giá mô hình

- **Hàm mất mát:** Binary Cross-Entropy with logits, có `pos_weight` trên mẫu dương:

\[
\mathcal{L}_{\mathrm{BCE}} = -\frac{1}{N}\sum_{i=1}^{N} w_i\left[ y_i \log \sigma(z_i) + (1-y_i)\log(1-\sigma(z_i)) \right],
\]

\(w_i = \texttt{pos\_weight}\) nếu \(y_i=1\), bằng 1 nếu \(y_i=0\).

- **Bộ tối ưu:** AdamW. UCI: `lr = 8.61×10⁻⁵`, `weight_decay = 3.29×10⁻³`. OULAD: `lr = 1.18×10⁻⁴`, `weight_decay = 7.11×10⁻⁴`. Dropout UCI 0.406; OULAD 0.320.
- **Xác suất và nhãn vận hành:**

\[
p=\sigma(z)=\frac{1}{1+e^{-z}},\qquad \hat y=\mathbf{1}[p\ge t].
\]

- **Bất định nhị phân** (định tuyến HUMAN_REVIEW):

\[
H_2(p)= -\frac{p\log p + (1-p)\log(1-p)}{\log 2}.
\]

- **AP (sklearn, không tự tính thang hình thang rồi gọi PR-AUC):**

\[
\mathrm{AP} = \sum_n (R_n-R_{n-1}) P_n
\]

với \(P_n, R_n\) là precision và recall tại ngưỡng thứ \(n\) trên thứ tự `p` giảm dần.

- **Chọn ngưỡng `t`:** chỉ trên STOP, lưới 0.05–0.95; xếp F1, rồi recall, rồi `|t − 0.5|`. VALID không chọn `t`.
- **Đánh giá:** không k-fold xáo iid trên dòng. Inner 3 fold group-disjoint × 3 seed = 9 số / mốc. Báo cáo trung bình 9 số. Outer không dùng khi khóa.

## 2.8. Công trình liên quan (khác protocol — không phải trần của khóa luận)

Một số công bố trên OULAD hoặc early warning dùng ROC-AUC, nhãn khác, hoặc không công bố quy tắc cutoff:

- Jha và cộng sự (2019): AUC dropout khoảng 0.91 (GBM, VLE) — ROC, khác split.
- Kuznetsov (2025): AUC 0.789 ngày 14; AP 0.722 — ultra-early, GB ≈ LR.
- Một số bài BiLSTM+MLP báo ROC-AUC 0.95 — có thể dùng assessment sau cutoff; không phải protocol khóa luận.
- CNN–LSTM báo accuracy rất cao trên bài khác — không cutoff-safe, không dùng làm trần.

Khóa luận **không** claim “vượt 0.95 ROC”. Số công bố của đề tài là AP inner 3×3, nhãn Fail|Withdrawn hoặc `G3 < 10`, cutoff-safe. Transformer / temporal GNN là hướng phát triển (Chương 5), chưa dùng vì protocol CNN ∥ BiLSTM đã khóa và UCI T = 2 gần như không có cửa sổ attention.
