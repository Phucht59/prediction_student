# Chương 1. Tổng quan đề tài nghiên cứu

## 1.1. Lý do chọn đề tài

### 1.1.1. Nhu cầu cảnh báo sớm nguy cơ học tập

Bối cảnh thực trạng: nhà trường cần biết **ai đang có nguy cơ không hoàn thành môn** khi vẫn còn thời gian can thiệp, chứ không phải sau khi đã có điểm cuối kỳ hoặc trạng thái hủy đăng ký. Đây là bài toán **cảnh báo sớm** (early warning) trong Educational Data Mining: đặc trưng chỉ được lấy từ thông tin đã quan sát **trước** một mốc cutoff.

Hai nguồn dữ liệu công khai thường dùng, nhưng khác nhau về bản chất thời gian:

- **UCI Student Performance** (Cortez và Silva, 2008): 395 bản ghi Mathematics và 649 bản ghi Portuguese, gộp thành **1 044** cặp (học sinh, môn). Chuỗi điểm tối đa **hai bước** (G1, G2); nhãn khóa luận lấy từ `G3 < 10` (thang 0–20). Tỷ lệ lớp nguy cơ trên 1 044 bản ghi là **0.220** (230/1044).
- **OULAD** (Kuzilek, Hlosta và Zdrahal, 2017): **32 593** enrollment, **28 785** sinh viên, nhật ký `studentVle` **10 655 280** dòng click. Nhãn khóa luận: Fail hoặc Withdrawn. Năm mốc cutoff: 20 / 35 / 50 / 75 / 100% chiều dài môn (`module_presentation_length`).

Nếu đưa `G3`, `final_result`, `score` hoặc ngày hủy đăng ký vào predictor thì mô hình không còn là cảnh báo sớm — đó là rò rỉ nhãn / tương lai.

### 1.1.2. Hậu quả khi xếp hạng nguy cơ sai

Lớp nguy cơ là thiểu số. Accuracy dễ cao khi mô hình luôn đoán “không rủi ro”: trên UCI, luôn đoán âm đã đạt khoảng **0.78** accuracy mà không xếp được ai cần hỗ trợ.

Hậu quả thực tế của một hệ thống xếp hạng kém:

- Bỏ sót sinh viên đang trượt / rút (recall thấp) → can thiệp muộn.
- Báo động giả hàng loạt (precision thấp) → quá tải tư vấn.
- Dùng một ngưỡng cố định cho mọi mốc thông tin → cùng một xác suất `p` bị diễn giải khác nhau khi G1 chưa có hoặc khi đã hết tuần VLE.

Do đó khóa luận chọn **AP** (`sklearn.metrics.average_precision_score`) làm chỉ số chính: đo chất lượng **xếp hạng lớp dương trên mọi ngưỡng**, không đối xứng hai lớp như ROC-AUC.

### 1.1.3. Nhu cầu một kiến trúc dùng được trên hai miền

Hai miền **không** gộp thành một tập huấn luyện. UCI là bản ghi học kỳ, T = 2; OULAD là chuỗi tuần, T tới khoảng 39. Một mô hình “chỉ hồi quy trên bảng phẳng” bỏ qua thứ tự G1→G2 và thứ tự tuần VLE. Một mô hình “chỉ CNN/LSTM” không có tín hiệu khi chuỗi rỗng (UCI S0: chưa có G1/G2).

Nhu cầu thiết kế:

- Cùng một class nhận tensor thống nhất (static, temporal có mask, aggregate, progress).
- Tắt nhánh CNN và BiLSTM khi `lengths = 0`, không học pad.
- Một checkpoint / miền chấm mọi mốc thông tin (UCI S0–S2; OULAD 20–100%), không huấn luyện mô hình riêng cho 100%.
- Cutoff-safe: sự kiện OULAD chỉ khi `observation_start ≤ event_time < cutoff`.

### 1.1.4. Tiềm năng của kiến trúc lai CNN–BiLSTM và lớp khuyến nghị

CNN 1D trích mẫu cục bộ trên cửa sổ ngắn (điểm liền kề, tuần liền kề). BiLSTM mã hóa phụ thuộc hai chiều **trong cửa sổ đã cắt tại cutoff**. Nhánh tabular giữ ngữ cảnh tĩnh và thống kê gộp. Cổng softmax 3 nhánh (tabular, CNN, BiLSTM) có mask availability quyết định nhánh nào được dùng.

Sau xác suất nguy cơ, nhà trường vẫn cần **hành động khả thi** (nộp bài, phục hồi tương tác, ôn đều, …), không chỉ một số `p`. **Recommendation V** xếp năm hành động trên `PredictionResult` của Hybrid, luật feasibility cứng, không ước lượng nhân quả lên `final_result`.

Đề tài **không** xây giao diện người dùng hay API công khai. Phạm vi là mô hình dự đoán, đánh giá inner 3×3, và Recommendation V.

## 1.2. Mục tiêu nghiên cứu

### 1.2.1. Mục tiêu tổng quát

Xây dựng và đánh giá mô hình **Hybrid CNN–BiLSTM** dự đoán nguy cơ học tập nhị phân trên UCI và OULAD theo protocol cutoff-safe, rồi gắn **Recommendation V** để xếp hành động hỗ trợ trên OULAD 20 / 35 / 50 / 75%.

### 1.2.2. Mục tiêu cụ thể

- **Về mô hình dự báo:**
  - Một class `Hybrid` (`model_id = hybrid`), `architecture_id = C0`.
  - Một checkpoint UCI chấm S0, S1, S2; một checkpoint OULAD chấm 20 / 35 / 50 / 75 / 100%.
  - CNN song song BiLSTM, cổng softmax 3 nhánh có mask; tắt CNN/BiLSTM khi `lengths = 0`.
  - Không huấn luyện mô hình riêng cho OULAD 100%.
- **Về đánh giá khoa học:**
  - Chỉ số chính: AP = `sklearn.metrics.average_precision_score` trên VALID inner (không gọi PR-AUC).
  - Inner group-disjoint 3 fold × 3 seed (`42`, `1201`, `2026`) = **9 số / mốc**; báo cáo trung bình 9 số, không lấy run đẹp nhất.
  - So sánh cùng protocol với LR, DT, RF, SVM, MLP, XGB (một-trọng-số, không HPO trên roster khóa).
  - Outer fold tồn tại nhưng **không** dùng để chọn kiến trúc, siêu tham số hay mô hình khóa.
  - Kiểm định giả thuyết:
    - **H1:** trên UCI S1 và OULAD 35%, AP Hybrid đầy đủ > AP tabular-only (Wilcoxon hai phía, 9 run, α = 0.05).
    - **H2:** trên mốc có chuỗi (UCI S1; OULAD 35% trở lên), AP Hybrid > AP LR và AP RF (9 cặp fold×seed, α = 0.05).
    - **H3:** cổng softmax tăng khối lượng CNN+BiLSTM khi cutoff OULAD tăng.
- **Về Recommendation V:**
  - Chỉ OULAD 20 / 35 / 50 / 75; 100% bị từ chối trước khi xếp hạng.
  - Đọc đúng `PredictionResult` (`p`, `t`, `ŷ`, `H₂`); không đọc vector CNN/LSTM nội bộ.
  - NDCG@3 trên Panel C held-out; invalid-action = 0; không ATE, không RCT.

Claim chính đặt ở **UCI S1/S2** và **OULAD 35–75%**. S0 và 20% là mốc thiếu chuỗi / lạnh, không dùng để bác kiến trúc lai. Kết quả số liệu: Chương 4.

## 1.3. Đối tượng và phạm vi nghiên cứu

### 1.3.1. Đối tượng nghiên cứu

- **Dữ liệu đầu vào (Input):**
  - UCI: context tĩnh (categorical + numeric hợp lệ), chuỗi G1/G2 theo mốc S0/S1/S2, tóm tắt aggregate 5 chiều trên điểm **đã quan sát** (tắt tại S0). Cấm `G3`, `absences` làm predictor.
  - OULAD: context tĩnh (8 categorical + 4 numeric), 11 kênh temporal / tuần, 13 số aggregate tại cutoff. Cấm `final_result`, `score`, `date_unregistration` làm predictor. Enrollment có `date_unregistration < cutoff` bị loại khỏi mốc đó.
- **Dữ liệu đầu ra (Target):**
  - UCI: `risk = 1` khi `G3 < 10`.
  - OULAD: `risk = 1` khi `final_result ∈ {Fail, Withdrawn}`.
  - Serving: xác suất `p = σ(z)`, ngưỡng `t` chọn trên STOP, nhãn vận hành `ŷ = [p ≥ t]`, bất định `H₂(p)`.
- **Mô hình:**
  - Dự đoán: Hybrid CNN–BiLSTM, PyTorch, `d_fuse = 128`, CNN 64 kênh kernel 2 dilation (1, 2), BiLSTM hidden 128 một lớp hai chiều, cổng softmax 3 nhánh.
  - So sánh: LR, DT, RF, SVM, MLP, XGB — cùng feature tabular cutoff-safe, cùng FIT/STOP/VALID, không thay Hybrid.
  - Khuyến nghị: Recommendation V, năm EBM (`interpret`), luật feasibility cứng, năm hành động chuẩn.

### 1.3.2. Phạm vi nghiên cứu

- **Không gian và thời gian dữ liệu:**
  - UCI: hai file gốc Cortes 2008, gộp 1 044 bản ghi, 662 `global_student_group`.
  - OULAD: Open University, enrollment + VLE; số bản ghi còn đủ điều kiện theo cutoff: 20% 26 697; 35% 25 606; 50% 24 599; 75% 23 159; 100% 22 522.
  - Không dùng dữ liệu sinh viên Việt Nam trong khóa luận này.
- **Phạm vi kỹ thuật:**
  - Học máy: PyTorch, FIT-only scaler, `pos_weight` FIT-only, AdamW, early-stop STOP macro AP.
  - Đánh giá: inner 3×3, group-split; AP / Acc / Precision / F1 / Recall / ECE.
  - Lưu trữ: PostgreSQL `student_db` (`raw` → `catalog` → `prediction` → `recommendation`); CLI `python project.py db predict|recommend`.
  - Recommendation V: OULAD 20–75%; Panel C 632 case / 150 sinh viên / 2 398 review.
- **Ngoài phạm vi:**
  - Giao diện người dùng, FastAPI công khai, ứng dụng di động.
  - Mở outer test để chọn mô hình.
  - Ước lượng hiệu ứng can thiệp (ATE) lên `final_result`.
  - Thử nghiệm với giảng viên thật; dữ liệu trường Việt Nam.

## 1.4. Phương pháp nghiên cứu

### 1.4.1. Phương pháp nghiên cứu lý thuyết

- **Nghiên cứu tài liệu:**
  - EDM và cảnh báo sớm: cutoff, rò rỉ nhãn, lệch lớp.
  - UCI (Cortez và Silva, 2008) và OULAD (Kuzilek và cộng sự, 2017): schema, nhãn, nhật ký VLE.
  - CNN 1D, BiLSTM, fusion có mask; AP versus ROC-AUC trên lớp hiếm.
  - Ranking hành động (NDCG@3), không nhầm với mô hình nhân quả.
- **Thiết kế kiến trúc:**
  - Tensor thống nhất `UnifiedHybridData`.
  - Hybrid: ResidualProjector tabular ∥ ResidualCNN ∥ BiLSTM + cổng softmax 3 nhánh.
  - Hợp đồng `PredictionResult` cho Recommendation V.

### 1.4.2. Phương pháp nghiên cứu thực nghiệm

- **Tiền xử lý:**
  - UCI: gộp hai môn, tạo `record_id` / `global_student_group`, chuỗi G1/G2 theo mốc, cấm `G3`/`absences`.
  - OULAD: gom VLE theo tuần `event_time < cutoff`; loại enrollment hủy trước cutoff.
  - Scale one-hot / masked / aggregate **chỉ trên FIT**.
- **Huấn luyện và kiểm định:**
  - Group-disjoint inner 3 fold (UCI theo nhóm học sinh, OULAD theo `id_student`).
  - FIT: gradient, scaler, `pos_weight`. STOP: early-stop AP, chọn `t` (lưới F1 → recall → `|t − 0.5|`). VALID: báo cáo.
  - 3 seed × 3 fold = 9 run / mốc.
  - Outer fold 0 là firewall, không tune.
- **Đối chiếu:**
  - Cùng protocol với LR / DT / RF / SVM / MLP / XGB.
  - Ablation một mốc (bản research): tabular-only, CNN-only, BiLSTM-only, concat — không thay bản khóa mixed-state.
  - Wilcoxon trên 9 cặp fold×seed cho H1/H2.
- **Recommendation V:**
  - OOF 66 685 dòng (3 fold, seed 42, mốc 20–75%).
  - Panel C held-out: NDCG@3, P@1, invalid-action; không tune trên Panel C.

## 1.5. Ý nghĩa khoa học và thực tiễn của đề tài

### 1.5.1. Ý nghĩa khoa học

- **Về lý thuyết:**
  - Một protocol cutoff-safe dùng chung cho miền T = 2 (UCI) và miền chuỗi tuần (OULAD), không gộp train.
  - Cổng softmax có mask: khi `lengths = 0` CNN/BiLSTM nhận khối lượng 0 — kiểm chứng được bằng mass cổng (Chương 4).
  - AP làm chỉ số chính trên lớp hiếm; Acc không dùng để chọn mô hình.
- **Về kỹ thuật:**
  - Một topology (`d_fuse = 128`, CNN 64, BiLSTM 128), hai bộ trọng số.
  - Group-split, FIT-only scale, STOP-only ngưỡng `t`, không outer khi khóa.
  - Recommendation V tách khỏi Hybrid: chỉ tiêu thụ `PredictionResult`.

### 1.5.2. Ý nghĩa thực tiễn

- **Cảnh báo sớm trên mốc còn thời gian:**
  - UCI S1 (đã có G1, chưa có G3); OULAD 35 / 50 / 75% chiều dài môn.
  - 100% không dùng cảnh báo sớm; Recommendation V từ chối 100%.
- **Hành động khả thi, không phải “làm vậy thì đỗ”:**
  - Năm hành động có luật eligible/chặn tường minh.
  - Bốn trạng thái: `RECOMMEND`, `HUMAN_REVIEW`, `INSUFFICIENT_EVIDENCE`, `NO_FEASIBLE_ACTION`.
- **Giới hạn thực tiễn đã chọn:**
  - Không giao diện; CLI và PostgreSQL chỉ để đọc kết quả đã khóa.
  - Không dữ liệu trường Việt Nam trong bản khóa.

## 1.6. Bố cục của khóa luận

Chương I: Tổng quan đề tài nghiên cứu — lý do, mục tiêu, đối tượng, phạm vi, phương pháp, ý nghĩa.

Chương II: Cơ sở lý thuyết — cảnh báo sớm, hai bộ dữ liệu, khai phá dữ liệu, chuỗi có cutoff, tiền xử lý, CNN/BiLSTM/cổng, huấn luyện và chỉ số AP.

Chương III: Phân tích và thiết kế hệ thống — dữ liệu, tiền xử lý, kiến trúc, cấu hình huấn luyện, đóng gói mô hình, thiết kế Recommendation V. **Không đưa bảng hiệu suất hay nhận xét kết quả.**

Chương IV: Kết quả thực nghiệm và đánh giá — môi trường, quá trình huấn luyện, AP 3×3, đối chiếu bộ so sánh, trực quan hóa, Recommendation V, cổng XAI.

Chương V: Kết luận, hạn chế và hướng phát triển.
