# Chương 5. Kết luận và hướng phát triển

## 5.1. Kết luận

Khóa luận đã hoàn thành các mục tiêu đề ra: xây dựng và đánh giá mô hình học sâu lai **Hybrid CNN–BiLSTM** dự đoán nguy cơ học tập nhị phân trên UCI và OULAD theo protocol cutoff-safe, rồi gắn **Recommendation V** để xếp hành động hỗ trợ trên OULAD 20 / 35 / 50 / 75%. Đề tài không xây giao diện người dùng.

Qua quá trình nghiên cứu và thực nghiệm, đề tài đã đạt được những kết quả cụ thể sau:

- **Xây dựng thành công mô hình học sâu lai:** Đã xây dựng và huấn luyện Hybrid CNN–BiLSTM bằng PyTorch: ResidualProjector tabular, CNN song song BiLSTM (kernel 2, dilation 1–2, hidden 128), cổng softmax 3 nhánh có mask, head logit nhị phân. Một class, một checkpoint UCI (S0–S2), một checkpoint OULAD (20–100%), 482 116 tham số trên checkpoint OULAD serving. Khi `lengths = 0`, CNN/BiLSTM tắt đúng thiết kế.
- **Đánh giá hiệu năng một cách khách quan:** Inner group-disjoint 3 fold × 3 seed = 9 số / mốc; AP = `sklearn.metrics.average_precision_score`; outer không dùng để chọn mô hình. UCI S1 AP **0.8214**, S2 **0.9101** (Wilcoxon S1 hơn LR và RF, p = 0.0039, 9/9; S2 hơn LR p = 0.0078; cùng protocol S1/S2 cao hơn XGB +0.044 / +0.014). OULAD một checkpoint: AP 0.762 → 0.806 → 0.848 → 0.889 → 0.920; từ 35% trở đi Hybrid hơn LR và RF có ý nghĩa. XGB lệch Hybrid ±0.002 trên 35–100%.
- **Kiểm chứng cổng (H3) và ablation (H1):** Mass cổng UCI S0 tabular = 1; OULAD BiLSTM tăng 0.45 → 0.59 khi cutoff tăng. Ablation một mốc: OULAD 35% full 0.809 vượt tabular/CNN/BiLSTM-only; UCI S1 full 0.799 cao nhất sáu arm. Số khóa serving vẫn là mixed-state S1 0.821.
- **Recommendation V:** NDCG@3 **0.88785** vs B1 0.86649 (Δ +0.021, 95% CI không chứa 0); invalid-action **0**; đủ năm hành động Top-1. 57.4% INSUFFICIENT_EVIDENCE là an toàn, không phải lỗi xếp hạng.

Từ các kết quả trên, khóa luận mang lại đóng góp:

- **Về học thuật và kỹ thuật:** Protocol cutoff-safe dùng chung hai miền khác T; AP trên lớp hiếm; group-split; STOP-only ngưỡng `t`; cổng có mask kiểm chứng được. Toàn bộ quy trình từ cấm trường rò rỉ, FIT-only scale, đến 9-run và Wilcoxon được ghi tường minh.
- **Về thực tiễn:** Hybrid đưa ra `p`, `t`, `ŷ`, `H₂` tại các mốc còn thời gian (S1; OULAD 35–75%). Recommendation V chuyển xác suất thành hành động khả thi có luật eligible/chặn, không viết “làm vậy thì đỗ”. CLI/PostgreSQL chỉ đọc kết quả đã khóa.

Claim đặt ở **UCI S1/S2** và **OULAD 35–75%**. S0 và 20% không dùng để phủ nhận kiến trúc lai. 100% không dùng cảnh báo sớm.

## 5.2. Hạn chế

Mặc dù khóa luận đạt các mục tiêu chính, vẫn còn hạn chế cần nhìn nhận khách quan. Việc nêu rõ hạn chế là cơ sở cho hướng phát triển, không phải gian lận số liệu.

- **Hạn chế về dữ liệu:**
  - **Phạm vi địa lý:** Không có dữ liệu sinh viên Việt Nam. UCI và OULAD là bộ công khai; mô hình khóa **không** được diễn giải như đã triển khai tại một trường cụ thể ở Việt Nam.
  - **UCI quy mô nhỏ và T = 2:** 1 044 bản ghi, chuỗi tối đa hai bước. S0 không có G1/G2 (khe FIT−VALID 0.125, ECE 0.254). View serving vẫn điền G1/G2 vào cả temporal và aggregate tại S1/S2.
  - **OULAD 100%:** Prevalence giảm 0.424 → 0.317 vì enrollment rút trước cutoff bị loại. AP 0.920 tại 100% không phải chỉ số cảnh báo sớm. Recommendation V từ chối 100%.
- **Hạn chế về mô hình và kỹ thuật:**
  - **Outer không mở:** Chưa có ước lượng test cuối cùng ngoài vòng FIT/STOP/VALID. Outer fold 0 chỉ là firewall.
  - **Kiến trúc:** CNN ∥ BiLSTM + cổng đã khóa. Transformer / temporal GNN mask-safe chưa thử trên cùng split và AP.
  - **HPO:** Siêu tham số Hybrid là bộ khóa một-trọng-số theo miền. Roster so sánh (kể cả XGB) cũng một-trọng-số, không Optuna trên bản phục vụ.
  - **XAI:** Không SHAP từng điểm. Bằng chứng diễn giải là mass cổng trung bình theo cutoff.
  - **Checkpoint không lưu history epoch:** Không vẽ đường loss giả. Đường epoch (nếu có) thuộc bản research ablation, không thay bản khóa.
- **Hạn chế về phạm vi ứng dụng:**
  - **Không giao diện:** Không FastAPI công khai, không app di động. CLI chỉ đọc `prediction` / `recommendation` đã materialize.
  - **Recommendation V không phải can thiệp nhân quả:** Không ATE, không RCT, không thử nghiệm với giảng viên thật. Panel C dùng reviewer LLM lúc gán nhãn yếu, không chạy lúc serving.
  - **Nhãn nhị phân gộp:** Fail và Withdrawn chung một lớp dương. Tách hai đầu ra là hướng khác, không phải bản khóa.

Các hạn chế trên không phủ nhận ưu thế Hybrid tại S1/S2 và OULAD 35–75% trên protocol đã khóa.

## 5.3. Hướng phát triển trong tương lai

Từ những hạn chế đã phân tích, đề tài mở ra các hướng sau. Mọi hướng mới phải giữ cutoff-safe, group-split và AP; không mở outer để chọn mô hình rồi mới “khóa”.

- **Nâng cao dữ liệu:**
  - Thu thập dữ liệu trường Việt Nam (cùng quy tắc cấm nhãn / tương lai).
  - Mở outer **một lần** sau khi đóng băng mọi lựa chọn (cần duyệt tường minh).
  - Tách Fail và Withdrawn thành bài toán riêng nếu có đủ mẫu Withdrawn tại VALID.
- **Cải tiến mô hình:**
  - Thử Transformer hoặc temporal GNN **mask-safe** trên OULAD 35–75%, cùng split và AP với bản khóa.
  - Bỏ nhánh aggregate trùng G1/G2 trên UCI nếu ablation `temporal_only` không kém `both` trên 9 run khóa.
  - Hiệu chỉnh xác suất (ECE S0 còn 0.254).
- **Hoàn thiện phục vụ (ngoài phạm vi khóa luận hiện tại):**
  - API/giao diện cảnh báo sớm nếu triển khai sản phẩm — không thuộc bản khóa này.
  - Thử nghiệm với tư vấn học tập thật; đo hành vi, không đo ATE giả.
  - Cá nhân hóa hành động Recommendation V theo ngữ cảnh môn, không refit Hybrid trên Panel C.
