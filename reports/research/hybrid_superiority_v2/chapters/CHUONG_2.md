# Chương 2. Cơ sở lý thuyết

## 2.1. Educational Data Mining và cảnh báo sớm

EDM dùng dữ liệu học tập để mô hình hóa kết quả, hành vi và nguy cơ. Bài toán **cảnh báo sớm** khác bài toán “dự đoán cuối khóa khi đã có mọi sự kiện”: đặc trưng chỉ được lấy khi `event_time < cutoff`. Đưa `final_result`, `G3`, `score` hay ngày hủy đăng ký vào predictor là rò rỉ.

Lớp nguy cơ thường là thiểu số. Accuracy dễ cao khi mô hình luôn đoán “không rủi ro”. Do đó khóa luận chọn **AP (average precision)** làm chỉ số chính: đo chất lượng **xếp hạng** lớp dương trên toàn bộ ngưỡng.

## 2.2. UCI Student Performance và OULAD

**UCI** (Cortez & Silva, 2008): điểm G1, G2, G3 thang 0–20. Nhãn khóa: `risk = [G3 < 10]`. Chuỗi tối đa T = 2. Phù hợp kiểm tra hành vi “tắt CNN/BiLSTM khi chưa có điểm”.

**OULAD** (Kuzilek, Hlosta & Zdrahal, 2017): 32 593 enrollment, nhật ký VLE. Nhãn khóa: Fail|Withdrawn. Cutoff 20/35/50/75/100% `module_presentation_length`. Sự kiện: `observation_start ≤ t < cutoff`.

Hai miền **không** gộp train. AP hai miền **không** so trực tiếp (khác prevalence, khác sinh dữ liệu).

## 2.3. CNN, BiLSTM và cổng fusion

CNN 1D trích mẫu cục bộ trên chuỗi đã mask. BiLSTM mã hóa phụ thuộc hai chiều **trong cửa sổ đã cắt**. Fusion softmax 3 nhánh (tabular, CNN, BiLSTM) có availability: nhánh tắt nhận logit −∞. Khi `lengths = 0`, chỉ tabular — đây là ràng buộc thiết kế, không phải lỗi.

Khác kiến trúc CNN→LSTM nối tiếp (như một số khóa luận PM2.5): bản này chạy **CNN ∥ BiLSTM**, rồi trộn với tabular.

## 2.4. Tại sao AP chứ không ROC-AUC

ROC-AUC đối xử đối xứng hai lớp và có thể cao khi precision của lớp hiếm vẫn kém. AP = diện tích dưới đường precision–recall theo định nghĩa `average_precision_score` của scikit-learn (không tự tính thang hình thang rồi gọi là PR-AUC). F1/Precision/Recall chỉ báo cáo tại **một** ngưỡng STOP; không thay AP.

## 2.5. Ranking hành động

Recommendation V xếp năm hành động khả thi từ `PredictionResult` + bằng chứng cutoff-safe. Chỉ số: NDCG@3, P@1, invalid-action. **Không** phải mô hình nhân quả (không ATE, không RCT).

## 2.6. Công thức toán dùng trong khóa luận

Ký hiệu: \(z\) logit, \(p=\sigma(z)\), \(y\in\{0,1\}\), \(t\) ngưỡng STOP.

**BCE with logits (cost-sensitive):**

\[
\mathcal{L}_{\mathrm{BCE}} = -\frac{1}{N}\sum_{i=1}^{N} w_i\left[ y_i \log \sigma(z_i) + (1-y_i)\log(1-\sigma(z_i)) \right]
\]

với \(w_i = \texttt{pos\_weight}\) nếu \(y_i=1\), bằng 1 nếu \(y_i=0\), và

\[
\texttt{pos\_weight}_{\mathrm{FIT}} = \frac{n_{\mathrm{neg}}}{n_{\mathrm{pos}}}\Big|_{\mathrm{FIT}} \times \lambda.
\]

\(\lambda\) UCI = 1.183; OULAD = 0.779. Chỉ tính trên FIT.

**Xác suất và nhãn vận hành:**

\[
p=\sigma(z)=\frac{1}{1+e^{-z}},\qquad \hat y=\mathbf{1}[p\ge t].
\]

**Bất định nhị phân (dùng cho HUMAN_REVIEW):**

\[
H_2(p)= -\frac{p\log p + (1-p)\log(1-p)}{\log 2}.
\]

**Cổng softmax có mask.** Đặt \(a_{\mathrm{tab}}=1\), \(a_{\mathrm{cnn}}=a_{\mathrm{lstm}}=\mathbf{1}[\mathrm{lengths}>0]\). Logit cổng:

\[
\ell = W\,[h_{\mathrm{tab}};h_{\mathrm{cnn}};h_{\mathrm{lstm}};a_{\mathrm{tab}};a_{\mathrm{cnn}};a_{\mathrm{lstm}};\mathrm{progress}].
\]

Nhánh \(k\) tắt: \(\ell_k \leftarrow -\infty\). Khối lượng \(g=\mathrm{softmax}(\ell)\). Biểu diễn trộn:

\[
h = g_{\mathrm{tab}} h_{\mathrm{tab}} + g_{\mathrm{cnn}} h_{\mathrm{cnn}} + g_{\mathrm{lstm}} h_{\mathrm{lstm}}.
\]

**AP (sklearn):**

\[
\mathrm{AP} = \sum_n (R_n-R_{n-1}) P_n
\]

với \(P_n,R_n\) là precision và recall tại ngưỡng thứ \(n\) trên thứ tự \(p\) giảm dần (định nghĩa `average_precision_score`).

## 2.7. Kiến trúc hiện đại hơn: Transformer và temporal GNN

Transformer (self-attention) học phụ thuộc xa trên chuỗi dài; Graph Neural Network thời gian mô hình hóa quan hệ sinh viên–tài nguyên–diễn đàn. Trên OULAD (tới ~39 tuần) đây là hướng hợp lý.

Khóa luận **chưa** dùng Transformer/GNN vì:

1. UCI T = 2: attention gần như không có chuỗi.
2. Protocol đã khóa CNN ∥ BiLSTM + cổng; đổi backbone sẽ phá so sánh 9-run.
3. Cần mask cutoff-safe và group-split — nhiều bài Transformer trên OULAD không công bố cutoff rule, nên số ROC 0.95 trên tài liệu **không** phải trần công bằng.

Hướng phát triển (Chương 5): thử Transformer mask-safe trên OULAD 35–75%, giữ nguyên split và AP.

## 2.8. Số liệu công bố (khác protocol — không so AP trực tiếp)

| Nguồn | Chỉ số | Ghi chú protocol |
|---|---|---|
| Jha et al. (2019), OULAD | AUC dropout ~0.91 (GBM, VLE) | ROC, khác nhãn/split |
| Kuznetsov (2025), early warning | AUC 0.789 ngày 14; AP 0.722 | Ultra-early; GB ≈ LR |
| Frontiers (2026) BiLSTM+MLP | ROC-AUC 0.95 | Có thể dùng thêm assessment; khác cutoff |
| CNN–LSTM MDPI (2025) | Accuracy 98.9% | Không cutoff-safe; không dùng làm trần |

Khóa luận không claim “vượt 0.95 ROC”. Số khóa là AP inner 3×3, Fail|Withdrawn, cutoff-safe.
