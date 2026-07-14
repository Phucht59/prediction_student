# CONTEXT DỰ ÁN KHÓA LUẬN

## 1. Thông tin định danh

**Tên đề tài:** Xây dựng mô hình học kết hợp để dự đoán thành tích học tập sinh
viên.

**Mô hình nghiên cứu đề xuất:** CNN–BiLSTM phân loại ba mức kết quả học tập.

**Nguồn dữ liệu:** UCI Student Performance, tập `student-mat`.

**Kiến trúc dữ liệu chính thức:** PostgreSQL-first; CSV chỉ là nguồn ingestion
một lần.

**Mục đích của tài liệu:** Đây là context chuẩn để viết khóa luận, báo cáo kỹ
thuật, slide bảo vệ và phần trả lời phản biện. Khi có khác biệt giữa tài liệu cũ
và evidence, thứ tự ưu tiên là:

1. `artifacts/final/LATEST_RUN.txt` và bundle final đang active;
2. `artifacts/model_selection/nested-full-20260710/`;
3. `artifacts/baseline_comparison/fair-model-comparison-full/` tại workspace;
4. `docs/report_context/14_FINAL_FACTS.json`;
5. nội dung mô tả trong tài liệu này.

Không lấy kết quả smoke, debug, rehearsal hoặc một split thử nghiệm để thay thế
evidence final.

## 2. Tóm tắt điều hành

Dự án giải quyết bài toán dự đoán `G3` theo ba mức Low, Medium và High. Mô hình
CNN–BiLSTM final sử dụng hai điểm quá trình G1 và G2 như chuỗi hai bước, đạt
Macro-F1 nested outer CV `0.8781 ± 0.0448` và Macro-F1 `0.9262` trên tập test
khóa 79 mẫu. Kết quả cho thấy mô hình khả thi, tái lập được và tạo xác suất dự
đoán có chất lượng tốt.

Tuy nhiên, đóng góp của đề tài không phải là chứng minh deep learning luôn vượt
machine learning. Trong benchmark công bằng mới, khi Decision Tree, Random
Forest, SVM-RBF, XGBoost, Gradient Boosting, CNN+LSTM và CNN+BiLSTM cùng dùng
G1/G2, cùng nested CV 5×3, cùng 30 trial/model/fold và cùng chính sách không xử
lý mất cân bằng, Random Forest đứng đầu với `0.8915 ± 0.0240`, trong khi
CNN–BiLSTM benchmark đạt `0.8380 ± 0.0475`.

Hai kết quả CNN–BiLSTM `0.8781` và `0.8380` thuộc hai giao thức khác nhau:

- `0.8781 ± 0.0448` là ước lượng nested CV của quy trình lựa chọn mô hình final;
- `0.8380 ± 0.0475` là kết quả CNN–BiLSTM được tuning lại trong benchmark chung
  không resampling và không class weighting.

Do đó, không được đặt hai số này trong cùng bảng như hai lần đo của một cấu hình
giống hệt nhau. Kết luận khoa học phù hợp là CNN–BiLSTM có tính khả thi và khả
năng tái lập, nhưng chưa cho thấy ưu thế hiệu năng so với baseline ML mạnh trên
dữ liệu nhỏ và chuỗi rất ngắn.

## 3. Bối cảnh và vấn đề nghiên cứu

Các cơ sở giáo dục thường muốn phát hiện sớm người học có nguy cơ đạt kết quả
thấp để tư vấn và phân bổ hỗ trợ. Bài toán kỹ thuật là ánh xạ thông tin có trước
thời điểm kết thúc môn học thành xác suất thuộc một trong ba mức kết quả.

Trong dự án này, G1 và G2 là hai điểm đánh giá trước G3. G2 xuất hiện gần thời
điểm cuối kỳ nên có sức dự báo rất mạnh. Điều này tạo ra hai vấn đề cần trình bày
thẳng thắn:

- hiệu năng late-stage cao có thể chủ yếu đến từ G2;
- kiến trúc phức tạp không mặc nhiên tạo thêm giá trị so với mô hình đơn giản.

Vì vậy, đề tài phải đánh giá đồng thời tính khả thi của kiến trúc CNN–BiLSTM,
khả năng tái lập của pipeline và vị trí của mô hình so với baseline.

## 4. Mục tiêu nghiên cứu

### 4.1. Mục tiêu tổng quát

Xây dựng một pipeline có kiểm soát để dự đoán kết quả học tập ba mức, lưu vết dữ
liệu và thực nghiệm bằng PostgreSQL, đồng thời sinh khuyến nghị hỗ trợ mang tính
tham khảo dưới cơ chế human-in-the-loop.

### 4.2. Mục tiêu cụ thể

1. Chuẩn hóa dữ liệu và nhãn, ngăn target leakage.
2. Xây dựng CNN–BiLSTM xử lý chuỗi G1/G2.
3. Chọn siêu tham số bằng nested cross-validation chỉ trên development set.
4. Đánh giá một lần trên locked test sau khi cấu hình đã đóng băng.
5. So sánh với các baseline ML và CNN+LSTM theo giao thức chung.
6. Đánh giá phân loại, thứ tự lớp, xác suất, calibration và độ bất định.
7. Lưu lineage dữ liệu, split, prediction, metric và recommendation trong
   PostgreSQL/evidence bundle.
8. Xây dựng policy khuyến nghị có giải thích, không tự động ra quyết định giáo
   dục.

## 5. Câu hỏi nghiên cứu gợi ý

- **RQ1:** CNN–BiLSTM có dự đoán tốt ba mức thành tích học tập từ G1/G2 không?
- **RQ2:** CNN–BiLSTM có ổn định qua các fold và tái lập được từ dữ liệu
  PostgreSQL không?
- **RQ3:** CNN–BiLSTM có tạo thêm giá trị so với các baseline ML trên cùng đầu
  vào và ngân sách tuning không?
- **RQ4:** Việc dùng hoặc loại G2 ảnh hưởng thế nào đến khả năng dự báo?
- **RQ5:** Xác suất dự đoán có đủ calibration để hỗ trợ quy trình tư vấn thận
  trọng không?
- **RQ6:** Hệ thống khuyến nghị có đáp ứng yêu cầu cấu trúc, giải thích và an
  toàn thông tin không?

## 6. Dữ liệu nghiên cứu

### 6.1. Nguồn và phạm vi

`student-mat` gồm 395 quan sát của học sinh trung học Bồ Đào Nha trong môn Toán.
Schema gốc có 33 cột, gồm thông tin nhân khẩu, gia đình, học tập, xã hội, vắng
học và ba điểm G1, G2, G3.

Không mô tả dữ liệu này là dữ liệu sinh viên đại học Việt Nam. Từ “sinh viên”
trong tên đề tài là phạm vi ứng dụng mong muốn, không phải mô tả quần thể quan
sát hiện tại.

### 6.2. Nhãn dự đoán

| Lớp | Quy tắc | Tổng số | Locked-test support |
| --- | ---: | ---: | ---: |
| Low | `G3 <= 9` | 130 | 26 |
| Medium | `10 <= G3 <= 14` | 192 | 38 |
| High | `G3 >= 15` | 73 | 15 |

G3 bị loại khỏi toàn bộ đặc trưng và chỉ được join từ bảng target tại bước huấn
luyện/đánh giá. Phân bố lớp không cân bằng tuyệt đối, nhưng lớp nhỏ nhất vẫn có
73 mẫu toàn bộ và 15 mẫu trong locked test.

### 6.3. Chia dữ liệu

- Development set: 316 mẫu, dùng cho nested CV và model selection.
- Locked test: 79 mẫu, không dùng cho Optuna, threshold selection, calibration
  selection hoặc quyết định kiến trúc.
- Split cố định, phân tầng, có hash và danh sách membership trong evidence.
- Outer fold membership được dùng chung cho benchmark công bằng.

### 6.4. Các kịch bản thời điểm dự báo

| Kịch bản | Thông tin cho phép | Ý nghĩa |
| --- | --- | --- |
| Late-stage | G1, G2 | Dự báo gần cuối kỳ, độ chính xác cao nhưng can thiệp muộn |
| Early-warning | Loại G2 | Cảnh báo sớm hơn, ít thông tin hơn |
| Pre-assessment | Loại G1 và G2 | Dự báo trước đánh giá, khó nhất |

Best OOF Macro-F1 đã ghi nhận là khoảng `0.6974` cho early-warning và `0.4344`
cho pre-assessment. Không so trực tiếp hai số này với late-stage để kết luận mô
hình suy giảm, vì feature availability khác nhau theo thiết kế.

## 7. Kiến trúc hệ thống

```text
CSV nguồn
  -> ingestion có kiểm soát
  -> PostgreSQL source_dataset_versions
     -> source_records
     -> source_record_targets
     -> split/run ledger
  -> DB-native loader(dataset_version_id)
  -> fold-local preprocessing
  -> nested model selection / fair benchmark
  -> frozen model configuration
  -> prediction + metrics + calibration + ordinal evaluation
  -> rule-based recommendation
  -> PostgreSQL persistence + final evidence bundle
```

### 7.1. Nguyên tắc PostgreSQL-first

- CSV chỉ được đọc bởi đường ingestion.
- Mọi đường chạy chính thức tải dữ liệu theo `dataset_version_id`.
- Target lưu riêng trong `source_record_targets`.
- Migration 003 đã được áp dụng trên database `student_predict`.
- Có 395 source records và 395 target rows, không duplicate và không orphan.
- Credentials chỉ đến từ environment; `.env` không được commit.
- DB-first verification tái tạo đúng toàn bộ 79 predicted classes.

### 7.2. Chống rò rỉ dữ liệu

- Preprocessor và scaler chỉ fit trên partition train tương ứng.
- Resampling, nếu có trong quy trình final selection, chỉ diễn ra bên trong train
  fold.
- Outer validation không được dùng cho early stopping hoặc class weight.
- Locked test không được đưa vào Optuna hay lựa chọn cấu hình.
- Feature availability được kiểm tra theo kịch bản.
- G3 và các biến dẫn xuất từ G3 bị chặn khỏi feature pipeline.

## 8. Kiến trúc CNN–BiLSTM cuối cùng

### 8.1. Đầu vào

Mỗi mẫu được biểu diễn thành tensor `[timesteps=2, channels=1]` theo thứ tự
`[G1, G2]`. Đây là chuỗi điểm ngắn, không phải time series dài.

### 8.2. Các tầng

| Thành phần | Cấu hình final | Vai trò |
| --- | --- | --- |
| Conv1D | 16 channels, kernel 1 | Biến đổi biểu diễn cục bộ từng thời điểm |
| BatchNorm + ReLU | sau convolution | Ổn định và phi tuyến hóa biểu diễn |
| Sequence dropout | 0.197248 | Regularization biểu diễn chuỗi |
| BiLSTM | hidden 32, 1 layer, bidirectional | Tổng hợp G1/G2 theo hai hướng |
| Head dropout | 0.456984 | Regularization trước classifier |
| Linear head | 64 -> 3 logits | Phân loại Low/Medium/High |
| Inference | Softmax + argmax | Xác suất và lớp dự đoán |

Tổng số tham số học được: 13.059.

### 8.3. Huấn luyện final

| Siêu tham số | Giá trị |
| --- | ---: |
| Seed | 42 |
| Batch size | 32 |
| Learning rate | 0.0046677139 |
| Weight decay | 0.0003541244 |
| Max epochs | 40 |
| Early-stopping patience | 12 |
| Scheduler patience | 3 |
| Resampling | none |
| Class weighting | none |
| Calibration | none |
| Decision rule | argmax |

Cấu hình lưu tên loss lịch sử `weighted_ce`, nhưng criterion thực tế là
CrossEntropyLoss không class weighting vì `class_weight_mode=none`.

### 8.4. Giới hạn diễn giải kiến trúc

CNN với kernel 1 không khai thác cửa sổ nhiều thời điểm; nó chủ yếu học phép
chiếu đặc trưng tại mỗi bước. Chuỗi dài hai bước cũng không đủ để tuyên bố mô
hình học được phụ thuộc thời gian dài hạn. Vai trò hợp lý của CNN–BiLSTM trong
đề tài là kiến trúc nghiên cứu kết hợp và bằng chứng khả thi, không phải bằng
chứng mặc định về ưu thế deep learning.

## 9. Giao thức lựa chọn mô hình final

1. Tạo split development/locked test cố định.
2. Chỉ sử dụng 316 development records cho model selection.
3. Dùng 5 outer stratified folds để ước lượng hiệu năng tổng quát hóa.
4. Trong mỗi outer train, dùng 3 inner folds và 30 Optuna trials.
5. Objective là mean inner-CV Macro-F1.
6. Early stopping dùng một tập nhỏ tách từ fold train, không dùng scoring fold.
7. Sau khi chọn epoch, refit trên toàn bộ partition train tương ứng.
8. Cấu hình cuối được đóng băng trước khi đánh giá locked test.
9. Final strategy là single seed 42, không ensemble, không calibration, argmax.

Macro-F1 được chọn vì bài toán có ba lớp và cần cân bằng vai trò giữa các lớp,
không để lớp Medium chiếm ưu thế như accuracy đơn thuần.

## 10. Benchmark so sánh không thiên vị

### 10.1. Danh sách mô hình

| Nhóm | Mô hình |
| --- | --- |
| Machine Learning | Decision Tree |
| Machine Learning | Random Forest |
| Machine Learning | Support Vector Machine với RBF kernel |
| Machine Learning | XGBoost |
| Machine Learning | Gradient Boosting Machine |
| Deep Learning | CNN+LSTM |
| Mô hình đề xuất | CNN+BiLSTM |

### 10.2. Điều kiện chung

- Cùng đầu vào G1 và G2.
- Cùng 316 development records.
- Cùng immutable outer-fold membership.
- 5 outer folds và 3 inner folds.
- 30 trial cho mỗi model/outer fold.
- Objective mean inner-CV Macro-F1.
- Seed 42 và random state xác định.
- Không resampling cho mọi mô hình.
- Không class weighting cho mọi mô hình.
- Argmax decision rule.
- Locked test không dùng cho selection hoặc evaluation benchmark.
- Scaler của mô hình cổ điển chỉ fit trên train fold.
- Preprocessing mạng sâu chỉ fit trên train fold.

Các mô hình có search space riêng vì siêu tham số không đồng nhất về ý nghĩa,
nhưng có cùng ngân sách trial và cùng cấu trúc nested CV. Đây là cách so sánh
hợp lý hơn việc dùng cấu hình mặc định cho baseline trong khi tuning sâu mô hình
đề xuất.

### 10.3. Kết quả benchmark

| Xếp hạng | Mô hình | Outer Macro-F1 | OOF Macro-F1 | OOF accuracy | Brier | ECE |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Random Forest | **0.8915 ± 0.0240** | 0.8932 | 0.8861 | 0.1733 | 0.0243 |
| 2 | Decision Tree | 0.8906 ± 0.0248 | 0.8919 | 0.8861 | 0.1874 | 0.0368 |
| 3 | SVM-RBF | 0.8894 ± 0.0290 | 0.8906 | 0.8829 | 0.1756 | 0.0327 |
| 4 | Gradient Boosting | 0.8872 ± 0.0290 | 0.8882 | 0.8829 | 0.1939 | 0.0867 |
| 5 | XGBoost | 0.8739 ± 0.0341 | 0.8748 | 0.8703 | 0.1910 | 0.0246 |
| 6 | CNN+BiLSTM | 0.8380 ± 0.0475 | 0.8366 | 0.8354 | 0.2412 | 0.0335 |
| 7 | CNN+LSTM | 0.7970 ± 0.1253 | 0.8123 | 0.8133 | 0.2627 | 0.0498 |

Mean paired Macro-F1 delta so với CNN–BiLSTM benchmark:

| Mô hình | Mean delta |
| --- | ---: |
| Random Forest | +0.0535 |
| Decision Tree | +0.0526 |
| SVM-RBF | +0.0514 |
| Gradient Boosting | +0.0492 |
| XGBoost | +0.0359 |
| CNN+LSTM | -0.0410 |

Tất cả năm baseline ML đều thắng CNN–BiLSTM ở cả năm outer folds. CNN+LSTM có
độ biến thiên lớn, đặc biệt một fold thấp, cho thấy kiến trúc recurrent một
hướng không ổn định trên tập dữ liệu nhỏ này. Với chỉ hai đầu vào số, các mô
hình ML có inductive bias phù hợp hơn và ít phụ thuộc vào tối ưu hóa gradient.

Không tuyên bố khác biệt có ý nghĩa thống kê theo nghĩa kiểm định giả thuyết chỉ
từ năm folds. Các delta là mô tả paired theo fold, không phải p-value.

## 11. Kết quả mô hình CNN–BiLSTM final

### 11.1. Nested CV

- Mean outer Macro-F1: `0.878089`.
- Standard deviation: `0.044829`.
- Fold scores: `0.9075`, `0.8232`, `0.8236`, `0.9171`, `0.9190`.

### 11.2. Locked test

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9114 |
| Macro-F1 | 0.9262 |
| Weighted F1 | 0.9122 |
| Balanced accuracy | 0.9345 |
| Quadratic weighted kappa | 0.9152 |
| Ordinal MAE | 0.0886 |
| Macro PR-AUC | 0.9699 |
| Multiclass Brier | 0.1683 |
| ECE | 0.0591 |

### 11.3. Kết quả theo lớp

| Lớp | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Low | 0.8065 | 0.9615 | 0.8772 | 26 |
| Medium | 0.9697 | 0.8421 | 0.9014 | 38 |
| High | 1.0000 | 1.0000 | 1.0000 | 15 |

Confusion matrix, hàng thật và cột dự đoán theo Low/Medium/High:

```text
[[25,  1,  0],
 [ 6, 32,  0],
 [ 0,  0, 15]]
```

Mô hình sai 7 trường hợp, toàn bộ là sai lệch giữa hai lớp kề nhau. Không có
trường hợp Low bị dự đoán thành High hoặc ngược lại.

### 11.4. Độ bất định

- Bootstrap resamples: 2.000, seed 42.
- Accuracy 95% CI: `0.8481–0.9620`.
- Macro-F1 95% CI: `0.8704–0.9694`.

CI tương đối rộng vì locked test chỉ có 79 mẫu. F1 bằng 1.0 của lớp High được đo
trên 15 mẫu và không được diễn giải là hiệu năng hoàn hảo trong quần thể.

## 12. Đối chiếu với baseline lịch sử

Trong evidence final trước benchmark mới:

- G2 threshold/logistic baseline có locked Macro-F1 `0.9365`.
- CNN–BiLSTM final có locked Macro-F1 `0.9262`.
- HistGradientBoosting full-feature có locked Macro-F1 `0.9463`.

HGB full-feature và CNN–BiLSTM G1/G2 không dùng cùng tập đặc trưng hoặc cùng mức
tuning, nên locked-test values này chỉ là so sánh hệ thống post-hoc, không phải
architecture comparison tuyệt đối được kiểm soát. Benchmark mới ở Mục 10 mới là
nguồn chính để so sánh kiến trúc trên cùng G1/G2.

## 13. Hệ thống khuyến nghị

### 13.1. Bản chất

`student_mat_rule_policy_v3` là policy luật xác định. Nó không phải mô hình học
máy và không được mô tả là recommender được huấn luyện.

### 13.2. Đầu vào và đầu ra

Đầu vào gồm predicted class, confidence và một số feature context được phép.
Đầu ra gồm risk band, risk factors, recommended actions, lý do, mức thận trọng
theo confidence và disclaimer yêu cầu human review.

### 13.3. Kết quả kiểm tra cấu trúc

Trên 79 trường hợp locked test:

- valid schema rate: 1.0;
- with explanation rate: 1.0;
- with specific action rate: 1.0;
- no contradiction rate: 1.0;
- no sensitive metadata leak rate: 1.0;
- cautious low-confidence rate: 1.0.

Risk-band coverage là High 31, Medium 33 và Low 15. Các yếu tố xuất hiện nhiều
gồm prior-grade gap, partial support gap, absences, low study time và failure
history.

Các tỷ lệ 100% chỉ chứng minh consistency cấu trúc và rule compliance. Chưa có
expert review và chưa có nghiên cứu can thiệp để chứng minh khuyến nghị làm tăng
điểm hoặc phù hợp với từng người học.

## 14. Tái lập và kiểm thử

### 14.1. Evidence identifiers

- Selection run: `nested-full-20260710`.
- Original scientific run: `a2945d79-9845-4979-b148-159f4853eca3`.
- Active DB-first verification run: `5a0b5041-5216-4a48-9e46-b0c16ab14866`.
- Active pointer: `artifacts/final/LATEST_RUN.txt`.
- Selected config SHA-256:
  `cda38460197627ac1d71e764f61d784e4c03cf6f86775339d38787c6890678ad`.
- Frozen prediction checksum:
  `d5b6f86d50a1a4c90b6a68139ec0eb6f4635e55c572c647d6d9b62d5a31f4a74`.

DB-first verification tạo đúng 79 predicted classes; maximum probability
absolute delta chỉ `2.78e-08` và metric delta bằng 0.

### 14.2. Test status

Sau khi dọn các giao thức thử nghiệm cũ, test suite hiện tại đạt 87 passed và 5
skipped. Các skipped tests cần PostgreSQL credentials/runtime phù hợp. Lần xác
minh live final trước đó có 62 passed, 0 skipped. Hai con số này phản ánh hai
phạm vi test khác nhau và không mâu thuẫn.

## 15. Giới hạn và đe dọa tính hợp lệ

### 15.1. Tính hợp lệ nội tại

- Mẫu nhỏ làm tăng variance của deep learning.
- Nhiều trial trên dữ liệu nhỏ vẫn có nguy cơ overfit validation dù dùng nested
  CV.
- Early stopping và stochastic optimization làm mạng sâu nhạy với seed.
- Benchmark dùng một seed chung; chưa phải multi-seed stability study.
- Năm outer folds chưa đủ để suy luận chắc chắn về statistical significance.

### 15.2. Tính hợp lệ ngoại tại

- Dữ liệu chỉ có 395 quan sát.
- Chỉ có một môn Toán.
- Dữ liệu thuộc bối cảnh trung học Bồ Đào Nha.
- Không đại diện cho sinh viên đại học Việt Nam.
- Cần huấn luyện/validation lại trên dữ liệu địa phương trước triển khai.

### 15.3. Tính hợp lệ cấu trúc

- G2 gần G3 nên late-stage performance không đồng nghĩa early intervention.
- Chuỗi G1/G2 dài hai bước, không hỗ trợ tuyên bố long-term temporal modeling.
- Các ngưỡng Low/Medium/High là lựa chọn operational, không phải chuẩn phổ quát.
- Recommendation rules là heuristic, không phải causal treatment policy.

### 15.4. Fairness

Fairness slices chỉ mang tính mô tả vì subgroup support nhỏ. Không nên tuyên bố
mô hình công bằng giữa các nhóm chỉ vì metric tổng thể cao. Các thuộc tính nhạy
cảm không được dùng để tự động sinh hành động khuyến nghị.

## 16. Đạo đức và điều kiện triển khai

- Hệ thống chỉ hỗ trợ quyết định, không thay thế giáo viên/cố vấn.
- Không tự động xếp hạng, kỷ luật, từ chối hỗ trợ hoặc gắn nhãn người học.
- Cần human-in-the-loop cho mọi hành động có tác động thực tế.
- Cần data minimization, phân quyền, mã hóa, audit log và quy trình xóa dữ liệu.
- Phải thông báo mục đích sử dụng và giới hạn cho người học.
- Không diễn giải yếu tố tương quan thành nguyên nhân.
- Cần expert review cho recommendation policy.
- Cần đánh giá drift, calibration và fairness định kỳ khi chuyển miền dữ liệu.

## 17. Kết luận khoa học được phép sử dụng

### 17.1. Có thể khẳng định

- CNN–BiLSTM khả thi về kỹ thuật trên dữ liệu nghiên cứu.
- Pipeline PostgreSQL-first và evidence bundle hỗ trợ tái lập.
- CNN–BiLSTM đạt hiệu năng locked-test tốt trong kịch bản late-stage.
- Sai số ordinal của mô hình nhỏ và không có lỗi nhảy hai mức trên locked test.
- G2 là tín hiệu dự báo late-stage rất mạnh.
- Trong benchmark chung G1/G2, Random Forest và các baseline ML vượt
  CNN–BiLSTM.
- Recommendation outputs đáp ứng các kiểm tra cấu trúc đã định nghĩa.

### 17.2. Không được khẳng định

- CNN–BiLSTM vượt mọi baseline.
- Deep learning là lựa chọn tốt nhất cho dữ liệu này.
- Dataset đại diện cho sinh viên đại học Việt Nam.
- F1 lớp High bằng 1.0 sẽ giữ nguyên trên dữ liệu mới.
- Khuyến nghị đã được chuyên gia xác nhận.
- Khuyến nghị gây ra cải thiện điểm số.
- Hệ thống sẵn sàng ra quyết định giáo dục tự động.
- Tương quan giữa absences/study time và kết quả là quan hệ nhân quả.

## 18. Gợi ý cấu trúc báo cáo

1. **Mở đầu:** động cơ, vấn đề, mục tiêu, câu hỏi nghiên cứu, phạm vi.
2. **Cơ sở lý thuyết:** classification, CNN, LSTM/BiLSTM, tree ensemble, SVM,
   class imbalance, nested CV, calibration, ordinal metrics.
3. **Dữ liệu và tiền xử lý:** nguồn, schema, nhãn, split, leakage controls,
   feature availability.
4. **Phương pháp đề xuất:** CNN–BiLSTM, loss, optimizer, early stopping,
   hyperparameter search.
5. **Thiết kế hệ thống:** PostgreSQL schema, lineage, ingestion, training,
   inference và recommendation.
6. **Thiết kế thực nghiệm:** final selection protocol và fair benchmark protocol
   phải được trình bày tách biệt.
7. **Kết quả:** nested CV, locked test, per-class, confusion matrix, calibration,
   ordinal metrics, CI và benchmark ML.
8. **Thảo luận:** vai trò G2, tại sao ML thắng, giới hạn chuỗi ngắn, variance và
   external validity.
9. **Khuyến nghị và đạo đức:** rule policy, structural evaluation,
   human-in-the-loop, privacy và fairness.
10. **Kết luận và hướng phát triển:** validation dữ liệu Việt Nam, multi-seed,
    external test, expert review và prospective intervention study.

## 19. Bảng truy xuất nguồn số liệu

| Nội dung báo cáo | Nguồn chính |
| --- | --- |
| Dataset, split, checksum | `artifacts/final/<active>/dataset_manifest.json`, `split_manifest.json` |
| Cấu hình CNN–BiLSTM | `artifacts/final/<active>/selected_config.json` |
| Nested CV final | `artifacts/final/<active>/outer_fold_metrics.json` |
| Locked-test classification | `classification_report.json`, `confusion_matrix.csv` |
| Ordinal metrics | `ordinal_metrics.json` |
| Calibration | `calibration_metrics.json`, `reliability_curve_data.csv` |
| Confidence intervals | `bootstrap_confidence_intervals.json` |
| Baseline lịch sử | `baseline_results.csv` |
| Fair model benchmark | `artifacts/baseline_comparison/fair-model-comparison-full/summary.csv` |
| Paired benchmark deltas | `paired_macro_f1_deltas.csv` trong cùng run |
| Recommendation | `recommendation_evaluation.json` |
| DB lineage | `database_run_manifest.json`, `run_manifest.json` |
| Reproducibility | `reproducibility_manifest.json`, checksum manifests |

`<active>` hiện là `final-5a0b5041-5216-4a48-9e46-b0c16ab14866`.

## 20. Trạng thái hiện tại và hướng phát triển

### Đã hoàn thành

- PostgreSQL source architecture và migrations 001–003.
- Ingestion và target separation.
- Nested model selection và frozen CNN–BiLSTM final.
- DB-first verification và evidence bundle.
- Fair comparison với 5 ML models, CNN+LSTM và CNN+BiLSTM.
- Rule-based recommendation policy và structural evaluation.
- Cleanup code/test/log của các giao thức thử nghiệm đã loại.
- Test suite hiện hành.

### Chưa hoàn thành

- Expert review của khuyến nghị.
- Validation trên dữ liệu đại học Việt Nam.
- External validation trên một cơ sở/dataset độc lập.
- Multi-seed benchmark chính thức cho mọi kiến trúc.
- Prospective evaluation về hiệu quả can thiệp.
- Production monitoring cho drift, calibration và fairness.

### Hướng phát triển ưu tiên

1. Thu thập dữ liệu địa phương theo thời gian với governance rõ ràng.
2. Định nghĩa lại thời điểm early-warning theo quy trình đào tạo thực tế.
3. Đánh giá Random Forest như production candidate đơn giản bên cạnh mô hình
   nghiên cứu CNN–BiLSTM.
4. Chạy multi-seed repeated nested CV và paired bootstrap ở cấp prediction.
5. Hiệu chỉnh xác suất trên validation data nếu calibration suy giảm.
6. Mời chuyên gia giáo dục đánh giá recommendation cases.
7. Thử nghiệm can thiệp có kiểm soát trước khi tuyên bố hiệu quả thực tế.

---

**Thông điệp trung tâm khi viết báo cáo:** Đề tài thành công ở việc xây dựng một
hệ thống dự đoán có kiểm soát, tái lập và có khả năng giải thích phạm vi sử dụng.
Kết quả benchmark cho thấy mô hình phức tạp không mặc nhiên tốt hơn baseline;
đây là một phát hiện khoa học cần được trình bày trung thực, không phải điểm yếu
cần che giấu.
