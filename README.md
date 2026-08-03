# Dự đoán thành tích và cảnh báo sớm sinh viên

Repository chính thức của khóa luận:

> **Xây dựng mô hình học kết hợp để dự đoán thành tích học tập sinh viên**

Đề tài xây dựng và đánh giá họ mô hình **Hybrid CNN–BiLSTM** trên ba bộ dữ liệu giáo dục, đồng thời so sánh công bằng với các mô hình Machine Learning truyền thống. Hệ thống phục vụ hai mục tiêu:

1. **Dự đoán kết quả học tập cuối cùng** của sinh viên.
2. **Cảnh báo sớm theo từng giai đoạn**, khi lượng thông tin quan sát được còn hạn chế.

Hybrid CNN–BiLSTM là đóng góp trung tâm của khóa luận. Tuy nhiên, repository không tuyên bố mô hình Hybrid luôn vượt mọi mô hình ML. Kết luận khoa học phù hợp là:

> Hybrid đạt hiệu năng cạnh tranh với nhóm mô hình tốt nhất, nằm trong top 3 trên cả ba dataset, vượt các biến thể CNN-only/BiLSTM-only trong thí nghiệm UCI và thể hiện rõ giá trị ở bài toán cảnh báo sớm OULAD.

---

## 1. Kết quả dự đoán chính thức

Đây là các **endpoint authority** chính thức dùng để báo cáo kết quả cuối của khóa luận:

| Dataset | Kiến trúc Hybrid chính thức | Bài toán | Macro-F1 |
|---|---|---|---:|
| Student-Mat | Frozen UCI CNN–BiLSTM | Low / Medium / High | **0.901460** |
| Student-Por | Frozen UCI CNN–BiLSTM | Low / Medium / High | **0.862259** |
| OULAD FINAL | H1 Tabular Residual CNN–BiLSTM | At-risk / Not-at-risk | **0.894071** |

Nguồn authority:

- `configs/final/final_model_authority.yaml`
- `reports/final/thesis_v3/01_FINAL_MAIN_RESULTS.md`
- `reports/final/thesis_v3/02_FULL_ML_VS_HYBRID.md`

Các kết quả OULAD cũ như `0.7984` và `0.828084` chỉ còn là **historical context**, không phải authority cuối hiện tại.

---

## 2. Phải tách endpoint chính thức và stage phụ

Kết quả endpoint và stage không được đặt cạnh nhau mà không giải thích vì chúng thuộc các authority/checkpoint khác nhau và trả lời các câu hỏi nghiên cứu khác nhau.

Ví dụ Student-Mat:

| Kết quả | Macro-F1 | Ý nghĩa |
|---|---:|---|
| Endpoint chính thức | **0.901460** | Mô hình chính thức cho nhiệm vụ dự đoán kết quả cuối |
| Stage S2 | **0.846139** | Mô hình stage-aware dùng chung cơ chế S0–S1–S2 |

Hai số này **không thể được diễn giải là mô hình giảm từ 0.9015 xuống 0.8461**.

- Endpoint `0.901460` trả lời: mô hình cuối dự đoán thành tích tốt đến đâu khi được đánh giá theo authority chính thức?
- Stage S2 `0.846139` trả lời: trong hệ thống cảnh báo theo giai đoạn, khi đã quan sát G1 và G2, mô hình dùng chung cho S0–S1–S2 hoạt động tốt đến đâu?

Stage experiment được thiết kế để đánh giá sự cải thiện khi thông tin xuất hiện dần, không thay thế endpoint chính thức.

---

## 3. Bộ dữ liệu và nhiệm vụ dự đoán

### UCI Student Performance

Hai bộ dữ liệu có cấu trúc đặc trưng tương đồng:

- `Student-Mat`: kết quả môn Toán, 395 sinh viên.
- `Student-Por`: kết quả môn Tiếng Bồ Đào Nha, 649 sinh viên.

Nhãn chính được tạo từ điểm cuối `G3`:

- `Low`: `0 <= G3 < 10`
- `Medium`: `10 <= G3 < 15`
- `High`: `15 <= G3 <= 20`

`G3` chỉ được dùng để tạo nhãn và **bị cấm làm predictor**.

### OULAD

OULAD là dữ liệu học tập trực tuyến theo thời gian. Bài toán chính là phân loại nhị phân:

- `At-risk`: sinh viên có nguy cơ.
- `Not-at-risk`: sinh viên không thuộc nhóm nguy cơ.

OULAD sử dụng chính sách `STRICT_REAL_TIME`:

- 47 kênh temporal.
- 165 đặc trưng aggregate.
- Loại trừ giá trị điểm số.
- Loại trừ các aggregate được suy ra từ điểm số.
- Mỗi feature phải tồn tại trước cutoff của stage tương ứng.

---

## 4. Pipeline tổng quát

```text
Dữ liệu gốc
    ↓
Kiểm tra schema và làm sạch
    ↓
Tạo nhãn dự đoán
    ↓
Áp dụng chính sách feature theo thời điểm
    ↓
Chia outer cross-validation
    ↓
Fit preprocessing chỉ trên tập train
    ↓
Tạo nhánh temporal và nhánh tabular/context
    ↓
CNN trích xuất mẫu cục bộ
    ↓
BiLSTM học quan hệ theo trình tự
    ↓
Fusion với đặc trưng tĩnh/aggregate
    ↓
Đầu phân loại tạo xác suất
    ↓
Chọn checkpoint bằng inner validation
    ↓
Ensemble nhiều seed
    ↓
Ghép out-of-fold predictions
    ↓
Tính Macro-F1, Balanced Accuracy, PR-AUC và các metric khác
    ↓
Sinh risk profile và khuyến nghị có kiểm soát
```

Nguyên tắc quan trọng nhất là **outer test không được dùng để chọn mô hình, checkpoint, cấu hình hay threshold**.

---

## 5. Pipeline UCI chi tiết

### Bước 1 — Đọc và kiểm tra dữ liệu

Mỗi dòng đại diện cho một sinh viên, gồm các nhóm thông tin:

- Cá nhân và trường học.
- Gia đình và hỗ trợ giáo dục.
- Thời gian học, số lần trượt và số lần nghỉ.
- Điểm quá trình G1, G2.
- Điểm cuối G3 dùng để tạo nhãn.

### Bước 2 — Tạo nhãn ba lớp

`G3` được chuyển thành `Low`, `Medium`, `High`. Sau khi tạo nhãn, G3 không được đưa vào đầu vào mô hình để tránh target leakage.

### Bước 3 — Chia dữ liệu theo outer folds

UCI sử dụng protocol đóng băng:

- 5 outer folds.
- 5 seed cố định: `42, 1201, 2026, 3407, 7319`.
- Dự đoán cuối được ghép từ các outer test folds.

Với Student-Mat, mỗi vòng gần đúng sử dụng bốn fold để train và một fold để test. Sau năm vòng, mỗi sinh viên được làm outer test đúng một lần.

### Bước 4 — Tiền xử lý chỉ trên tập train

Encoder, scaler và các phép biến đổi chỉ được fit trên training partition của fold hiện tại:

```text
Outer train → fit preprocessing
Outer test  → transform bằng preprocessing đã học từ train
```

Cách làm này ngăn dữ liệu test ảnh hưởng đến quá trình huấn luyện.

### Bước 5 — Tạo hai nhóm đầu vào

#### Nhánh temporal

G1 và G2 được tổ chức thành chuỗi ngắn theo quá trình học:

```text
G1 → G2
```

#### Nhánh context

Các đặc trưng tĩnh như thông tin cá nhân, gia đình, thời gian học, số lần nghỉ và số lần trượt được xử lý ở nhánh context/tabular.

### Bước 6 — CNN trích xuất mẫu cục bộ

CNN tìm các mẫu ngắn trong quá trình điểm số, chẳng hạn:

- G1 thấp và G2 tăng.
- G1 cao và G2 ổn định.
- G1 trung bình nhưng G2 giảm.

CNN có thể được hiểu là bộ lọc phát hiện các kiểu biến động gần nhau.

### Bước 7 — BiLSTM học quan hệ theo trình tự

BiLSTM xử lý chuỗi theo hai hướng để học quan hệ giữa các thời điểm. Mục tiêu là nhận diện xu hướng cải thiện, ổn định hoặc suy giảm.

Do UCI chỉ có chuỗi G1–G2 rất ngắn, lợi thế thời gian của CNN–BiLSTM không được khai thác mạnh như trên OULAD. Đây cũng là lý do các mô hình cây vẫn rất cạnh tranh trên UCI.

### Bước 8 — Fusion với đặc trưng context

Biểu diễn temporal từ CNN–BiLSTM được kết hợp với biểu diễn context:

```text
G1, G2 → CNN → BiLSTM ───────┐
                              ├→ Fusion → P(Low, Medium, High)
Đặc trưng tĩnh → Context MLP ─┘
```

### Bước 9 — Transfer learning giữa hai môn

Cấu hình UCI chính thức sử dụng:

- `transfer_learning: true`
- `shared_trunk: true`
- `subject_specific_head: true`

Cơ chế này chuyển **biểu diễn/trọng số đã học**, không chuyển trực tiếp các mẫu test giữa Student-Mat và Student-Por.

```text
Biểu diễn chung về hành vi học tập
            ↓
Shared CNN–BiLSTM trunk
       ┌────┴────┐
MAT-specific   POR-specific
    head           head
```

Dữ liệu của từng môn vẫn được đánh giá theo fold và authority riêng. Không được hiểu transfer learning là trộn tập test hoặc sao chép sinh viên từ bộ này sang bộ kia.

### Bước 10 — Chọn checkpoint bằng inner validation

Trong mỗi outer training fold, inner validation được dùng để:

- Theo dõi chất lượng mô hình.
- Chọn checkpoint.
- Kiểm soát overfitting.

Outer test chỉ được mở để đánh giá sau khi quá trình lựa chọn đã kết thúc.

### Bước 11 — Ensemble nhiều seed

Mỗi seed tạo một bộ xác suất. Dự đoán cuối lấy trung bình xác suất:

```text
Final probability = mean(probability_seed_1, ..., probability_seed_5)
```

Đây là `mean_probability ensemble`, giúp giảm ảnh hưởng của khởi tạo ngẫu nhiên. Ensemble không phải là oversampling và không tạo thêm sinh viên giả.

### Bước 12 — Out-of-fold evaluation

Các dự đoán outer test được ghép thành một tập OOF hoàn chỉnh để tính:

- Accuracy.
- Balanced Accuracy.
- Macro Precision, Macro Recall, Macro-F1.
- Weighted-F1.
- PR-AUC và ROC-AUC.
- NLL, Brier score và ECE.
- Confusion matrix và metric từng lớp.

---

## 6. Pipeline OULAD chi tiết

OULAD sử dụng kiến trúc **H1 Tabular Residual CNN–BiLSTM** với 160,492 tham số.

### Đầu vào

```text
Chuỗi hành vi theo tuần: 47 temporal channels
Đặc trưng tổng hợp:      165 aggregate features
Thông tin tĩnh:          static/context features
```

### Nhánh temporal

CNN và BiLSTM học diễn biến hành vi theo thời gian. Masked pooling bảo đảm mô hình chỉ tổng hợp các tuần đã quan sát tại stage hiện tại.

### Nhánh tabular residual expert

Đặc trưng aggregate/tabular được đưa qua một expert riêng. Expert này học phần thông tin mà nhánh temporal chưa giải thích tốt và hiệu chỉnh đầu ra bằng residual có giới hạn.

```text
Temporal CNN–BiLSTM → base risk logit ─┐
                                       ├→ final risk probability
Tabular residual expert → correction ──┘
```

### Fusion và auxiliary objectives

Mô hình kết hợp các nhánh temporal, aggregate và static bằng gated residual fusion. Các nhiệm vụ phụ về survival/outcome hỗ trợ backbone học biểu diễn có ý nghĩa hơn, trong khi bài toán chính vẫn là dự đoán `At-risk / Not-at-risk`.

### Protocol đánh giá

- 3 outer folds.
- 5 seed cố định.
- Cấu trúc mô hình đã đóng băng.
- Chọn mô hình bằng inner validation.
- Threshold được chọn từ pooled inner-OOF để tối đa Macro-F1.
- Outer label không được dùng để chọn threshold.

### Chính sách strict real-time

Các stage:

- `E1`: quan sát 20% tiến trình.
- `E2`: quan sát 35% tiến trình.
- `M1`: quan sát 50% tiến trình.
- `L1`: quan sát 75% tiến trình.
- `FINAL`: authority cuối.

Tập feature có tính lồng nhau:

```text
F20 ⊂ F35 ⊂ F50 ⊂ F75 ⊂ FFINAL
```

Thông tin tương lai không được đi ngược vào stage sớm.

---

## 7. Xử lý mất cân bằng lớp

### Kết luận chính thức

Authority/config cuối của repository **không khai báo SMOTE, random oversampling, undersampling hoặc weighted sampler**. Vì vậy README không tuyên bố nhóm đã tạo mẫu tổng hợp để cân bằng dữ liệu.

Vấn đề mất cân bằng được kiểm soát chủ yếu ở ba mức sau.

### 7.1. Mức đánh giá

Không chỉ dùng Accuracy. Các metric chính gồm:

- **Macro-F1**: mỗi lớp có trọng số ngang nhau.
- **Balanced Accuracy**: trung bình recall của các lớp.
- **PR-AUC**: phù hợp khi quan tâm lớp thiểu số/nguy cơ.
- Precision, Recall và F1 từng lớp.
- Confusion matrix.

Macro-F1 được ưu tiên vì lớp ít mẫu vẫn đóng góp ngang với lớp nhiều mẫu:

```text
Macro-F1 = (F1 lớp 1 + F1 lớp 2 + ... + F1 lớp K) / K
```

### 7.2. Mức lựa chọn mô hình

Checkpoint/candidate được chọn bằng inner validation dựa trên metric cân bằng thay vì chỉ tối đa Accuracy. Outer test không được dùng để điều chỉnh mô hình.

### 7.3. Mức quyết định của OULAD

OULAD không mặc định cố định threshold `0.5`. Threshold được chọn trên pooled inner-OOF để tối đa Macro-F1, sau đó mới khóa và đánh giá trên outer test.

Đây là xử lý mất cân bằng ở **mức ra quyết định**, không phải tạo thêm mẫu.

Cách trả lời vấn đáp phù hợp:

> Nhóm không sử dụng SMOTE trong authority chính thức. Mất cân bằng được kiểm soát bằng Macro-F1, Balanced Accuracy, PR-AUC, metric từng lớp và threshold được chọn trên inner validation đối với OULAD.

---

## 8. So sánh Machine Learning và Hybrid

Macro-F1 trên endpoint authority chính thức:

| Model | Student-Mat | Student-Por | OULAD FINAL |
|---|---:|---:|---:|
| Logistic Regression | 0.879318 | 0.820541 | 0.891358 |
| Decision Tree | **0.906654** | 0.848718 | 0.875871 |
| Random Forest | 0.901387 | **0.869244** | 0.889279 |
| HistGradientBoosting | 0.878546 | 0.850630 | 0.891350 |
| SVM | 0.814271 | 0.782477 | 0.892274 |
| XGBoost | 0.888000 | 0.866388 | 0.892991 |
| MLP | 0.859507 | 0.830399 | **0.895349** |
| **Hybrid CNN–BiLSTM** | **0.901460** | **0.862259** | **0.894071** |

### Diễn giải trung thực

- Student-Mat: Decision Tree cao nhất; Hybrid xếp thứ 2 và gần như ngang Random Forest.
- Student-Por: Random Forest cao nhất; Hybrid xếp thứ 3 sau Random Forest và XGBoost.
- OULAD: MLP cao nhất; Hybrid xếp thứ 2 với chênh lệch Macro-F1 rất nhỏ.
- Hybrid là mô hình duy nhất nằm trong top 3 ở cả ba dataset.

Do đó, luận văn không dùng claim “Hybrid luôn thắng ML”. Claim phù hợp hơn là:

> Hybrid có hiệu năng ổn định và cạnh tranh trên nhiều loại dữ liệu, đồng thời cung cấp kiến trúc thống nhất để học cả đặc trưng thời gian và đặc trưng tabular.

---

## 9. Kết quả UCI theo giai đoạn

### Hybrid CNN–BiLSTM

| Dataset | Stage | Thông tin được phép quan sát | Macro-F1 |
|---|---|---|---:|
| Student-Mat | S0 | Không có G1/G2 | 0.413558 |
| Student-Mat | S1 | Chỉ có G1 | 0.743811 |
| Student-Mat | S2 | Có G1 và G2 | 0.846139 |
| Student-Por | S0 | Không có G1/G2 | 0.508886 |
| Student-Por | S1 | Chỉ có G1 | 0.754180 |
| Student-Por | S2 | Có G1 và G2 | 0.851947 |

Ý nghĩa chính của bảng này là chất lượng dự đoán tăng khi thông tin học tập xuất hiện dần.

- S0 khó nhất vì chưa có điểm quá trình.
- G1 giúp kết quả tăng mạnh ở S1.
- G2 bổ sung tín hiệu quan trọng ở S2.

Các stage này là **secondary evidence**, không thay thế endpoint `0.901460` và `0.862259`.

---

## 10. Kết quả OULAD theo giai đoạn

| Stage | Tiến trình quan sát | Hybrid Macro-F1 |
|---|---:|---:|
| E1 | 20% | 0.707682 |
| E2 | 35% | 0.748207 |
| M1 | 50% | 0.795071 |
| L1 | 75% | 0.852491 |
| FINAL | Authority cuối | **0.894071** |

Ở E1, Hybrid có Risk Recall cao hơn nhiều mô hình ML mạnh, cho thấy khả năng phát hiện sinh viên nguy cơ sớm tốt hơn nhưng phải chấp nhận nhiều cảnh báo sai hơn. Khi tiến trình học tăng, độ chính xác và Macro-F1 cải thiện rõ rệt.

Tại FINAL:

| Metric | H1 Hybrid |
|---|---:|
| Accuracy | 0.907674 |
| Balanced Accuracy | 0.879360 |
| Macro-F1 | 0.894071 |
| PR-AUC | 0.934988 |
| ROC-AUC | 0.944963 |
| Risk Precision | 0.941290 |
| Risk Recall | 0.785069 |
| Risk F1 | 0.856111 |
| Specificity | 0.973650 |
| ECE | 0.007871 |

Hybrid có Risk Precision và Specificity cao, nghĩa là mô hình khá thận trọng: khi phát cảnh báo nguy cơ thì thường chính xác, nhưng vẫn có thể bỏ sót một phần sinh viên at-risk.

---

## 11. Ý nghĩa của từng nhóm mô hình

- **Decision Tree:** mạnh trên Student-Mat, dễ giải thích và phù hợp với các ngưỡng điểm rõ ràng.
- **Random Forest:** ổn định trên dữ liệu tabular nhỏ và tốt nhất trên Student-Por.
- **XGBoost:** comparator mạnh, ổn định và có chất lượng xác suất tốt.
- **Logistic Regression:** cạnh tranh trên OULAD, cho thấy feature engineering tạo tín hiệu gần tuyến tính đáng kể.
- **MLP:** tốt nhất ở OULAD FINAL nhưng yếu hơn trên hai bộ UCI nhỏ.
- **SVM:** yếu trên UCI nhưng cạnh tranh trên OULAD.
- **CNN-only/BiLSTM-only:** dùng làm ablation để kiểm tra đóng góp của từng thành phần.
- **Hybrid CNN–BiLSTM:** kết hợp pattern cục bộ, quan hệ thời gian và thông tin tabular; là kiến trúc trọng tâm của khóa luận.

---

## 12. Protocol khoa học và chống data leakage

Repository tuân thủ các nguyên tắc:

- Preprocessing chỉ fit trên training partition.
- G3 không được dùng làm predictor.
- OULAD loại bỏ score values và score-derived aggregates.
- Outer test không dùng để chọn candidate/checkpoint/threshold.
- UCI sử dụng five-fold OOF evaluation và nhiều seed cố định.
- OULAD sử dụng 3 outer folds × 5 seeds.
- Kết quả cuối được đóng băng bằng prediction artifact, checkpoint, config và checksum.
- Validation release chỉ replay evidence, không âm thầm train lại.

Lưu ý: đây là nested cross-validation benchmark, không phải một external holdout hoàn toàn độc lập chưa từng xuất hiện trong quá trình phát triển.

---

## 13. Hạn chế cần trình bày trung thực

1. UCI có kích thước nhỏ và chuỗi temporal chỉ gồm G1–G2, nên lợi thế sequence modeling bị hạn chế.
2. Các mô hình cây có thể khai thác rất tốt các quan hệ ngưỡng và phi tuyến trên dữ liệu tabular nhỏ.
3. Student-Mat Hybrid có ECE cao hơn một số comparator; xác suất cần được diễn giải thận trọng nếu dùng để ra quyết định can thiệp.
4. Pipeline chính thức không tuyên bố đã dùng SMOTE hoặc oversampling.
5. Kết quả là bằng chứng dự đoán, không chứng minh quan hệ nhân quả giữa khuyến nghị và sự cải thiện học tập.
6. Endpoint và stage phải luôn được báo cáo tách biệt.

---

## 14. Mô-đun khuyến nghị

Mô-đun khuyến nghị tiêu thụ các prediction đã đóng băng để tạo:

```text
Prediction
    ↓
Risk profile
    ↓
Recommendation plan
    ↓
Actions có ràng buộc
    ↓
PostgreSQL / evidence artifacts
```

Khuyến nghị là policy dựa trên evidence, không thay đổi checkpoint hoặc metric của mô hình dự đoán. Coverage của khuyến nghị không được gọi là Accuracy và chưa được diễn giải như hiệu quả nhân quả.

---

## 15. Chạy validation

Cài môi trường từ `requirements.txt` hoặc `requirements-lock.txt`, sau đó chạy:

```powershell
python project.py final status
python project.py final report
python project.py final validate
python project.py pipeline uci validate
python project.py pipeline oulad validate
pytest
```

Các lệnh validation chỉ đọc/replay evidence cuối. Training chỉ chạy khi được gọi rõ qua pipeline train tương ứng.

---

## 16. Cấu trúc repository

```text
artifacts/final/          prediction, metric, checkpoint và checksum cuối
artifacts/canonical_v3/   OULAD canonical predictions, metrics và checkpoints
configs/final/            model authority và cấu hình đóng băng
data/                     dữ liệu và manifest được phép sử dụng
database/final/           PostgreSQL schema và migration cuối
docs/                     tài liệu kiến trúc, protocol và project map
reports/final/thesis_v3/  báo cáo authority dùng cho luận văn
scripts/                  replay, validation và pipeline scripts
src/                      model, training, evaluation và recommendation code
tests/                    kiểm thử tự động
project.py                CLI điều phối pipeline và validation
```

---

## 17. Các tệp authority quan trọng

| Mục đích | Đường dẫn |
|---|---|
| Authority mô hình cuối | `configs/final/final_model_authority.yaml` |
| Cấu hình Student-Mat | `configs/final/cnn_bilstm_mat.yaml` |
| Cấu hình Student-Por | `configs/final/cnn_bilstm_por.yaml` |
| Cấu hình OULAD H1 | `configs/final/h1_tabular_residual_oulad.yaml` |
| Kết quả Hybrid chính thức | `reports/final/thesis_v3/01_FINAL_MAIN_RESULTS.md` |
| So sánh ML và Hybrid | `reports/final/thesis_v3/02_FULL_ML_VS_HYBRID.md` |
| Kết quả stage UCI | `reports/final/thesis_v3/03_UCI_STAGE_RESULTS.md` |
| Kết quả stage OULAD | `reports/final/thesis_v3/04_OULAD_STAGE_RESULTS.md` |
| Registry kiến trúc cuối | `reports/final/thesis_v3/10_FINAL_MODEL_ARCHITECTURES.md` |
| Sơ đồ mã nguồn | `docs/project_map/PROJECT_CODE_MAP.md` |

---

## 18. Tuyên bố kết quả cuối

- Student-Mat Hybrid Macro-F1: **0.901460**.
- Student-Por Hybrid Macro-F1: **0.862259**.
- OULAD H1 Hybrid Macro-F1: **0.894071**.
- Hybrid không luôn đứng hạng 1 nhưng nằm trong top 3 trên cả ba dataset.
- Stage evidence không thay thế endpoint authority.
- Transfer learning chuyển biểu diễn/trọng số, không trộn outer test samples.
- Mất cân bằng được kiểm soát bằng metric, protocol lựa chọn và threshold; không tuyên bố sử dụng SMOTE.
- Kết quả cuối được đóng băng và có thể replay/validate từ artifact hiện có.
