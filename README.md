# Dự đoán thành tích và cảnh báo sớm sinh viên

Repository chính thức của khóa luận:

> **Xây dựng mô hình học kết hợp để dự đoán thành tích học tập sinh viên**

Đề tài xây dựng họ mô hình **Hybrid CNN–BiLSTM** để dự đoán kết quả học tập và phát hiện sớm sinh viên có nguy cơ. Mô hình được đánh giá trên ba bộ dữ liệu:

- **Student-Mat**: dự đoán kết quả môn Toán theo ba mức `Low / Medium / High`.
- **Student-Por**: dự đoán kết quả môn Tiếng Bồ Đào Nha theo ba mức `Low / Medium / High`.
- **OULAD**: dự đoán sinh viên `At-risk / Not-at-risk` từ hành vi học tập theo thời gian.

Hybrid CNN–BiLSTM là mô hình trung tâm của khóa luận. Kết luận của đề tài không phải là Hybrid luôn thắng mọi mô hình Machine Learning, mà là:

> **Hybrid đạt hiệu năng cạnh tranh với nhóm mô hình tốt nhất, nằm trong top 3 trên cả ba dataset, vượt các biến thể CNN-only và BiLSTM-only trong thí nghiệm UCI, đồng thời thể hiện rõ giá trị ở bài toán cảnh báo sớm OULAD.**

---

## 1. Kết quả mô hình cuối

Các kết quả dưới đây là **authority chính thức** dùng để báo cáo kết quả cuối của khóa luận.

| Dataset | Mô hình Hybrid chính thức | Accuracy | Balanced Accuracy | Macro-F1 | PR-AUC |
|---|---|---:|---:|---:|---:|
| Student-Mat | Frozen UCI CNN–BiLSTM | 0.891139 | 0.902089 | **0.901460** | 0.944184 |
| Student-Por | Frozen UCI CNN–BiLSTM | 0.889060 | 0.867576 | **0.862259** | 0.914679 |
| OULAD FINAL | H1 Tabular Residual CNN–BiLSTM | 0.907674 | 0.879360 | **0.894071** | 0.934988 |

Đối với OULAD, mô hình cuối còn đạt:

- Risk Precision: **0.941290**.
- Risk Recall: **0.785069**.
- Risk F1: **0.856111**.
- Specificity: **0.973650**.
- ECE: **0.007871**.

Các kết quả OULAD cũ như `0.7984` và `0.828084` chỉ còn là dữ liệu lịch sử, không phải kết quả cuối hiện tại.

Nguồn authority:

- `configs/final/final_model_authority.yaml`
- `reports/final/thesis_v3/01_FINAL_MAIN_RESULTS.md`
- `reports/final/thesis_v3/02_FULL_ML_VS_HYBRID.md`

---

## 2. Kiến trúc mô hình Hybrid cuối

### 2.1. Hybrid CNN–BiLSTM cho Student-Mat và Student-Por

Mô hình UCI nhận hai nhóm thông tin.

```text
Điểm theo tiến trình G1, G2
        ↓
      CNN
        ↓
     BiLSTM
        ↓
Biểu diễn quá trình học ──────────────┐
                                      ├→ Fusion → Classification head
Đặc trưng cá nhân, gia đình, hành vi ─┘
        ↓
      MLP
```

#### Nhánh thời gian

- `G1` và `G2` được tổ chức thành chuỗi ngắn theo thứ tự học tập.
- CNN nhận diện các mẫu cục bộ như điểm tăng, giảm hoặc duy trì.
- BiLSTM học quan hệ theo trình tự giữa các thời điểm.

#### Nhánh ngữ cảnh

Các đặc trưng tĩnh như thông tin cá nhân, gia đình, thời gian học, số lần nghỉ, số lần trượt và hỗ trợ giáo dục được xử lý qua nhánh MLP.

#### Fusion

Biểu diễn từ nhánh thời gian và nhánh ngữ cảnh được kết hợp để dự đoán ba xác suất:

```text
P(Low), P(Medium), P(High)
```

Lớp có xác suất lớn nhất được chọn làm kết quả cuối.

#### Transfer learning

Họ mô hình UCI sử dụng phần thân dùng chung và đầu ra riêng theo từng môn học:

```text
Shared CNN–BiLSTM trunk
          ↓
Kiến thức chung về quá trình học tập
       ┌──────────────┴──────────────┐
MAT-specific head              POR-specific head
```

Transfer learning ở đây là chuyển hoặc dùng lại **trọng số và biểu diễn đã học**, không phải sao chép mẫu dữ liệu từ bộ này sang bộ kia. Dữ liệu test của mỗi bộ vẫn được giữ riêng và không được dùng để huấn luyện.

### 2.2. H1 Tabular Residual CNN–BiLSTM cho OULAD

OULAD có dữ liệu theo tuần nên phù hợp hơn với kiến trúc chuỗi thời gian.

```text
47 temporal channels → CNN → BiLSTM → Masked pooling ─┐
                                                       ├→ Gated fusion → Risk probability
165 aggregate features → Tabular residual expert ─────┤
Static/context features ───────────────────────────────┘
```

Mô hình gồm:

- Nhánh temporal học sự thay đổi hành vi học tập theo tuần.
- CNN nhận diện mẫu hoạt động cục bộ.
- BiLSTM học xu hướng dài hơn trong quá trình học.
- Masked pooling chỉ tổng hợp các tuần đã được phép quan sát.
- Nhánh tabular residual expert học phần thông tin mà nhánh temporal chưa giải thích tốt.
- Gated residual fusion kết hợp temporal, aggregate và static features.
- Các nhiệm vụ phụ survival/outcome hỗ trợ backbone học biểu diễn tốt hơn.

Kiến trúc cuối có **160,492 tham số** và sử dụng chính sách `STRICT_REAL_TIME`.

---

## 3. Pipeline dự đoán UCI từng bước

### Bước 1 — Đọc và kiểm tra dữ liệu

Mỗi dòng đại diện cho một sinh viên. Dữ liệu gồm thông tin cá nhân, gia đình, hành vi học tập, số lần nghỉ, điểm G1, G2 và điểm cuối G3.

### Bước 2 — Tạo nhãn

G3 chỉ được dùng để tạo nhãn:

- `Low`: `0 <= G3 < 10`.
- `Medium`: `10 <= G3 < 15`.
- `High`: `15 <= G3 <= 20`.

**G3 bị cấm làm predictor**, vì sử dụng G3 để dự đoán chính nhãn tạo từ G3 sẽ gây data leakage.

### Bước 3 — Chia dữ liệu thành outer folds

UCI sử dụng 5 outer folds. Mỗi vòng:

- 4 folds dùng để huấn luyện.
- 1 fold giữ lại để đánh giá.

Sau 5 vòng, mỗi sinh viên được làm mẫu test đúng một lần.

### Bước 4 — Fit preprocessing trên tập train

Encoder, scaler và các phép biến đổi chỉ được học từ training partition của fold hiện tại.

```text
Outer train → fit preprocessing
Outer test  → transform bằng preprocessing đã fit
```

Không fit preprocessing trên toàn bộ dataset trước khi chia fold.

### Bước 5 — Tạo hai nhánh đầu vào

- G1, G2 tạo thành nhánh thời gian.
- Các đặc trưng còn lại tạo thành nhánh context/tabular.

### Bước 6 — CNN trích xuất mẫu cục bộ

CNN phát hiện các kiểu biến động ngắn trong chuỗi điểm.

### Bước 7 — BiLSTM học quan hệ theo trình tự

BiLSTM học cách G1 và G2 liên hệ với nhau và phản ánh xu hướng học tập.

### Bước 8 — MLP xử lý đặc trưng tĩnh

Thông tin cá nhân, gia đình và hành vi học tập được mã hóa thành biểu diễn ngữ cảnh.

### Bước 9 — Fusion và classification head

Hai nhánh được kết hợp để tạo xác suất cho ba lớp Low, Medium và High.

### Bước 10 — Chọn checkpoint bằng inner validation

Checkpoint và candidate được chọn trên inner validation. Outer test không được dùng để chọn epoch, threshold hoặc cấu hình mô hình.

### Bước 11 — Huấn luyện nhiều seed và ensemble

Mỗi mô hình được đánh giá với các seed cố định:

```text
42, 1201, 2026, 3407, 7319
```

Xác suất của các seed được lấy trung bình:

```text
Final probability = mean(probability của các seed)
```

Ensemble giúp giảm ảnh hưởng của khởi tạo ngẫu nhiên và làm dự đoán ổn định hơn.

### Bước 12 — Out-of-fold evaluation

Dự đoán của các outer test folds được ghép thành một tập OOF hoàn chỉnh để tính Accuracy, Balanced Accuracy, Macro-F1, PR-AUC, ROC-AUC, NLL, Brier score, ECE và confusion matrix.

---

## 4. Pipeline dự đoán OULAD từng bước

### Bước 1 — Tạo dữ liệu theo thời gian

Hoạt động của sinh viên được gom theo tuần và tạo thành 47 temporal channels.

### Bước 2 — Tạo aggregate và static features

Hệ thống tạo 165 aggregate features cùng các đặc trưng tĩnh phù hợp với thời điểm quan sát.

### Bước 3 — Áp dụng cutoff thời gian

Các stage chỉ được sử dụng thông tin xuất hiện trước cutoff tương ứng:

- `E1`: 20% tiến trình.
- `E2`: 35% tiến trình.
- `M1`: 50% tiến trình.
- `L1`: 75% tiến trình.
- `FINAL`: dữ liệu cuối hợp lệ.

Tập feature có tính lồng nhau:

```text
F20 ⊂ F35 ⊂ F50 ⊂ F75 ⊂ FFINAL
```

Score values và score-derived aggregates không được sử dụng trong authority strict real-time.

### Bước 4 — CNN và BiLSTM học hành vi theo tuần

CNN tìm mẫu hoạt động cục bộ; BiLSTM học xu hướng dài hơn của quá trình tương tác học tập.

### Bước 5 — Masked pooling

Mô hình chỉ tổng hợp các tuần đã xuất hiện, ngăn thông tin của tương lai đi vào stage sớm.

### Bước 6 — Tabular residual expert

Nhánh expert xử lý aggregate features và tạo phần hiệu chỉnh residual cho dự đoán từ temporal backbone.

### Bước 7 — Fusion và auxiliary objectives

Các nhánh temporal, aggregate và static được kết hợp bằng gated residual fusion. Các nhiệm vụ phụ hỗ trợ học biểu diễn nhưng đầu ra chính vẫn là `At-risk / Not-at-risk`.

### Bước 8 — Nested evaluation

OULAD sử dụng:

- 3 outer folds.
- 5 seed cố định.
- Chọn mô hình bằng inner validation.
- Không dùng outer label để tuning.

### Bước 9 — Chọn threshold

Threshold được chọn từ pooled inner-OOF để tối đa Macro-F1, sau đó được khóa trước khi đánh giá outer test.

### Bước 10 — Tính kết quả cuối

Các outer predictions được ghép lại để tính Macro-F1, PR-AUC, Risk Precision, Risk Recall, Risk F1, Specificity và calibration metrics.

---

## 5. Kết quả cảnh báo theo từng thời điểm

Các kết quả dưới đây dùng để trả lời câu hỏi:

> Khi hệ thống mới quan sát được một phần quá trình học, mô hình có thể cảnh báo sớm tốt đến đâu?

### 5.1. Student-Mat và Student-Por

| Dataset | S0: chưa có G1/G2 | S1: có G1 | S2: có G1 và G2 |
|---|---:|---:|---:|
| Student-Mat | 0.413558 | 0.743811 | 0.846139 |
| Student-Por | 0.508886 | 0.754180 | 0.851947 |

Các giá trị trên là Macro-F1 của nhánh nghiên cứu stage-aware. Mô hình stage-aware dùng một cơ chế chung qua S0, S1 và S2 để đo mức cải thiện khi thông tin xuất hiện dần.

Vì vậy, với Student-Mat:

- `0.901460` là kết quả của mô hình endpoint chính thức cho nhiệm vụ dự đoán cuối.
- `0.846139` là kết quả tại S2 của thí nghiệm stage-aware.

Hai kết quả thuộc hai evaluation authority/checkpoint khác nhau và trả lời hai câu hỏi khác nhau. `0.846139` không có nghĩa là mô hình chính thức bị giảm từ `0.901460`; nó cho biết một mô hình dùng chung cho toàn bộ S0–S1–S2 đạt hiệu quả thế nào khi đến stage S2.

### 5.2. OULAD

| Stage | Phần tiến trình quan sát | Macro-F1 Hybrid |
|---|---:|---:|
| E1 | 20% | 0.707682 |
| E2 | 35% | 0.748207 |
| M1 | 50% | 0.795071 |
| L1 | 75% | 0.852491 |
| FINAL | Cuối kỳ hợp lệ | 0.894071 |

Kết quả tăng dần cho thấy dữ liệu hành vi theo thời gian cung cấp thêm tín hiệu khi quá trình học diễn ra. Ở E1, Hybrid ưu tiên phát hiện sinh viên nguy cơ sớm hơn một số mô hình ML, nhưng đổi lại có thể tạo thêm false positive. Từ M1 đến L1, Hybrid đạt hiệu năng rõ ràng hơn khi chuỗi hành vi đã đủ dài.

---

## 6. So sánh Hybrid với Machine Learning

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

### Diễn giải

- Student-Mat: Decision Tree cao nhất; Hybrid xếp thứ 2 và gần như ngang Random Forest.
- Student-Por: Random Forest cao nhất; Hybrid xếp thứ 3 sau Random Forest và XGBoost.
- OULAD: MLP cao nhất; Hybrid xếp thứ 2 với chênh lệch rất nhỏ.
- Hybrid là mô hình duy nhất nằm trong top 3 trên cả ba dataset.

Hybrid không luôn thắng ML vì:

- UCI có số mẫu nhỏ và chuỗi thời gian chỉ gồm G1, G2 nên lợi thế của CNN–BiLSTM bị giới hạn.
- Các mô hình cây rất mạnh với dữ liệu tabular và các ngưỡng điểm rõ ràng.
- OULAD có dữ liệu hành vi theo tuần, phù hợp hơn với kiến trúc temporal Hybrid.

Ý nghĩa chính của Hybrid không chỉ nằm ở việc đạt điểm cao nhất, mà còn ở khả năng kết hợp thống nhất:

- Mẫu cục bộ từ CNN.
- Quan hệ theo thời gian từ BiLSTM.
- Đặc trưng tabular/context.
- Cảnh báo theo nhiều thời điểm.

---

## 7. Xử lý mất cân bằng lớp

Authority/config cuối không khai báo:

- SMOTE.
- Random oversampling.
- Random undersampling.
- Weighted sampler.

Do đó, repository không tuyên bố đã tạo thêm mẫu tổng hợp để cân bằng dữ liệu.

Mất cân bằng được kiểm soát chủ yếu ở các mức sau.

### Mức đánh giá

- **Macro-F1**: mỗi lớp đóng góp ngang nhau.
- **Balanced Accuracy**: trung bình recall của các lớp.
- **PR-AUC**: quan trọng khi quan tâm lớp thiểu số hoặc lớp nguy cơ.
- Precision, Recall và F1 từng lớp.
- Confusion matrix.

### Mức lựa chọn mô hình

Checkpoint và candidate được chọn bằng inner validation dựa trên metric cân bằng, không chỉ dựa trên Accuracy.

### Mức ra quyết định của OULAD

Threshold được chọn trên pooled inner-OOF để tối đa Macro-F1. Đây là xử lý mất cân bằng ở mức quyết định, không phải tạo thêm mẫu.

Cách mô tả chính xác:

> Nhóm không sử dụng SMOTE trong authority chính thức. Ảnh hưởng của mất cân bằng lớp được kiểm soát bằng Macro-F1, Balanced Accuracy, PR-AUC, metric theo lớp và threshold được chọn trên inner validation đối với OULAD.

---

## 8. Kiểm soát data leakage và overfitting

Pipeline áp dụng các nguyên tắc:

- G3 chỉ dùng tạo nhãn, không dùng làm predictor.
- Preprocessing chỉ fit trên training partition.
- Inner validation dùng để chọn model/checkpoint/threshold.
- Outer test chỉ dùng để báo cáo kết quả.
- Không chọn seed tốt nhất dựa trên outer test.
- Stage sớm không được nhìn feature của stage sau.
- OULAD strict real-time loại score values và score-derived aggregates.
- Nhiều seed và mean-probability ensemble giúp giảm biến động ngẫu nhiên.

Kết quả hiện tại là nested cross-validation/OOF benchmark; repository chưa tuyên bố có một external holdout độc lập hoàn toàn ngoài protocol này.

---

## 9. Chỉ số đánh giá

Chỉ số chính của bài toán phân loại là **Macro-F1** vì các lớp không hoàn toàn cân bằng.

Ngoài ra còn sử dụng:

- Accuracy.
- Balanced Accuracy.
- Macro Precision và Macro Recall.
- Weighted-F1.
- PR-AUC và ROC-AUC.
- NLL và Brier score.
- ECE để đánh giá calibration.
- Confusion matrix.
- Risk Precision, Risk Recall, Risk F1 và Specificity cho OULAD.

RMSE và R² chỉ là chỉ số phụ cho nhánh phân tích điểm số UCI; không dùng để xếp hạng chung giữa UCI và OULAD vì OULAD là bài toán phân loại rủi ro.

---

## 10. Hạn chế cần lưu ý

- Student-Mat và Student-Por có kích thước nhỏ đối với deep learning.
- Chuỗi UCI chỉ có hai thời điểm G1 và G2 nên chưa khai thác hết khả năng của BiLSTM.
- Mô hình Hybrid không đạt hạng nhất trên mọi dataset.
- Xác suất Student-Mat có ECE cao hơn các mô hình cây; cần thận trọng nếu sử dụng trực tiếp xác suất để ra quyết định can thiệp.
- OULAD phù hợp hơn để chứng minh giá trị của nhánh temporal vì có dữ liệu hành vi theo tuần.
- Kết quả hiện tại là bằng chứng dự đoán, không phải bằng chứng rằng khuyến nghị can thiệp tạo ra hiệu quả nhân quả.

---

## 11. Luồng hệ thống hoàn chỉnh

```text
DATA
  ↓
VALIDATION + LABEL CREATION
  ↓
TRAIN-ONLY PREPROCESSING
  ↓
TEMPORAL / TABULAR FEATURE CONSTRUCTION
  ↓
HYBRID CNN–BiLSTM
  ↓
INNER MODEL SELECTION
  ↓
MULTI-SEED ENSEMBLE / THRESHOLD LOCK
  ↓
OUT-OF-FOLD PREDICTION
  ↓
RISK PROFILE
  ↓
RECOMMENDATION POLICY
  ↓
POSTGRESQL / EVIDENCE / REPORT
```

Mô-đun khuyến nghị sử dụng các dự đoán đã đóng băng để tạo risk profile và kế hoạch hỗ trợ. Khuyến nghị là tầng sau mô hình dự đoán, không làm thay đổi authority của các metric dự đoán.

---

## 12. Chạy validation

Cài đặt môi trường từ `requirements.txt` hoặc `requirements-lock.txt`, sau đó chạy:

```powershell
python project.py final status
python project.py final report
python project.py final validate
python project.py pipeline uci validate
python project.py pipeline oulad validate
pytest
```

Các lệnh trên đọc và replay evidence đã đóng băng, không tự động train lại mô hình.

Lệnh train chỉ được gọi khi cần tái lập nghiên cứu:

```powershell
python project.py pipeline uci train
python project.py pipeline oulad train
```

---

## 13. Vị trí mã nguồn và artefact

| Nội dung | Vị trí |
|---|---|
| Authority mô hình cuối | `configs/final/final_model_authority.yaml` |
| Cấu hình UCI MAT | `configs/final/cnn_bilstm_mat.yaml` |
| Cấu hình UCI POR | `configs/final/cnn_bilstm_por.yaml` |
| Cấu hình OULAD H1 | `configs/final/h1_tabular_residual_oulad.yaml` |
| Báo cáo kết quả chính | `reports/final/thesis_v3/01_FINAL_MAIN_RESULTS.md` |
| So sánh ML và Hybrid | `reports/final/thesis_v3/02_FULL_ML_VS_HYBRID.md` |
| Kết quả stage UCI | `reports/final/thesis_v3/03_UCI_STAGE_RESULTS.md` |
| Kết quả stage OULAD | `reports/final/thesis_v3/04_OULAD_STAGE_RESULTS.md` |
| Kiến trúc mô hình | `reports/final/thesis_v3/10_FINAL_MODEL_ARCHITECTURES.md` |
| Prediction và metric cuối | `artifacts/final/`, `artifacts/canonical_v3/` |
| Recommendation runtime | `src/recommend_hybrid/` |
| Database cuối | `database/final/` |
| Bản đồ repository | `docs/project_map/PROJECT_CODE_MAP.md` |

---

## 14. Kết luận khoa học

Kết quả cho thấy Hybrid CNN–BiLSTM là một mô hình ổn định và cạnh tranh trên nhiều loại dữ liệu giáo dục:

- Trên UCI, Hybrid đạt kết quả gần nhóm mô hình ML tốt nhất dù dữ liệu nhỏ và chuỗi rất ngắn.
- Trên OULAD, kiến trúc temporal kết hợp tabular residual expert thể hiện rõ hơn lợi thế trong cảnh báo theo tiến trình.
- Hybrid vượt các biến thể CNN-only và BiLSTM-only trong thí nghiệm UCI, cho thấy việc kết hợp hai thành phần có giá trị hơn việc sử dụng từng thành phần riêng lẻ.
- Mô hình không được tuyên bố là luôn tốt nhất, nhưng cung cấp một kiến trúc thống nhất cho dự đoán cuối, cảnh báo sớm và tích hợp với hệ thống khuyến nghị.

> **Đóng góp chính của khóa luận là xây dựng và đánh giá một hệ thống Hybrid CNN–BiLSTM có protocol chống rò rỉ dữ liệu, hỗ trợ dự đoán theo nhiều thời điểm và duy trì hiệu năng cạnh tranh với các mô hình Machine Learning mạnh.**
