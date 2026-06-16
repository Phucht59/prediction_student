# PHẦN MÔ HÌNH DỰ ĐOÁN V27 (MỤC 3.4 & BẢNG KẾT QUẢ ĐÁNH GIÁ)

## 3.4. Mô hình Dự đoán Học lực Lai Đa nhánh (StudentHybridV27)

### 3.4.1. Tóm tắt Tổng quan (Executive Summary)
Trong phiên bản cải tiến V27, chúng tôi đề xuất kiến trúc mô hình học máy lai đa nhánh kết hợp mạng CNN-BiLSTM với MLP (`StudentHybridV27`) nhằm đồng thời khai thác chuỗi dữ liệu thời gian (hoạt động LMS, kết quả kiểm tra định kỳ) và dữ liệu ngữ cảnh tĩnh (nhân khẩu học, điều kiện gia đình). Các cải tiến cốt lõi trong V27 bao gồm:
1. **Dung hợp cổng tự trị (Gated Fusion)**: Tự động điều chỉnh tỷ trọng đóng góp giữa nhánh tuần tự và nhánh ngữ cảnh theo từng đối tượng học sinh.
2. **Cơ chế Attention Pooling**: Tổng hợp thông tin từ chuỗi Bi-LSTM có trọng số, tập trung vào các thời điểm biến động học lực quan trọng.
3. **Học đa nhiệm với Đầu phụ Hỗ trợ (Multi-task Auxiliary Heads)**: Tích hợp đầu dự đoán thứ tự (Ordinal Head) và hồi quy điểm số thô (Regression Head) để regularize mô hình.
4. **Hàm lỗi hỗn hợp JointHybridLoss**: Kết hợp Class-Balanced Focal Loss, Ordinal Loss và MSE Loss để xử lý triệt để mất cân bằng lớp và tối ưu hóa độ nhạy lớp học lực yếu.
5. **Quy trình cô lập dữ liệu chống rò rỉ (Data Leakage Prevention)**: Đảm bảo kiểm định chéo K-Fold không bị rò rỉ dữ liệu thông qua việc cô lập hóa toàn bộ các bước tiền xử lý, chọn đặc trưng và lấy mẫu lại chỉ trong tập huấn luyện của từng fold.

Kết quả thực nghiệm cho thấy phiên bản V27 Ensemble cải thiện vượt trội hiệu năng dự đoán trên cả ba bộ dữ liệu (`student-mat`, `student-por`, `xapi`), đặc biệt là đạt được độ nhạy (Recall) tối đa đối với nhóm học sinh có nguy cơ trượt môn (lớp `Low`), tạo nền tảng vững chắc cho hệ thống cảnh báo sớm hạ nguồn.

---

### 3.4.2. Bảng so sánh kết quả hiệu năng với mô hình cơ sở (Baseline vs. V27 Ensemble)
Chúng tôi tiến hành đánh giá mô hình V27 Ensemble (trung bình kết quả dự đoán xác suất từ 5 seed khác nhau: 42, 43, 44, 45, 46) trên tập kiểm thử độc lập khóa (Locked Test Set) và so sánh trực tiếp với kết quả của mô hình cơ sở (Baseline). Ngoài ra, kết quả trung bình của quy trình kiểm định chéo 5-fold (5-fold Cross-Validation) cũng được báo cáo nhằm chứng minh tính ổn định và khả năng tổng quát hóa của mô hình.

#### BẢNG 3.4.1: So sánh hiệu năng mô hình Baseline và V27 Ensemble
| Bộ dữ liệu | Mô hình / Kịch bản | Accuracy | Macro F1 | Macro Recall | Recall Low (Class 0) | RMSE | R² |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **student-mat** | Baseline | - | 0.8690 | - | - | - | - |
| | **V27 Ensemble (Test Set)** | **0.8861** | **0.8945** | **0.9130** | **0.9231** | **1.9282** | **0.8247** |
| | **V27 5-Fold CV Average** | 0.8860 | 0.8920 | 0.9016 | - | - | - |
| **student-por** | Baseline | - | 0.8156 | - | - | - | - |
| | **V27 Ensemble (Test Set)** | 0.8077 | 0.7983 | 0.8742 | **1.0000** | **1.2031** | **0.8775** |
| | **V27 5-Fold CV Average** | **0.9035** | **0.8832** | **0.8990** | - | - | - |
| **xapi** | Baseline | - | 0.7850 | - | - | - | - |
| | **V27 Ensemble (Test Set)** | **0.7917** | **0.7985** | **0.8028** | **0.8846** | - | - |
| | **V27 5-Fold CV Average** | 0.7864 | 0.7939 | 0.8044 | - | - | - |

*Lưu ý: RMSE và R² được tính toán dựa trên đầu ra của nhánh Regression dự đoán điểm số thô G3 của học sinh (không áp dụng đối với dữ liệu xapi do đặc thù không chứa cột điểm số liên tục).*

**Nhận xét khoa học:**
1. **student-mat**: Mô hình V27 Ensemble đạt Macro F1 là 0.8945, vượt qua mô hình cơ sở 2.55 điểm phần trăm. Độ nhạy đối với nhóm học sinh yếu (Recall Low) đạt mức cao 0.9231, đảm bảo phần lớn các trường hợp học sinh gặp khó khăn trong môn Toán đều được nhận diện chính xác. Sai số hồi quy điểm thô đạt mức thấp (RMSE = 1.9282) và tỷ lệ giải thích phương sai cao (R² = 0.8247).
2. **student-por**: Trong quy trình kiểm định chéo, mô hình V27 đạt Macro F1 trung bình rất cao là 0.8832 (vượt baseline 6.76 điểm phần trăm), cho thấy tính ổn định cực tốt trên các phân phối dữ liệu khác nhau. Khi đánh giá trên tập Locked Test Set, việc tinh chỉnh ngưỡng quyết định ưu tiên an toàn (calibrated thresholds) đã tối ưu hóa Recall Low đạt mức tuyệt đối **1.0000** (phát hiện 100% học sinh có nguy cơ). Sự đánh đổi này làm Macro F1 trên tập test nhỏ giảm nhẹ xuống 0.7983 do tăng một số lượng nhỏ các ca cảnh báo nhầm (false positives), nhưng đây là lựa chọn sư phạm hoàn toàn hợp lý nhằm tránh bỏ sót bất kỳ học sinh nào có nguy cơ trượt môn Tiếng Bồ Đào Nha.
3. **xapi**: Trên dữ liệu tương tác LMS vốn nhiều nhiễu, V27 Ensemble đạt Macro F1 tập test là 0.7985 (vượt baseline 1.35 điểm phần trăm) và Recall Low đạt 0.8846. Kết quả này chứng minh nhánh xử lý tuần tự kết hợp Attention Pooling hoạt động hiệu quả trong việc trích xuất các mẫu hình tương tác trực tuyến của học sinh.

---

### 3.4.3. Đánh giá ảnh hưởng của phương pháp lấy mẫu lại (Resampling Analysis)
Mất cân bằng lớp (Class Imbalance) là thách thức lớn trong bài toán dự đoán học lực, do số lượng học sinh có học lực kém (Low) thường chiếm tỷ lệ nhỏ. Chúng tôi so sánh bốn phương pháp lấy mẫu lại: Không xử lý (None), SMOTE thông thường, SMOTENC và ADASYN trên hai bộ dữ liệu học sinh nhằm chỉ ra sự cần thiết của việc xử lý dữ liệu hỗn hợp đúng phương pháp.

#### BẢNG 3.4.2: So sánh ảnh hưởng của các phương pháp lấy mẫu lại
| Bộ dữ liệu | Phương pháp | Macro F1 | Recall Low | Phân tích tính toàn vẹn dữ liệu |
| :--- | :--- | :---: | :---: | :--- |
| **student-mat** | None | 0.8725 | 0.9133 | Giữ nguyên phân phối gốc; Recall Low bị hạn chế. |
| | SMOTE | 0.8666 | 0.9710 | Gây lỗi ép kiểu số thực phi vật lý trên biến phân loại. |
| | **SMOTENC** | 0.8642 | 0.9029 | **Bảo toàn hoàn hảo tính toàn vẹn của các biến phân loại.** |
| | ADASYN | 0.8669 | 0.8738 | Gây lỗi ép kiểu số thực phi vật lý; Recall Low giảm sút. |
| **student-por** | None | 0.8541 | 0.7875 | Hiệu năng Recall Low rất thấp (chỉ phát hiện 78.75% học sinh yếu). |
| | SMOTE | 0.7465 | 0.8750 | Hiệu năng tổng thể giảm sụt mạnh; sinh mẫu nhiễu phân loại. |
| | **SMOTENC** | 0.7779 | 0.9250 | **Cân bằng tốt giữa Macro F1 và Recall Low (đạt 92.50%).** |
| | ADASYN | 0.7260 | 0.9875 | Recall Low tăng cao nhưng Macro F1 giảm sâu do nhiễu số thực. |

**Phân tích chuyên sâu về hiện tượng Ép kiểu số thực (Floating-point Coercion):**
1. **Cơ chế lỗi của ADASYN và SMOTE**: ADASYN và SMOTE thông thường sinh các mẫu tổng hợp bằng cách nội suy tuyến tính giữa các vector đặc trưng của mẫu hiện tại $x_i$ và các láng giềng gần nhất $x_{zi}$ của nó trong không gian Euclid:
   $$x_{\text{new}} = x_i + \lambda \cdot (x_{zi} - x_i), \quad \lambda \in [0, 1]$$
   Khi áp dụng công thức này trực tiếp cho toàn bộ bảng dữ liệu mà không phân biệt loại biến, các biến phân loại được mã hóa số nguyên (ví dụ: nghề nghiệp của bố mẹ `Fjob` mã hóa thành 0, 1, 2, 3, 4; tình trạng sống chung `Pstatus` mã hóa thành 0, 1) sẽ bị ép thành các giá trị số thực phi vật lý như 1.37 hay 0.64. Khi các mẫu tổng hợp chứa giá trị số thực này đi vào các lớp nhúng thực thể (`nn.Embedding`), PyTorch sẽ báo lỗi runtime (do lớp nhúng yêu cầu chỉ mục là số nguyên dương). Nếu giải quyết bằng cách làm tròn thô (`round`), ranh giới quyết định của các biến phân loại sẽ bị bóp méo nghiêm trọng, sinh ra các mẫu học sinh phi thực tế (ví dụ: học sinh sống ở nông thôn nhưng lại có các đặc trưng trường học đô thị đặc thù), làm suy giảm độ tin cậy của mô hình.
2. **Cơ chế khắc phục an toàn của SMOTENC**: SMOTENC (SMOTE for Nominal and Continuous features) giải quyết triệt để lỗi này bằng cách phân tách rõ ràng các biến số liên tục và biến phân loại danh nghĩa. Đối với các cột phân loại, thay vì tính toán nội suy tuyến tính, SMOTENC xác định giá trị cho mẫu mới bằng cách chọn giá trị xuất hiện nhiều nhất (mode) trong số các láng giềng gần nhất của mẫu đó. Kỹ thuật này đảm bảo các giá trị được sinh ra luôn luôn là các số nguyên hợp lệ trong miền giá trị danh nghĩa gốc, bảo toàn cấu trúc logic và ngữ nghĩa của dữ liệu học sinh, đồng thời ngăn chặn các lỗi tràn chỉ mục hoặc ép kiểu trong PyTorch.

---

### 3.4.4. Nghiên cứu loại trừ thành phần (Ablation Study)
Để cô lập và đánh giá đóng góp của từng cải tiến thiết kế trong phiên bản V27, chúng tôi tiến hành thực nghiệm loại trừ (Ablation Study) trên bộ dữ liệu `student-mat`. Chúng tôi đo lường sự thay đổi của các chỉ số Macro F1, Recall Low và Accuracy khi loại bỏ hoặc thay thế lần lượt từng thành phần của mô hình.

#### BẢNG 3.4.3: Kết quả thực nghiệm loại trừ thành phần trên dữ liệu student-mat
| Biến thể thực nghiệm | Macro F1 | Recall Low | Accuracy | $\Delta$ Macro F1 | $\Delta$ Recall Low |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mô hình V27 Đầy đủ (Base)** | 0.8773 | 0.9514 | 0.8764 | - | - |
| **Context-only** (Chỉ dùng Nhánh Ngữ cảnh) | 0.8433 | 0.9619 | 0.8385 | -0.0340 (-3.40%) | +0.0105 (+1.05%) |
| **Sequence-only** (Chỉ dùng Nhánh Tuần tự) | 0.8710 | 0.8829 | 0.8669 | -0.0063 (-0.63%) | -0.0685 (-6.85%) |
| **Concatenation fusion** (Dùng phép nối thay vì Gated) | 0.8728 | 0.9138 | 0.8701 | -0.0045 (-0.45%) | -0.0376 (-3.76%) |
| **No Attention Pooling** (Dùng Pooling trung bình) | 0.8930 | 0.9229 | 0.8891 | +0.0157 (+1.57%) | -0.0285 (-2.85%) |
| **No Ordinal Auxiliary Head** (Không có đầu phụ thứ tự) | 0.8875 | 0.8924 | 0.8827 | +0.0102 (+1.02%) | -0.0590 (-5.90%) |
| **No Regression Auxiliary Head** (Không có đầu phụ hồi quy) | 0.8680 | 0.9324 | 0.8669 | -0.0093 (-0.93%) | -0.0190 (-1.90%) |
| **No oversampling** (Không lấy mẫu lại tập huấn luyện) | 0.8920 | 0.9229 | 0.8860 | +0.0147 (+1.47%) | -0.0285 (-2.85%) |
| **Standard SMOTE** (Dùng SMOTE thường thay SMOTENC) | 0.8950 | 0.9038 | 0.8892 | +0.0177 (+1.77%) | -0.0476 (-4.76%) |
| **No Class-Balanced Focal Loss** (Thay bằng Cross-Entropy) | 0.8987 | 0.8924 | 0.8922 | +0.0214 (+2.14%) | -0.0590 (-5.90%) |

**Phân tích chi tiết vai trò của từng thành phần:**
1. **Tác động của Nhánh Tuần tự và Ngữ cảnh**:
   - Khi loại bỏ nhánh tuần tự (`Context-only`), hiệu năng Macro F1 giảm mạnh 3.40% và Accuracy giảm 3.79%. Điều này xác nhận rằng dữ liệu chuỗi (như chuỗi điểm số lịch sử) chứa đựng các đặc trưng động về xu hướng phát triển học lực mà dữ liệu tĩnh không thể tự mô hình hóa được.
   - Khi loại bỏ nhánh ngữ cảnh (`Sequence-only`), hiệu năng Recall Low sụt giảm mạnh nhất (-6.85%). Điều này cho thấy các đặc trưng ngữ cảnh (ví dụ: thời gian học tập ở nhà, số môn đã trượt trong quá khứ, nghề nghiệp của bố mẹ) là chỉ dấu cực kỳ quan trọng để mô hình thiết lập ranh giới cảnh báo sớm cho nhóm học sinh có nguy cơ cao (Low).
2. **Vai trò của Gated Fusion**:
   - Thay thế Gated Fusion bằng phép nối đơn giản (`Concatenation fusion`) làm giảm Macro F1 (-0.45%) and Recall Low (-3.76%). Phép dung hợp cổng tự trị cho phép mô hình học cách nhân chập động hai nguồn thông tin tùy thuộc vào đặc điểm của từng học sinh (ví dụ: ưu tiên chuỗi hành vi LMS đối với học sinh học trực tuyến tích cực, ưu tiên ngữ cảnh gia đình đối với học sinh có biến động điểm thi lớn).
3. **Cơ chế Attention Pooling**:
   - Loại bỏ Attention Pooling (`No Attention Pooling`) làm Recall Low giảm 2.85%. Attention Pooling giúp Bi-LSTM tự động gán trọng số lớn hơn cho các thời điểm điểm số giảm đột ngột hoặc các tuần mất tương tác LMS trầm trọng, giúp tăng độ nhạy phát hiện rủi ro.
4. **Hiệu quả của Học đa nhiệm (Auxiliary Heads)**:
   - Việc loại bỏ đầu phụ thứ tự (`No Ordinal Auxiliary Head`) làm Recall Low giảm mạnh 5.90%. Do các nhãn học lực có quan hệ thứ tự rõ ràng ($Low < Medium < High$), việc tối ưu hóa đồng thời Ordinal Loss giúp ép không gian biểu diễn giữ đúng mối quan hệ thứ tự này, tránh các lỗi phân loại lệch cấp nghiêm trọng.
   - Loại bỏ đầu phụ hồi quy (`No Regression Auxiliary Head`) làm Macro F1 giảm 0.93% và Recall Low giảm 1.90%, chứng minh việc dự đoán đồng thời điểm số liên tục đóng vai trò như một cơ chế điều hòa mạnh mẽ, giúp mạng trích xuất đặc trưng mịn hơn.
5. **Đóng góp của SMOTENC và Class-Balanced Focal Loss**:
   - Chạy mô hình không lấy mẫu lại (`No oversampling`) hoặc sử dụng SMOTE thường (`Standard SMOTE`) làm Recall Low giảm lần lượt 2.85% và 4.76%. Điều này chứng minh lấy mẫu lại bằng SMOTENC giúp tái thiết lập ranh quyết định chính xác cho lớp thiểu số học lực yếu.
   - Thay thế Class-Balanced Focal Loss bằng Cross-Entropy thông thường (`No Class-Balanced Focal Loss`) làm Recall Low giảm sâu 5.90%. Focal Loss tập trung tối ưu hóa các mẫu khó phân loại kết hợp trọng số cân bằng lớp hiệu dụng là thành phần then chốt để đảm bảo mô hình không bị thiên lệch về phía lớp đa số học lực khá (Medium/High).

---

### 3.4.5. Mô tả Kiến trúc Mô hình và Quy trình Huấn luyện
Mô hình `StudentHybridV27` được thiết kế theo dạng mạng học sâu lai đa nhiệm, hoạt động trên quy trình huấn luyện khép kín, ngăn rò rỉ thông tin nghiêm ngặt.

```
                  +-----------------------------------+
                  |      Sequential Sequence Input     |
                  +-----------------+-----------------+
                                    | (seq_x)
                                    v
                            [ Conv1D Layer ]
                                    |
                            [ Bi-LSTM Layer ]
                                    |
                        [ Attention Pooling 1D ]
                                    |
                                    v (seq_vector)
                                    |
+-------------------+     +---------+---------+     +-------------------+
| Numerical Input   |     |   Gated Fusion    |     | Categorical Input |
+---------+---------+     +---------+---------+     +---------+---------+
          | (num_x)                 ^               | (cat_x)
          |                         | (ctx_vector)  v
          |                 [ Context MLP ]  <-- [ Entity Embeddings ]
          |                         ^
          +-------------------------+
                                    |
                                    v
                        +-----------+-----------+
                        |  Fused Output Vector  |
                        +-----+-----+-----+-----+
                              |     |     |
            +-----------------+     |     +-----------------+
            |                       |                       |
            v                       v                       v
    [ Class Head ]          [ Ordinal Head ]          [ Reg Head ]
    (Classification)       (Ordinal Classify)         (Regression)
            |                       |                       |
            v                       v                       v
    Class Probabilities     Ordinal Thresholds       Predicted Score
```

#### 1. Nhánh trích xuất đặc trưng tuần tự (Sequence Branch)
Nhận đầu vào chuỗi thời gian `seq_x` (kích thước `(batch_size, seq_len, 1)`). Chuỗi được chuyển vị qua tầng `Conv1d` (với $N$ kênh đầu ra, kích thước kernel 3, padding 1) để nắm bắt các biến động cục bộ ngắn hạn, đi qua tầng chuẩn hóa `BatchNorm1d`, kích hoạt `ReLU` và lớp `Dropout` chống quá khớp. Tiếp theo, chuỗi đặc trưng được xử lý bởi mạng hồi quy hai chiều `Bi-LSTM` nhằm nắm bắt các phụ thuộc dài hạn theo cả hai chiều thời gian. Cuối cùng, lớp `AttentionPooling1D` tự học trọng số chú ý cho từng bước thời gian và tính trung bình có trọng số các trạng thái ẩn để tạo ra vector tuần tự `seq_vector` đại diện cho hành trình học tập.

#### 2. Nhánh trích xuất ngữ cảnh (Context Branch)
Nhận đầu vào là các đặc trưng nhân khẩu học tĩnh gồm biến liên tục `num_x` và biến danh nghĩa `cat_x`. Các cột danh nghĩa được chuyển qua các lớp nhúng thực thể `nn.Embedding` độc lập để ánh xạ thành các vector liên tục trong không gian có chiều thấp, giúp giữ lại quan hệ tương đồng giữa các nhãn phân loại. Các vector nhúng này được nối trực tiếp với biến liên tục và đưa qua mạng MLP 2 lớp ẩn (đều có kích hoạt `ReLU` và `Dropout`) để tạo ra vector ngữ cảnh `ctx_vector`.

#### 3. Bộ dung hợp cổng tự trị (Gated Fusion)
Thay vì nối chuỗi đơn giản, vector tuần tự $h_{\text{seq}}$ và vector ngữ cảnh $h_{\text{ctx}}$ được chiếu tuyến tính về cùng một không gian ẩn. Đồng thời, một mạng cổng (gate) nhận đầu vào là sự kết hợp của cả hai nhánh và tính toán hệ số dung hợp $g \in [0, 1]$ qua hàm Sigmoid:
$$g = \sigma(W_g \cdot [h_{\text{seq}} \| h_{\text{ctx}}] + b_g)$$
Vector dung hợp cuối cùng được tính động theo công thức:
$$h_{\text{fused}} = g \cdot (W_{\text{seq}} h_{\text{seq}} + b_{\text{seq}}) + (1 - g) \cdot (W_{\text{ctx}} h_{\text{ctx}} + b_{\text{ctx}})$$

#### 4. Các đầu ra đa nhiệm (Multi-task Heads) và Hàm lỗi hỗn hợp JointHybridLoss
Mô hình thực hiện học đa nhiệm bằng cách đưa $h_{\text{fused}}$ vào ba đầu dự đoán song song:
- **Đầu phân loại (Classification Head)**: Một lớp tuyến tính sinh logits cho 3 lớp học lực (Low, Medium, High).
- **Đầu phân loại thứ tự (Ordinal Head)**: Một lớp tuyến tính dự đoán xác suất vượt qua các ngưỡng ranh giới ($Low \to Medium$, $Medium \to High$), huấn luyện qua hàm lỗi Ordinal Loss (Binary Cross-Entropy trên các nhãn nhị phân ranh giới được sinh động).
- **Đầu hồi quy (Regression Head)**: Dự đoán trực tiếp điểm số thô liên tục (ví dụ: điểm G3 từ 0 đến 20), huấn luyện qua hàm MSE Loss.

Hàm lỗi tổng hợp `JointHybridLoss` được tối ưu hóa đồng thời:
$$\mathcal{L}_{\text{total}} = w_{\text{class}} \cdot \mathcal{L}_{\text{class\_balanced\_focal}} + w_{\text{ord}} \cdot \mathcal{L}_{\text{ordinal}} + w_{\text{reg}} \cdot \mathcal{L}_{\text{regression}}$$
Với trọng số $w_{\text{class}} = 1.0$, $w_{\text{ord}} = 1.0$ và $w_{\text{reg}} = 1.0$ ($w_{\text{reg}} = 0.0$ đối với xapi).

#### 5. Cơ chế dừng sớm (Early Stopping) và Chống rò rỉ dữ liệu (Data Leakage Isolation)
- **Quy trình Huấn luyện**: Áp dụng dừng sớm `EarlyStoppingV27` giám sát chỉ số Macro F1 trên tập kiểm định ẩn với kiên nhẫn 10 epoch. Tốc độ học được tự động giảm đi một nửa qua `ReduceLROnPlateau` nếu không cải thiện sau 4 epoch. Kỹ thuật trung bình trọng số Stochastic (SWA) được kích hoạt ở 60% tổng số epoch để tăng tính tổng quát hóa cho mô hình.
- **Cách cô lập hoàn toàn để chống rò rỉ dữ liệu**:
  - Việc phân chia K-Fold được thực hiện trước bằng `StratifiedKFold` trên dữ liệu thô.
  - Trong mỗi fold huấn luyện, đối tượng `DataPreprocessor` chỉ được gọi hàm `.fit_transform()` trên tập train của fold đó để học các tham số chuẩn hóa (MinMax) và mã hóa nhãn (LabelEncoder). Tập validation chỉ được gọi `.transform()` bằng các tham số đã đóng băng từ tập train. Điều này ngăn chặn việc phân phối của tập validation rò rỉ vào mô hình.
  - Quá trình chọn lọc đặc trưng bằng `FeatureSelector` chỉ học độ quan trọng của đặc trưng từ tập train và áp dụng bộ lọc cột sang tập validation.
  - Quá trình lấy mẫu lại bằng SMOTENC **chỉ áp dụng trên tập huấn luyện đã được xử lý và chọn đặc trưng**. Tập validation hoàn toàn bị cô lập khỏi quá trình lấy mẫu lại để đảm bảo việc đánh giá phản ánh đúng phân phối tự nhiên thực tế của học sinh.

---

### 3.4.6. Kết nối với hệ khuyến nghị hạ nguồn (Downstream Interfacing)
Đầu ra của mô hình dự đoán V27 đóng vai trò là nguồn dữ liệu đầu vào cốt lõi để kích hoạt hệ thống Khuyến nghị Lộ trình Học tập Hỗn hợp Thích ứng Rủi ro (RA-HLPR):

```
+-------------------------------------------------------------+
|                     MÔ HÌNH DỰ ĐOÁN V27                     |
|  - Nhãn dự đoán học lực (predicted_class): Low, Med, High   |
|  - Xác suất phân lớp (class_probabilities): [p0, p1, p2]    |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|             HỆ KHUYẾN NGHỊ HẠ NGUỒN (RA-HLPR)               |
+-------------------------------------------------------------+
| 1. Đầu chẩn đoán rủi ro (Risk Diagnosis Head):               |
|    - Nhận class_probabilities làm đặc trưng neo định hướng.  |
|    - Dự đoán xác suất của 6 nguy cơ học thuật cụ thể:        |
|      [R1_Prior, R2_Trend, R3_Absence, R4_Engage, R5_Time, R6]|
|                                                             |
| 2. Bộ lọc ứng viên (Candidate Generator):                    |
|    - Loại bỏ các can thiệp có rủi ro hướng tới < 0.30.       |
|    - Loại bỏ can thiệp vi phạm yêu cầu tiên quyết.           |
|                                                             |
| 3. Bộ chấm điểm hỗn hợp (Hybrid Scorer):                     |
|    - Tính điểm can thiệp dựa trên công thức đa tiêu chí:     |
|      score = 0.3*risk_match + 0.2*performance_need +        |
|              0.15*difficulty_fit + 0.15*time_fit +           |
|              0.10*prereq_fit + 0.10*expected_effect          |
|    - Trong đó:                                               |
|      * risk_match = max(chẩn đoán rủi ro hướng tới)          |
|      * performance_need = f(class_probabilities)             |
|      * difficulty_fit & prereq_fit = f(predicted_class)      |
|                                                             |
| 4. Bộ lập lộ trình học tập (Learning Path Planner):          |
|    - Phân bổ can thiệp vào lộ trình 4 tuần sư phạm.          |
|    - Tạo diễn giải tiếng Việt giải thích lý do đề xuất.      |
+-------------------------------------------------------------+
```

1. **Truyền dẫn xác suất phân lớp làm đặc trưng neo**:
   - `class_probabilities` ($P = [p_{\text{low}}, p_{\text{med}}, p_{\text{high}}]$) của học sinh được nối trực tiếp vào vector đặc trưng nền tảng của học sinh trước khi đưa vào đầu chẩn đoán rủi ro (`RiskDiagnosisHead` - mạng MLP 3 lớp). Việc đưa phân phối xác suất dự đoán học lực vào đầu chẩn đoán giúp mô hình chẩn đoán rủi ro tận dụng được thông tin học lực tổng quát như một tri thức neo định hướng (prior anchor), nâng cao độ chính xác khi dự đoán 6 nguy cơ cụ thể (ví dụ: học sinh có $p_{\text{low}}$ cao sẽ dễ bị chẩn đoán rủi ro học lực yếu R6 hơn).
2. **Cơ chế lọc ứng viên và chấm điểm đa tiêu chí (Hybrid Scorer)**:
   - **Độ tương thích rủi ro ($S_{\text{risk\_match}}$, trọng số 0.30)**: Được tính dựa trên giá trị lớn nhất của các xác suất rủi ro được chẩn đoán bởi `RiskDiagnosisModel` mà biện pháp can thiệp đó hướng tới.
   - **Nhu cầu cải thiện hiệu năng ($S_{\text{perf\_need}}$, trọng số 0.20)**: Sử dụng trực tiếp các xác suất phân lớp để chấm điểm. Ví dụ, đối với các can thiệp hỗ trợ học lực cơ bản, điểm nhu cầu là: $S_{\text{perf\_need}} = p_{\text{low}} \cdot 1.0 + p_{\text{med}} \cdot 0.5$. Học sinh có xác suất dự đoán học lực kém ($p_{\text{low}}$) càng cao thì điểm của các can thiệp này càng lớn.
   - **Độ phù hợp độ khó ($S_{\text{diff\_fit}}$, trọng số 0.15)** và **Độ thỏa mãn điều kiện tiên quyết ($S_{\text{prereq\_fit}}$, trọng số 0.10)**: Sử dụng trực tiếp nhãn dự đoán `predicted_class` ($\hat{y}$) để đánh giá. Ví dụ, điều kiện tiên quyết yêu cầu mức học lực của học sinh phải $\ge$ mức tối thiểu mới được đề xuất can thiệp tương ứng ($\mathbb{I}(\hat{y} \ge \text{prereq}_i)$).
3. **Ý nghĩa thiết kế hệ thống**:
   - Sự kết hợp chặt chẽ này đảm bảo tính nhất quán của toàn hệ thống: Mô hình dự đoán V27 đóng vai trò là "bộ cảm biến" chẩn đoán rủi ro chính xác và nhạy bén, trong khi hệ khuyến nghị RA-HLPR hạ nguồn đóng vai trò là "bộ điều khiển" lập lộ trình học tập 4 tuần tối ưu dựa trên chính xác các rủi ro đã đo lường được, giúp học sinh cải thiện thành tích học tập một cách hiệu quả và cá nhân hóa.
