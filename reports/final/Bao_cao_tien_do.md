# BÁO CÁO TIẾN ĐỘ THỰC HIỆN KHÓA LUẬN

**Tên đề tài:** Hệ thống dự đoán và khuyến nghị cải thiện thành tích học tập  
**Ngày cập nhật:** 12/06/2026  
**Trạng thái chung:** Đã hoàn thành refactor, tích hợp và chạy thực nghiệm chính thức cho toàn bộ ba bộ dữ liệu.

## 1. Mục tiêu giai đoạn hiện tại

Giai đoạn này tập trung điều chỉnh hệ thống cho đúng đề cương khóa luận đã được phê duyệt:

1. Thay kiến trúc tabular/RecSys bằng mô hình lai CNN - BiLSTM - MLP.
2. Thay khuyến nghị counterfactual bằng lộ trình học tập theo luật.
3. Tích hợp PostgreSQL để lưu dữ liệu đánh giá và khuyến nghị.
4. Giữ cơ chế xử lý mất cân bằng bằng SMOTE/ADASYN.
5. Duy trì quy trình Optuna, locked test và ensemble nhiều seed.

## 2. Các hạng mục đã hoàn thành

### 2.1. Refactor kiến trúc mô hình

Đã viết lại `src/models.py` theo đúng kiến trúc được phê duyệt:

- Nhánh dữ liệu tuần tự:
  - `Conv1D` trích xuất đặc trưng cục bộ.
  - `Bi-LSTM` phân tích chuỗi điểm số hoặc chuỗi tương tác học tập.
  - Attention Pooling tổng hợp biểu diễn của chuỗi.
- Nhánh dữ liệu ngữ cảnh:
  - Biến số và biến phân loại được đưa qua MLP đơn giản.
- Hai nhánh được nối lại tại tầng fusion.
- Tầng cuối sinh logits cho ba lớp `Low`, `Medium`, `High`.
- Hàm `predict_proba()` chuyển logits thành phân phối xác suất Softmax.

Đã loại bỏ hoàn toàn khỏi mã nguồn chạy chính:

- DeepFM.
- DCN-V2.
- FT-Transformer và TabularTokenizer.
- Focal Loss.
- Hybrid Loss kết hợp ordinal loss.

### 2.2. Refactor quy trình huấn luyện

Đã cập nhật `src/train_pipeline.py` và `scripts/run_pipeline.py`:

- Sử dụng `CrossEntropyLoss` có trọng số lớp.
- Tiếp tục sử dụng SMOTE hoặc ADASYN trên tập huấn luyện.
- Optuna chỉ tối ưu các tham số thuộc CNN, Bi-LSTM và MLP.
- Sử dụng Stratified K-Fold Cross Validation trong quá trình Optuna.
- Giữ early stopping và learning-rate scheduler.
- Tách validation nội bộ trong bước huấn luyện ensemble, không sử dụng locked test để chọn epoch.
- Ensemble sử dụng các seed cố định `42`, `123`, `155`, `156`, `2025`.
- Dự đoán cuối sử dụng majority voting và xác suất trung bình của ensemble.

### 2.3. Hoàn thiện xử lý dữ liệu

Đã cập nhật `src/data_pipeline.py`:

- Bắt buộc giữ lại các đặc trưng tuần tự cần cho CNN-BiLSTM:
  - Student-Mat/Student-Por: `G1`, `G2`.
  - xAPI: `raisedhands`, `VisITedResources`, `AnnouncementsView`, `Discussion`.
- Dataset công khai rõ danh sách biến tuần tự, biến số và biến phân loại.
- Bổ sung kiểm tra kích thước dữ liệu đầu vào để hạn chế lỗi lệch shape giữa dataset và model.
- Tiếp tục bảo đảm SMOTE/ADASYN chỉ áp dụng trên dữ liệu huấn luyện.

### 2.4. Xây dựng hệ thống Learning Path

Đã thay `GreedyCounterfactualSearcher` bằng `RuleBasedLearningPathEngine` trong `src/explainability.py`.

Hệ thống hiện phân tích các rủi ro như:

- Số buổi hoặc số ngày vắng học cao.
- Thời gian tự học thấp.
- Có lịch sử trượt môn.
- Điểm G1/G2 thấp hoặc giảm.
- Mức sử dụng học liệu thấp.
- Mức tương tác lớp học thấp.
- Thiếu phối hợp giữa phụ huynh và nhà trường.

Mỗi sinh viên được sinh một lộ trình gồm:

- Mức rủi ro: `high`, `moderate` hoặc `stable`.
- Các yếu tố rủi ro và bằng chứng tương ứng.
- Mục tiêu theo từng giai đoạn.
- Hành động cụ thể theo tuần.
- Mốc theo dõi và điều kiện chuyển sang phụ đạo trực tiếp.

Đầu ra được lưu tại:

```text
reports/final/recommendations/<dataset>_3class_learning_paths.csv
```

### 2.5. Tích hợp PostgreSQL

Đã cập nhật `src/evaluation.py`, `src/config.py` và `database/schema.sql`.

Sau khi đánh giá locked test, pipeline có thể lưu:

- Đặc trưng gốc của sinh viên.
- Nhãn thật và nhãn dự đoán.
- Xác suất của ba lớp.
- Confidence score.
- Các chỉ số đánh giá.
- Mức rủi ro.
- Learning Path được sinh cho từng sinh viên.

Các bảng chính:

- `paper_runs`.
- `paper_predictions`.
- `paper_evaluation_metrics`.
- `paper_learning_recommendations`.

Kết nối PostgreSQL đã được kiểm tra bằng `DATABASE_URL` trong `.env`.

### 2.6. Cập nhật vận hành và tài liệu

- Đã cập nhật `README.md` theo kiến trúc mới.
- Đã bổ sung kiểm tra lỗi sau từng dataset trong `run_all.bat`.
- Pipeline mặc định ghi kết quả vào PostgreSQL.
- Có thể dùng `--skip-postgres` khi kiểm thử ngoại tuyến.
- Đã thay bộ test cũ bằng bộ test bám theo kiến trúc và đề cương mới.

## 3. Kết quả kiểm tra kỹ thuật

### 3.1. Unit test

Kết quả hiện tại:

```text
7 passed
```

Các test kiểm tra:

- Mô hình có Conv1D, Bi-LSTM hai chiều và MLP.
- Đầu ra có đúng ba lớp và tổng xác suất bằng 1.
- Các kiến trúc và loss không phù hợp đã được loại bỏ.
- Weighted Cross-Entropy hoạt động với dữ liệu mất cân bằng.
- Feature selector luôn giữ các biến tuần tự.
- Learning Path có đầy đủ giai đoạn, mục tiêu và hành động.
- Schema PostgreSQL có đủ trường confidence, original features và learning path.

### 3.2. Optuna smoke test

Đã chạy một Optuna trial thật trên Student-Mat với 2-fold CV để kiểm tra khả năng tích hợp của pipeline mới.

Kết quả kiểm tra kỹ thuật:

```text
Best CV F1-Macro: 0.7965
```

Đây chỉ là smoke test với một trial và hai fold, không phải kết quả thực nghiệm chính thức.

### 3.3. Kiểm tra end-to-end

Đã chạy pipeline debug từ đầu đến cuối trên Student-Mat:

- Đọc và xử lý dữ liệu thật.
- Huấn luyện CNN-BiLSTM + MLP.
- Dự đoán toàn bộ locked test.
- Sinh feature importance.
- Sinh Learning Path.
- Lưu artifact dạng file.
- Ghi kết quả vào PostgreSQL.

Kết quả kiểm tra tích hợp:

```text
79 prediction records
79 learning-path records
1 evaluation-metric record
```

Các bản ghi debug đã được xóa khỏi PostgreSQL sau khi kiểm tra, không làm lẫn với dữ liệu thực nghiệm chính thức.

## 4. Trạng thái kết quả thực nghiệm

Đã chạy chính thức `50 Optuna trials` và ensemble 5 seed cho cả ba bộ dữ liệu. Kết quả locked test của kiến trúc CNN-BiLSTM + Context MLP:

| Dataset | Best CV F1 | Accuracy | F1-Macro | Precision-Macro | Recall-Macro | RMSE | R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| Student-Mat | 0.9112 | 0.8861 | 0.8905 | 0.8781 | 0.9170 | 0.3375 | 0.7720 |
| Student-Por | 0.8898 | 0.8615 | 0.8394 | 0.8110 | 0.8816 | 0.3721 | 0.6063 |
| xAPI | 0.8050 | 0.7708 | 0.7773 | 0.7702 | 0.7918 | 0.4787 | 0.5923 |

### 4.1. Nhận xét

- Student-Mat có kết quả tốt nhất với F1-Macro `0.8905` và Recall-Macro `0.9170`.
- Student-Por đạt F1-Macro `0.8394`; chênh lệch giữa CV và locked test ở mức chấp nhận được nhưng vẫn cần trình bày trong phần thảo luận.
- xAPI đạt F1-Macro `0.7773`. Đây là bộ dữ liệu khó nhất; kết quả locked test thấp hơn CV khoảng `0.0277`.
- Cả ba pipeline đã tạo đầy đủ prediction, confidence, feature importance và Learning Path.
- Các file có `_v27` trong tên vẫn là artifact legacy; kết quả chính thức mới là các file không chứa `_v27`.

### 4.2. PostgreSQL

| Run ID | Dataset | Predictions | Learning Paths | Metrics |
|---:|---|---:|---:|---:|
| 7 | Student-Por | 130 | 130 | 1 |
| 8 | Student-Mat | 79 | 79 | 1 |
| 9 | xAPI | 96 | 96 | 1 |

## 5. Công việc còn lại

### 5.1. Hoàn thiện bàn giao

- Rà soát và xóa hoặc lưu trữ riêng các artifact legacy.
- Đưa bảng kết quả chính thức vào chương thực nghiệm của khóa luận.
- Chọn một số Learning Path tiêu biểu để minh họa trong báo cáo.
- Commit mã nguồn, schema, báo cáo và kết quả mới.

## 6. Đánh giá tiến độ

| Hạng mục | Trạng thái |
|---|---|
| Refactor CNN-BiLSTM + MLP | Hoàn thành |
| Weighted Cross-Entropy + SMOTE/ADASYN | Hoàn thành |
| Tích hợp Optuna | Hoàn thành |
| Learning Path theo luật | Hoàn thành |
| PostgreSQL persistence | Hoàn thành và đã kiểm tra |
| Unit test | Hoàn thành, 7/7 test đạt |
| Kiểm tra end-to-end | Hoàn thành |
| Chạy chính thức Student-Mat | Hoàn thành, 50 trials |
| Chạy chính thức Student-Por | Hoàn thành, 50 trials |
| Chạy chính thức xAPI | Hoàn thành, 50 trials |
| Ghi PostgreSQL | Hoàn thành, runs 7-9 |
| Cập nhật kết quả trong báo cáo tiến độ | Hoàn thành |

**Ước lượng tiến độ kỹ thuật:** 100%.  
**Phần còn lại:** biên tập chương kết quả, dọn artifact legacy và bàn giao/commit.

## 7. Kết luận tiến độ

Hệ thống đã được điều chỉnh đúng hướng đề cương về kiến trúc, cơ chế khuyến nghị và lưu trữ dữ liệu. Toàn bộ luồng từ xử lý dữ liệu, Optuna 50 trials, ensemble 5 seed, dự đoán, sinh Learning Path đến ghi PostgreSQL đã chạy thành công trên cả ba bộ dữ liệu.

Các kết quả chính thức hiện đã sẵn sàng để sử dụng trong chương thực nghiệm. Công việc tiếp theo không còn là phát triển kỹ thuật cốt lõi mà là biên tập báo cáo, lựa chọn trường hợp minh họa và chuẩn bị bàn giao.
