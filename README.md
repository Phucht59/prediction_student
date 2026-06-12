# Hệ thống dự đoán và khuyến nghị thành tích học tập

Dự án phục vụ khóa luận tốt nghiệp về phân loại thành tích học tập thành ba nhóm `Low`, `Medium`, `High`. Kiến trúc và chức năng bám theo đề cương đã duyệt: CNN-BiLSTM cho dữ liệu tuần tự, MLP cho thông tin ngữ cảnh, SMOTE/ADASYN cho mất cân bằng lớp, learning path theo luật và PostgreSQL cho lưu trữ kết quả.

## Kiến trúc

```text
Sequential input (G1, G2 hoặc interaction logs)
    -> Conv1D -> Bi-LSTM -> Attention Pooling

Context input (numerical + categorical)
    -> MLP

Sequence vector + Context vector
    -> Dense fusion -> 3-class logits -> Softmax probabilities
```

Hệ thống chỉ sử dụng CNN, Bi-LSTM và MLP. Không sử dụng DeepFM, DCN-V2, Transformer, Focal Loss hoặc counterfactual search.

## Quy trình huấn luyện

1. Tách `locked test` 20% trước mọi bước tối ưu.
2. Feature engineering, mã hóa và chuẩn hóa chỉ fit trên train.
3. Cân bằng lớp bằng SMOTE hoặc ADASYN.
4. Optuna tối ưu mô hình bằng Stratified 5-fold cross-validation.
5. Huấn luyện ensemble với các seed `42, 123, 155, 156, 2025`.
6. Đánh giá bằng F1-Macro, Accuracy, Precision, Recall, RMSE và R².
7. Sinh learning path theo tuần cho từng sinh viên.
8. Lưu feature gốc, dự đoán, confidence, metrics và learning path vào PostgreSQL.

## Cấu trúc mã nguồn

```text
src/models.py           CNN-BiLSTM + Context MLP
src/data_pipeline.py    split, preprocessing, SMOTE/ADASYN, feature selection
src/train_pipeline.py   weighted CrossEntropyLoss, Optuna, early stopping
src/explainability.py   permutation importance và rule-based learning paths
src/evaluation.py       báo cáo và PostgreSQL persistence
scripts/run_pipeline.py pipeline end-to-end
database/schema.sql     schema PostgreSQL
```

## Cài đặt

```powershell
conda env create -f environment.yml
conda activate kltn
```

Hoặc:

```powershell
pip install -r requirements.txt
```

Thiết lập `.env` bằng `DATABASE_URL` hoặc các biến `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`.

## Chạy pipeline

```powershell
py scripts/run_pipeline.py --dataset student-mat --n-trials 30
py scripts/run_pipeline.py --dataset student-por --n-trials 30
py scripts/run_pipeline.py --dataset xapi --n-trials 30
```

Pipeline mặc định ghi PostgreSQL sau khi đánh giá. Chỉ dùng `--skip-postgres` cho kiểm thử phát triển ngoại tuyến.

## Đầu ra

- `models/saved/final/`: trọng số ensemble và best parameters.
- `reports/final/metrics/`: metrics locked test.
- `reports/final/predictions/`: feature gốc, nhãn, xác suất và confidence.
- `reports/final/explanations/`: permutation feature importance.
- `reports/final/recommendations/`: learning path theo giai đoạn.
- PostgreSQL: `paper_runs`, `paper_predictions`, `paper_evaluation_metrics`, `paper_learning_recommendations`.

## Kiểm thử

```powershell
pytest -q
```
