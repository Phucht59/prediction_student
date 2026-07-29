# Dự đoán thành tích và cảnh báo sớm sinh viên

Đây là repository cuối của khóa luận:

> **Xây dựng mô hình học kết hợp để dự đoán thành tích học tập sinh viên**

Hệ thống công khai một họ mô hình chính duy nhất: **CNN-BiLSTM**. Ba định
danh triển khai là:

| Dataset | Mô hình cuối | Bài toán | Macro-F1 |
|---|---|---|---:|
| Student-Mat | CNN-BiLSTM MAT | Low / Medium / High | **0.9015** |
| Student-Por | CNN-BiLSTM POR | Low / Medium / High | **0.8623** |
| OULAD | CNN-BiLSTM OULAD | At-risk / Not-at-risk | **0.8281** |

Các con số trên là kết quả khóa luận đã đóng băng. Pipeline validation không
train lại, không chọn seed tốt nhất và không thay đổi checkpoint.

## Luồng hệ thống

```text
DATA
  -> PREPROCESSING
  -> CNN-BiLSTM
  -> PREDICTION
  -> RISK PROFILE
  -> RECOMMENDATION
  -> POSTGRESQL / EVIDENCE
```

### Một mô hình, nhiều thời điểm dự báo

Nghiên cứu cảnh báo sớm không tạo một mô hình hybrid khác cho từng thời điểm.
Trong mỗi dataset và outer fold, **một estimator CNN-BiLSTM được huấn luyện một
lần và dùng chung checkpoint ở mọi stage**. Stage chỉ thay đổi phần thông tin
được phép quan sát:

- UCI `S0`: chưa có G1/G2.
- UCI `S1`: có G1, không có G2.
- UCI `S2`: có G1 và G2.
- OULAD `E1`, `E2`, `M1`, `L1`: lần lượt quan sát 20%, 35%, 50% và 75%
  tiến trình học tập.

Mask khả dụng ngăn dữ liệu của stage sau đi vào stage trước. G3 chỉ tạo nhãn
UCI, tuyệt đối không phải predictor. Với OULAD, mọi feature phải nằm trước
cutoff của stage.

## Kết quả cảnh báo sớm UCI

Nhãn UCI:

- `Low`: `0 <= G3 < 10`
- `Medium`: `10 <= G3 < 15`
- `High`: `15 <= G3 <= 20`

### Student-Mat — CNN-BiLSTM dùng chung checkpoint

| Stage | Accuracy | Balanced Acc. | Macro-F1 | PR-AUC | Low F1 | Medium F1 | High F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0 — không G1/G2 | 0.4152 | 0.4396 | 0.4136 | 0.4624 | 0.5161 | 0.3988 | 0.3258 |
| S1 — chỉ G1 | 0.7367 | 0.7754 | 0.7438 | 0.8301 | 0.7708 | 0.6923 | 0.7683 |
| S2 — G1+G2 | 0.8405 | 0.8735 | 0.8461 | 0.9462 | 0.8542 | 0.8163 | 0.8679 |

### Student-Por — CNN-BiLSTM dùng chung checkpoint

| Stage | Accuracy | Balanced Acc. | Macro-F1 | PR-AUC | Low F1 | Medium F1 | High F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0 — không G1/G2 | 0.5116 | 0.5886 | 0.5089 | 0.4970 | 0.5328 | 0.5366 | 0.4573 |
| S1 — chỉ G1 | 0.7735 | 0.8209 | 0.7542 | 0.8531 | 0.6822 | 0.8037 | 0.7766 |
| S2 — G1+G2 | 0.8706 | 0.9007 | 0.8519 | 0.9283 | 0.7531 | 0.8923 | 0.9104 |

Kết quả S2 của nghiên cứu stage-aware không thay thế headline 0.9015/0.8623.
Headline thuộc mô hình khóa luận chính thức; nghiên cứu stage-aware dùng một
estimator chung cho cả ba stage để trả lời riêng câu hỏi cảnh báo sớm.

## Kết quả cảnh báo sớm OULAD

CNN-BiLSTM OULAD sử dụng 47 kênh temporal, nhánh aggregate, nhánh static,
masked pooling, gated residual fusion và các auxiliary objective survival /
outcome với trọng số 0.15. Một checkpoint của mỗi fold/seed phục vụ cả bốn
stage.

| Stage | Tiến trình quan sát | Accuracy | Balanced Acc. | Macro-F1 |
|---|---:|---:|---:|---:|
| E1 | 20% | 0.7168 | 0.6992 | 0.7003 |
| E2 | 35% | 0.7573 | 0.7398 | 0.7435 |
| M1 | 50% | 0.7926 | 0.7877 | 0.7852 |
| L1 | 75% | 0.8130 | 0.8213 | 0.8062 |

Mô hình chính thức OULAD vẫn có Macro-F1 **0.8281**, Balanced Accuracy
**0.8203**, PR-AUC **0.8934**, Risk Precision **0.8522**, Risk Recall
**0.7236**, Risk F1 **0.7826** và ECE **0.0087**.

## Mô hình so sánh

Mỗi dataset giữ cùng một bộ comparator phục vụ phản biện khoa học:

- Logistic Regression
- Decision Tree
- Random Forest
- HistGradientBoosting
- SVM
- XGBoost
- MLP
- CNN-only
- BiLSTM-only
- CNN-BiLSTM

CNN-only và BiLSTM-only là ablation/comparator, không phải mô hình hybrid cuối.
Kết quả ML được giữ vì CNN-BiLSTM không được tuyên bố là luôn vượt mọi phương
pháp ML ở mọi stage. Tất cả comparator dùng outer folds đóng băng, preprocessing
fit trên training partition và không tune bằng outer validation.

## Khuyến nghị

Mô-đun khuyến nghị biến dự báo rủi ro thành kế hoạch có kiểm soát:

| Trạng thái | Số bản ghi |
|---|---:|
| GENERATED | 10,953 |
| PARTIAL_EVIDENCE | 1,209 |
| ABSTAINED | 3,216 |
| Tổng risk profile / plan | 15,378 |
| Tổng action | 27,355 |

Tỷ lệ generated hoặc partial là **79.09%**; đây là coverage, không phải
accuracy. Hệ thống có 0 vi phạm workload, action-cap, duplicate, lineage,
post-cutoff và sensitive-attribute; deterministic replay đạt PASS. Đánh giá
chuyên gia vẫn là `PENDING_EXPERT_LABELS`, do đó không có tuyên bố hiệu quả
nhân quả.

## Chạy validation

Tạo môi trường từ `requirements.txt` hoặc `requirements-lock.txt`, sau đó:

```powershell
python project.py final status
python project.py final report
python project.py final validate
python project.py pipeline uci validate
python project.py pipeline oulad validate
pytest
```

Các lệnh trên chỉ đọc/replay evidence; không train model. Lệnh train, nếu thực
sự cần tái lập nghiên cứu, phải gọi rõ qua `pipeline uci train` hoặc
`pipeline oulad train`.

`project.py` được giữ lại vì đây là CLI điều phối validation, evidence và
database — không phải file viết luận văn.

## Cấu trúc repository

```text
artifacts/final/       evidence, prediction, metric, checkpoint và checksum
configs/final/         hợp đồng UCI, OULAD, recommendation và model registry
data/                  dữ liệu/manifest được phép công khai
database/final/        schema và migration PostgreSQL cuối
docs/                  tài liệu kiến trúc, dữ liệu và tái lập
reports/final/         báo cáo kỹ thuật dùng để bảo vệ kết quả
scripts/final/         trình tạo và kiểm tra evidence
src/
  data/                data contract và preprocessing
  models/              CNN, BiLSTM, CNN-BiLSTM
  pipelines/           pipeline stage-aware UCI và OULAD
  recommendation/      risk profile và policy
  database/            persistence/validation
  final_release/       catalog và freeze guard
tests/                 unit, release và database tests
test_lab/              lịch sử nghiên cứu local, bị Git ignore
```

## Nguồn sự thật

- Kết quả chính: `artifacts/final/final_results.json`
- Kết quả theo stage: `artifacts/final/final_stage_results.csv`
- Registry: `artifacts/final/model_registry.json`
- Cấu hình UCI: `configs/final/uci_prediction.yaml`
- Cấu hình OULAD: `configs/final/oulad_prediction.yaml`
- Khuyến nghị: `artifacts/final/recommendation/`
- Báo cáo chi tiết để viết luận văn: `PROJECT.md`

## Giới hạn khoa học

- Đây là dự báo quan sát, không chứng minh can thiệp gây ra kết quả tốt hơn.
- Kết quả stage muộn không nên được mô tả như cảnh báo sớm.
- ML có thể ngang hoặc tốt hơn CNN-BiLSTM ở một số stage OULAD.
- Future OULAD giữ trạng thái `LOCKED_NOT_EXECUTED`.
- xAPI không thuộc phạm vi release cuối.
- Không có expert label giả cho recommendation.
