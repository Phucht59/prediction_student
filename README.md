# Dự đoán thành tích và cảnh báo sớm sinh viên

Đây là repository cuối của khóa luận:

> **Xây dựng mô hình học kết hợp để dự đoán thành tích học tập sinh viên**

Hệ thống công khai một họ mô hình chính duy nhất: **CNN-BiLSTM**. Ba định
danh triển khai là:

| Dataset | Mô hình cuối | Bài toán | Macro-F1 |
|---|---|---|---:|
| Student-Mat | CNN-BiLSTM MAT | Low / Medium / High | **0.9015** |
| Student-Por | CNN-BiLSTM POR | Low / Medium / High | **0.8623** |
| OULAD | See dual endpoint authority below | At-risk / Not-at-risk | **0.8281 legacy / 0.7984 strict** |

OULAD now has explicitly separate endpoint authorities and early-warning evidence:

- **Legacy endpoint authority:** H0 CNN-BiLSTM, Macro-F1 **0.8281**, using a
  conservative score-availability proxy whose exact release-time validity
  cannot be fully verified from OULAD.
- **Strict endpoint authority:** the frozen H1 architecture at
  `F2_MIDDLE_OFFICIAL_SINGLE_CUTOFF`, Macro-F1 **0.7984**, excluding
  unverifiable score-progress values.
- **Secondary early warning:** the frozen shared-checkpoint evaluation at
  20%, 35%, 50% and 75%; its stage metrics remain unchanged.

The historical H0 CNN-BiLSTM endpoint result (**0.8281**) and MLP result
(**0.8283**) remain legacy endpoint evidence under the score proxy. They share
the target, population and outer folds with H1, but not the strict feature-
availability protocol. Phase 7 found that H1 did not improve either historical
comparator at the endpoint; no post-test tuning was performed.

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
| E1 | 20% | 0.7194 | 0.7126 | 0.7136 |
| E2 | 35% | 0.7606 | 0.7480 | 0.7506 |
| M1 | 50% | 0.8063 | 0.7894 | 0.7940 |
| L1 | 75% | 0.8664 | 0.8385 | 0.8503 |

Kết quả **strict final endpoint** của H1 có Macro-F1 **0.7984**, Balanced
Accuracy **0.7922**, PR-AUC **0.8630**, Risk Precision **0.8045**, Risk Recall
**0.6961**, Risk F1 **0.7464** và ECE **0.0120**. Kết quả H0 **0.8281** bên
dưới được giữ dưới vai trò legacy endpoint theo score-availability proxy,
không thuộc strict feature protocol và không phải kết quả H1.

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

### So sánh ML với mô hình chính thức

Macro-F1 trên evaluation authority chính thức:

| Model | Student-Mat | Student-Por | OULAD |
|---|---:|---:|---:|
| Logistic Regression | 0.8952 | 0.8379 | 0.8247 |
| Decision Tree | **0.9024** | 0.8461 | 0.8061 |
| Random Forest | 0.8998 | 0.8514 | 0.8220 |
| HistGradientBoosting | 0.8697 | 0.8441 | 0.8241 |
| SVM | 0.8710 | 0.8502 | 0.8250 |
| XGBoost | 0.8815 | **0.8677** | 0.8259 |
| MLP | 0.8595 | 0.8304 | **0.8283** |
| CNN-BiLSTM H0 (historical endpoint comparator) | — | — | 0.8281 |
| **Final CNN-BiLSTM family (strict H1 on OULAD)** | **0.9015** | **0.8623** | **0.7984** |

Các giá trị in đậm cho thấy model tốt nhất không giống nhau ở mọi dataset:
Decision Tree nhỉnh hơn ở MAT, XGBoost nhỉnh hơn ở POR và MLP tốt hơn H1 ở
OULAD endpoint. H1 vẫn là kiến trúc hybrid được đánh giá theo protocol đã đóng
băng; kết quả Phase 7 được báo cáo trung thực và không có claim universal
superiority.

### Macro-F1 theo thời điểm UCI

| Model | MAT S0 | MAT S1 | MAT S2 | POR S0 | POR S1 | POR S2 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.3979 | 0.7224 | 0.8754 | 0.4351 | 0.7572 | 0.8404 |
| Decision Tree | 0.4104 | 0.7310 | 0.8623 | 0.4312 | 0.7198 | 0.7954 |
| Random Forest | 0.4296 | 0.7118 | **0.8893** | 0.5082 | **0.7835** | **0.8571** |
| HistGradientBoosting | 0.4379 | 0.7014 | 0.8542 | 0.4586 | 0.6833 | 0.8182 |
| SVM | **0.4523** | 0.7245 | 0.8501 | 0.4493 | 0.7529 | 0.7998 |
| XGBoost | 0.4116 | 0.7092 | 0.8741 | 0.4533 | 0.7648 | 0.8428 |
| MLP | 0.4219 | 0.7306 | 0.8547 | 0.4002 | 0.7523 | 0.8520 |
| **CNN-BiLSTM** | 0.4136 | **0.7438** | 0.8461 | **0.5089** | 0.7542 | 0.8519 |

### Macro-F1 theo thời điểm OULAD

Các hàng dưới dùng threshold được chọn bằng inner-OOF, không dùng outer label:

| Model | E1 20% | E2 35% | M1 50% | L1 75% |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.6984 | 0.7444 | 0.7886 | 0.8253 |
| Decision Tree | 0.6125 | 0.6899 | 0.7547 | 0.8046 |
| Random Forest | 0.6969 | 0.7422 | 0.7897 | 0.8304 |
| HistGradientBoosting | 0.7020 | **0.7524** | **0.7938** | 0.8284 |
| SVM | 0.7032 | 0.7481 | 0.7922 | **0.8324** |
| XGBoost | **0.7070** | **0.7524** | 0.7911 | 0.8320 |
| MLP | 0.6993 | 0.7495 | 0.7930 | 0.8271 |
| **CNN-BiLSTM** | 0.7003 | 0.7435 | 0.7852 | 0.8062 |

Accuracy, Balanced Accuracy, PR-AUC, ECE, per-class/risk metrics và confusion
matrix đầy đủ nằm trong `artifacts/final/final_stage_results.csv` và các thư
mục stage evidence.

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

## Where is the code?

| Need | Location |
| --- | --- |
| Frozen prediction authority | `configs/final/final_model_authority.yaml` |
| Final recommendation runtime | `src/recommend_hybrid/` |
| Recommendation CLI | `scripts/recommend_hybrid/generate_plan.py` |
| Final validation | `scripts/recommend_hybrid/validate_final_evidence_recommender.py` |
| Final registry and metrics | `artifacts/final/recommendation/`, `artifacts/recommend_hybrid/final/` |
| Repository map | `docs/project_map/PROJECT_CODE_MAP.md` |

The final recommendation component is an evidence-based deterministic policy consuming frozen Hybrid CNN-BiLSTM predictions. Weak supervision and the neural ranker are scientific diagnostics only.

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
