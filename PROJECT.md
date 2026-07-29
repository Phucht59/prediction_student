# PROJECT — Ngữ cảnh kỹ thuật để viết báo cáo khóa luận

Tài liệu này là bản đồ nội dung của project cuối. Nó giải thích câu hỏi nghiên
cứu, hợp đồng dữ liệu, kiến trúc, protocol đánh giá, kết quả và giới hạn phát
biểu. Khi viết luận văn, ưu tiên số liệu trong machine-readable artifact được
liệt kê ở cuối tài liệu; không lấy số từ thư mục `test_lab`.

## 1. Bài toán và phạm vi

Khóa luận xây dựng một họ mô hình học kết hợp **CNN-BiLSTM** cho hai nhóm bài
toán:

1. Phân loại thành tích UCI Student Performance thành Low, Medium, High.
2. Dự báo rủi ro OULAD thành At-risk và Not-at-risk theo tiến trình học.

Sau dự báo OULAD, hệ thống tạo risk profile và recommendation plan có
abstention, constraint và lineage, rồi lưu evidence vào PostgreSQL.

Ba định danh chính thức:

| ID | Tên hiển thị | Dataset |
|---|---|---|
| `cnn_bilstm_mat` | CNN-BiLSTM MAT | Student-Mat |
| `cnn_bilstm_por` | CNN-BiLSTM POR | Student-Por |
| `cnn_bilstm_oulad` | CNN-BiLSTM OULAD | OULAD |

Tên version nghiên cứu hoặc candidate nội bộ chỉ được dùng trong provenance,
không phải tên model công khai.

## 2. Các câu hỏi nghiên cứu

### RQ1 — Chất lượng mô hình chính

CNN-BiLSTM dự đoán thành tích/rủi ro tốt đến đâu trên outer evaluation đóng
băng, so với các comparator ML và deep learning độc lập?

### RQ2 — Khả năng cảnh báo sớm

Khi chỉ quan sát thông tin đầu kỳ, mô hình còn dự báo được đến mức nào? Hiệu
năng tăng ra sao khi có thêm G1/G2 hoặc thêm tuần hoạt động OULAD?

### RQ3 — Giá trị của kiến trúc hybrid

CNN và BiLSTM kết hợp có lợi ích gì so với CNN-only, BiLSTM-only và các
comparator tabular? Câu trả lời phải theo từng dataset/stage; không tuyên bố
universal superiority.

### RQ4 — Khuyến nghị có an toàn kỹ thuật không?

Policy có tôn trọng cutoff, sensitive-feature prohibition, action cap,
workload, lineage, abstention và deterministic replay hay không? Project chưa
có expert labels nên không trả lời hiệu quả can thiệp.

## 3. Hợp đồng target và thông tin

### 3.1 UCI

`G3` là điểm cuối 0–20 và chỉ được dùng tạo target:

```text
Low     : 0 <= G3 < 10
Medium  : 10 <= G3 < 15
High    : 15 <= G3 <= 20
```

Boundary bắt buộc: 9→Low, 10→Medium, 14→Medium, 15→High, 20→High.
G3 không xuất hiện trong predictor, preprocessing fit hoặc derived feature.

Ba view của cùng bản ghi:

| Stage | Thông tin grade được phép | Mask |
|---|---|---|
| S0_EARLY_NO_GRADE | không G1, không G2 | `[0, 0]` |
| S1_MID_G1_ONLY | chỉ G1 | `[1, 0]` |
| S2_LATE_G1_G2 | G1 và G2 | `[1, 1]` |

Context feature giống nhau giữa các stage; availability mask mới quyết định
grade timestep nào hợp lệ. Một base record có ba view cân bằng trong training.

### 3.2 OULAD

Nhãn:

- At-risk: `Withdrawn` hoặc `Fail`
- Not-at-risk: `Pass` hoặc `Distinction`

Feature contract gồm:

- 47 kênh temporal (16 kênh cơ sở và các biểu diễn hợp lệ);
- 161 aggregate feature cơ sở;
- static feature về module/presentation và lịch sử đăng ký;
- stage context: progress fraction, observed weeks, weeks remaining,
  assessment availability.

`final_result`, `date_unregistration`, `code_presentation` và mọi tín hiệu sau
cutoff bị cấm làm predictor. Bốn stage:

| Stage | Phần tiến trình quan sát |
|---|---:|
| E1_EARLY_20PCT | 20% |
| E2_EARLY_35PCT | 35% |
| M1_MIDDLE_FROZEN | 50% |
| L1_LATE_75PCT | 75%, có outcome guard 14 ngày |

Future OULAD không được mở: `LOCKED_NOT_EXECUTED`.

## 4. Kiến trúc mô hình

### 4.1 Nguyên tắc thống nhất

Không có “CNN-BiLSTM S0”, “CNN-BiLSTM S1” hoặc model riêng cho từng stage.
Mỗi dataset/fold/seed có một training run, một checkpoint và một classification
head dùng chung. Stage chỉ là availability view của cùng estimator.

Điều này cho phép diễn giải thay đổi metric là do lượng thông tin khả dụng,
không phải do đổi model identity.

### 4.2 CNN-BiLSTM UCI

Pipeline nhận:

- temporal tensor `[batch, 2, 7]` cho G1/G2;
- availability mask `[batch, 2]`;
- context vector sau preprocessing training-only.

Các khối chính:

```text
temporal grades + availability mask
  -> input projection
  -> mask-aware CNN
  -> BiLSTM
  -> masked pooling
                         \
context -> context MLP ----> gated/fused representation
                               -> shared 3-class head
```

Loss là mean cross-entropy của ba view S0/S1/S2. Inner selection tối ưu mean
stage Macro-F1, rồi worst-stage Macro-F1, Low F1, NLL và độ đơn giản. Không
transfer giữa Student-Por và Student-Mat trong benchmark stage-aware, không
pretrained checkpoint, không synthetic resampling.

Mô hình MAT chính thức có transfer learning shared trunk và subject-specific
head được lựa chọn bằng inner validation. Đây là cấu hình kỹ thuật của
`cnn_bilstm_mat`, không phải tên một model khác.

### 4.3 CNN-BiLSTM OULAD

```text
47 temporal channels
  -> input projection
  -> multi-kernel CNN
  -> residual
  -> BiLSTM
  -> masked pooling
          + aggregate branch
          + static branch
  -> gated residual fusion
  -> shared representation
       -> risk head (main)
       -> survival head (auxiliary, 0.15)
       -> outcome head (auxiliary, 0.15)
```

Mô hình chính thức có 100,938 tham số và temporal pretraining đã đóng băng.
Trong benchmark bốn stage, cùng checkpoint/fold/seed được replay ở E1/E2/M1/L1.

## 5. Protocol đánh giá

### 5.1 Split và selection

- Student-Mat / Student-Por: 5 frozen outer folds, 3 inner folds.
- OULAD: 3 outer folds grouped theo `id_student`.
- Preprocessing chỉ fit trên outer-training hoặc inner-training tương ứng.
- Hyperparameter/epoch/threshold chỉ chọn bằng inner data.
- Outer labels không tham gia selection.
- Fixed seeds: 42, 1201, 2026, 3407, 7319.
- Báo cáo probability ensemble trung bình qua toàn bộ seed; không chọn best
  seed.
- Paired bootstrap dùng 5,000 replicate; OULAD resample theo student.

### 5.2 Imbalance

Plain SMOTE/ADASYN không được dùng trên categorical UCI đã label encode. Sampler
không fit trước split hoặc bằng validation data. Raw OULAD temporal tensor không
synthetic oversampling. Deep model có thể dùng class weighting/focal loss theo
cấu hình đã đăng ký.

### 5.3 Comparator

Mười family cho mỗi dataset:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. HistGradientBoosting
5. SVM
6. XGBoost
7. MLP
8. CNN-only
9. BiLSTM-only
10. CNN-BiLSTM

CNN-only/BiLSTM-only là ablation. ML comparator dùng cùng record, outer fold,
metric và training-only preprocessing theo feature contract tương ứng.

## 6. Kết quả chính thức đã đóng băng

| Mô hình | Macro-F1 | Balanced Accuracy | PR-AUC |
|---|---:|---:|---:|
| CNN-BiLSTM MAT | 0.9014601961 | 0.9021 | 0.9442 |
| CNN-BiLSTM POR | 0.8622587168 | 0.8676 | 0.9147 |
| CNN-BiLSTM OULAD | 0.8280835946 | 0.8203 | 0.8934 |

OULAD bổ sung:

- Risk Precision: 0.8522
- Risk Recall: 0.7236
- Risk F1: 0.7826
- ECE: 0.0087

MLP comparator Macro-F1:

- Student-Mat: 0.8595069899
- Student-Por: 0.8303986867
- OULAD: 0.8282857900

Paired bootstrap CNN-BiLSTM so với MLP:

- MAT: Δ Macro-F1 xấp xỉ +0.0420, CI 95% [0.0148, 0.0709].
- POR: Δ xấp xỉ +0.0319, CI 95% [0.0061, 0.0587].
- OULAD: Δ xấp xỉ -0.0002, CI 95% [-0.0036, 0.0031], chưa đủ bằng
  chứng về khác biệt.

Không dùng từ “tương đương” chỉ vì CI chứa 0.

## 7. Kết quả stage-aware UCI

### 7.1 Student-Mat

| Stage | Accuracy | Balanced Acc. | Macro-F1 | PR-AUC | Low P/R/F1 | Medium P/R/F1 | High P/R/F1 |
|---|---:|---:|---:|---:|---|---|---|
| S0 | 0.4152 | 0.4396 | 0.4136 | 0.4624 | .5424/.4923/.5161 | .4961/.3333/.3988 | .2432/.4932/.3258 |
| S1 | 0.7367 | 0.7754 | 0.7438 | 0.8301 | .7025/.8538/.7708 | .8014/.6094/.6923 | .6923/.8630/.7683 |
| S2 | 0.8405 | 0.8735 | 0.8461 | 0.9462 | .7785/.9462/.8542 | .9272/.7292/.8163 | .8023/.9452/.8679 |

### 7.2 Student-Por

| Stage | Accuracy | Balanced Acc. | Macro-F1 | PR-AUC | Low P/R/F1 | Medium P/R/F1 | High P/R/F1 |
|---|---:|---:|---:|---:|---|---|---|
| S0 | 0.5116 | 0.5886 | 0.5089 | 0.4970 | .4514/.6500/.5328 | .7395/.4211/.5366 | .3408/.6947/.4573 |
| S1 | 0.7735 | 0.8209 | 0.7542 | 0.8531 | .5570/.8800/.6822 | .9094/.7201/.8037 | .7063/.8626/.7766 |
| S2 | 0.8706 | 0.9007 | 0.8519 | 0.9283 | .6475/.9000/.7531 | .9613/.8325/.8923 | .8581/.9695/.9104 |

Diễn giải: G1 tạo bước tăng lớn; G2 tiếp tục cải thiện. S0 cho thấy tín hiệu
context đơn thuần còn hạn chế, vì vậy không được quảng bá kết quả S2 như chất
lượng cảnh báo đầu kỳ.

## 8. Kết quả stage-aware OULAD

### 8.1 CNN-BiLSTM

| Stage | Accuracy | Balanced Acc. | Macro-F1 |
|---|---:|---:|---:|
| E1 | 0.7168 | 0.6992 | 0.7003 |
| E2 | 0.7573 | 0.7398 | 0.7435 |
| M1 | 0.7926 | 0.7877 | 0.7852 |
| L1 | 0.8130 | 0.8213 | 0.8062 |

### 8.2 Ý nghĩa comparator

Comparator tốt nhất có thể thay đổi theo stage. Evidence hiện tại cho thấy ML
có thể cao hơn CNN-BiLSTM ở một số thời điểm; vì vậy luận văn chỉ kết luận kiến
trúc hybrid là mô hình chính được lựa chọn theo protocol chính thức, không kết
luận nó thống trị phổ quát.

Headline OULAD 0.8281 và bảng stage-aware trả lời hai câu hỏi khác nhau:
headline là model authority khóa luận; bảng stage là khả năng vận hành theo
cutoff với một estimator chung.

## 9. Recommendation và PostgreSQL

Luồng:

```text
prediction
  -> calibrated risk profile
  -> observed-state evidence
  -> constrained recommendation policy
  -> GENERATED / PARTIAL_EVIDENCE / ABSTAINED plan
  -> PostgreSQL
```

Trạng thái đóng băng:

- risk profiles: 15,378
- plan objects: 15,378
- actions: 27,355
- GENERATED: 10,953
- PARTIAL_EVIDENCE: 1,209
- ABSTAINED: 3,216
- reviews: 0
- expert status: `PENDING_EXPERT_LABELS`

ABSTAINED vẫn có plan object để trace, nhưng `recommended_actions = []`.
Schema công khai tập trung vào `system`, `catalog`, `ml`, `recommendation`.

## 10. Nguồn evidence

| Nội dung | Authority |
|---|---|
| Headline model | `artifacts/final/final_results.json` |
| Stage result | `artifacts/final/final_stage_results.csv` |
| Model registry | `artifacts/final/model_registry.json` |
| UCI stage evidence | `artifacts/final/unified_stage_aware_uci/` |
| OULAD stage evidence | `artifacts/final/unified_stage_aware_oulad/` |
| UCI target/split/safety | `artifacts/final/teacher_feedback_validation/` |
| Tuning | `artifacts/final/tuning_evidence/` |
| Ablation | `artifacts/final/ablation_evidence/` |
| Recommendation | `artifacts/final/recommendation/` |
| Database | `artifacts/final/database/` |
| Checksums | `artifacts/final/checksum_manifest.json` và evidence manifests |

Các file trong `artifacts/final/provenance/` chỉ giải thích nguồn gốc hoặc artifact
đã supersede. `test_lab/` là lịch sử local bị Git ignore và không phải dependency
cho inference, validation hay bảo vệ kết quả.

## 11. Cách tái lập và kiểm tra

Validation an toàn:

```powershell
python project.py final status
python project.py final report
python project.py final validate
python project.py pipeline uci validate
python project.py pipeline oulad validate
pytest
ruff check .
```

`final validate` không train. Training chỉ diễn ra khi gọi explicit subcommand
`train`; việc đó không cần thiết để viết báo cáo từ release đã đóng băng.

## 12. Claim boundaries khi viết luận văn

Được phép viết:

- CNN-BiLSTM là họ mô hình chính thức của khóa luận.
- Kết quả outer evaluation đạt các metric đã đóng băng.
- Hiệu năng thay đổi theo lượng thông tin được phép quan sát.
- Recommendation vượt technical validation và deterministic replay.
- Comparator cho biết vị trí tương đối của hybrid theo dataset/stage.

Không được viết:

- Recommendation đã chứng minh cải thiện kết quả học tập.
- 79.09% là accuracy.
- CI đi qua 0 nghĩa là hai model tương đương.
- S2/L1 là cảnh báo “rất sớm”.
- CNN-BiLSTM luôn tốt hơn mọi ML.
- Future OULAD đã được đánh giá.
- Expert evaluation đã hoàn tất.

## 13. Trạng thái cuối

- Scientific metrics changed: **NO**
- Model retrained trong cleanup: **NO**
- Outer test used for tuning: **NO**
- Best seed selection: **NO**
- Future OULAD accessed: **NO**
- xAPI: **ABSENT**
- Recommendation expert evaluation: **PENDING_EXPERT_LABELS**
