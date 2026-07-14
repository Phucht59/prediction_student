# 03 — Chẩn đoán mô hình dự đoán

## 1. Bài toán mà code thực sự đang giải quyết

Pipeline active giải bài toán sau:

> Với hai điểm đánh giá có thứ tự `G1` và `G2` của học sinh trong cùng môn/năm học, dự đoán lớp của điểm cuối `G3`: Low (0–9), Medium (10–14), hoặc High (15–20), tại cutoff sau khi đã biết G2.

Đây là **late-stage three-class classification với hai tín hiệu số có thứ tự**, không phải dự báo chuỗi dài qua nhiều học kỳ. Code có scenario configs khác, nhưng `StudentDataset` hard-code G1/G2 và yêu cầu ít nhất hai sequence columns; pre-assessment/early-warning hiện mới là contract, chưa phải pipeline chạy được.

Bài toán vẫn liên quan đề cương vì CNN và BiLSTM có thể được nghiên cứu trên hai assessment liên tiếp. Tuy nhiên, claim phải hạ đúng mức: mô hình học quan hệ giữa hai điểm và một bước thay đổi, không chứng minh khả năng học long-term temporal dynamics.

## 2. Kiến trúc active và giới hạn biểu diễn

### 2.1 Đường tensor

```text
[batch, length=2, channels=1]  — G1, G2
    → transpose [batch, 1, 2]
    → Conv1D(kernel=1, padding=0, channels=16)
    → BatchNorm + ReLU + Dropout
    → transpose [batch, 2, 16]
    → BiLSTM(hidden=32)
    → representation
    → Linear(3)
    → softmax / argmax
```

Selected historical config có 13.059 tham số, batch 32, CNN kernel 1, BiLSTM hidden 32, dropout khoảng 0,457 và sequence dropout khoảng 0,197. Model active bỏ toàn bộ numeric/categorical context; các feature selector và context tensors không đi vào forward path.

### 2.2 CNN và BiLSTM thực sự học được gì

- Với kernel 1, Conv1D dùng cùng phép biến đổi pointwise ở từng timestep. CNN **không trực tiếp trộn G1 và G2** và không trích pattern cục bộ dài hơn một điểm.
- BiLSTM nhìn đúng hai timestep. Nó có thể học mức điểm, hướng tăng/giảm và interaction có thứ tự; lợi thế recurrent so với MLP nhỏ chưa được chứng minh.
- 13.059 tham số trên tối đa 252 outer-train records, rồi còn ít hơn trong inner/early-stop splits, tạo tỷ lệ capacity/data cao.
- BatchNorm trong mô hình nhỏ kết hợp batch cuối bị drop và SWA không update BatchNorm statistics là thêm nguồn bất ổn.

Kết luận: architecture không “sai” về tensor, nhưng complexity và inductive bias chưa khớp rõ với sequence length 2. CNN–BiLSTM phải được giữ để kiểm chứng theo đề cương, không được mặc định là champion.

## 3. Bằng chứng thống kê từ development set

Chỉ 316 development records được dùng cho các số dưới đây; 79 observed records không được dùng.

| Thống kê | Giá trị | Ý nghĩa |
|---|---:|---|
| Class counts | 104 / 154 / 58 | Imbalance vừa, không cực đoan |
| Unique `(G1,G2)` pairs | 89 | Input support nhỏ, nhiều học sinh có cùng cặp điểm |
| Pearson corr(G1,G3) | 0,8000 | G1 mạnh |
| Pearson corr(G2,G3) | 0,9048 | G2 rất mạnh và sát outcome |
| Spearman corr(G1,G3) | 0,8781 | Quan hệ thứ hạng rõ |
| Spearman corr(G2,G3) | 0,9567 | Ordinal structure rất mạnh |
| G3−G2 mean / median | −0,313 / 0 | Zero-residual là prior hợp lý |
| Residual population SD | 1,987 | Có tail đáng kể dù trung tâm nhỏ |
| Absolute residual mean | 0,959 | G2 gần G3 theo raw score |
| Residual bằng 0 | 49,37% | Gần nửa số mẫu G3=G2 |
| Residual trong ±1 | 87,66% | Headroom residual nhỏ cho đa số |
| Residual min / max | −10 / +3 | Tail bất đối xứng, Huber hợp lý hơn MSE nếu thử |

Deterministic rule “bin G2 theo cùng target thresholds” đạt development OOF Macro-F1 `0,898836`, accuracy `0,892405`, ordinal MAE `0,107595`. Đây là baseline khó và bắt buộc phải có trong mọi comparison mới.

Điều này không phải leakage nếu prediction cutoff thực sự sau G2. Nó cho thấy bài toán gần với “dự báo final class từ điểm gần cuối” và mô hình phức tạp có headroom rất nhỏ.

## 4. Chẩn đoán estimator và training pipeline

### 4.1 Critical bug trong fair comparison

Đường lỗi đã được xác minh trực tiếp:

```text
student_search_space(fair_comparison=True)
  đặt constants:
    loss = weighted_ce
    class_weight_mode = none
    oversample_method = none
  → Optuna chỉ trả study.best_params
  → constants không phải suggested params nên bị mất
  → outer fit nhận config thiếu class_weight_mode
  → _criterion mặc định missing key thành balanced
```

Evidence:

- Constants: `src/model_selection.py:85-122`.
- Chỉ trả `study.best_params`: `src/model_selection.py:404-405`.
- Default `balanced`: `src/model_selection.py:139-146`.
- Outer nhận params thiếu: `scripts/run_fair_model_comparison.py:254-269`.
- Artifact protocol tuyên bố `none for every model`.

Inner trials của CNN dùng unweighted CE, còn outer evaluation dùng weighted CE. Như vậy search estimator khác evaluation estimator và vi phạm chính protocol của run. Hai hàng CNN-LSTM/CNN-BiLSTM, cũng như paired ML-vs-DL conclusion, phải đánh dấu **invalid under intended protocol**.

Sửa tối thiểu sau approval:

1. Search trả `resolved_params = constants ∪ suggested_params`.
2. Missing `class_weight_mode`, `loss` hoặc `oversample_method` phải raise; không có default nguy hiểm.
3. Serialize resolved config và hash.
4. Contract test inner/outer/final tạo cùng criterion/resampling.
5. Rerun affected comparisons; không vá số cũ.

### 4.2 Model selection và final training không cùng estimator

Current selection path:

1. Split fold-train thành early train/validation.
2. Fit preprocessing trên early train.
3. Chọn epoch bằng early stopping.
4. Refit preprocessing và model trên toàn fold-train trong `selected_epoch` fixed epochs.

Current final path trong `scripts/run_pipeline.py:404-489`:

1. Split development train pool 85/15 cho từng seed.
2. Fit preprocessing/model trên 85%.
3. Early stop bằng 15%.
4. Lưu chính model đó, không refit toàn development.

Outer nested score của current model-selection code không ước lượng đúng estimator cuối sẽ deploy. Đây là **High training-estimator issue**, không phải chi tiết implementation nhỏ.

### 4.3 Scheduler, SWA và refit semantics

`train_model` dùng `ReduceLROnPlateau`, early stopping và SWA (`src/train_pipeline.py:122-175`). `selected_epoch` được rút từ base validation history. `train_fixed_epochs` (`:178-190`) lại chạy fixed initial learning rate, không scheduler và không SWA.

Một one-fold smoke trên outer fold 0, seed 42, frozen params xác nhận:

- initial LR: khoảng 0,0046677;
- scheduler đã giảm LR 5 lần, final LR khoảng 0,0001459;
- selected epoch: 17;
- early phase chạy đến khoảng epoch 30, không hit cap 40;
- refit chạy 17 epochs ở fixed initial LR;
- early train 214 rows, batch 32, chỉ 192 rows được dùng mỗi epoch vì `drop_last=True`;
- full refit 252 rows, chỉ 224 rows được dùng mỗi epoch.

Macro-F1 one-fold của smoke không được dùng để ranking. Giá trị của diagnostic là chứng minh training dynamics được dùng để chọn epoch không được replay trong refit.

SWA còn được tính nhưng không phải estimator được full refit; model có BatchNorm nhưng không thấy `update_bn`. Nên chọn một contract rõ: hoặc bỏ SWA khỏi selection, hoặc định nghĩa/replay đầy đủ cả scheduler và SWA. Không giữ “dead complexity”.

### 4.4 `drop_last`

Với batch 32 và tập rất nhỏ, `drop_last=True` bỏ 22–28 records mỗi epoch trong smoke. Shuffle khiến record bị bỏ thay đổi theo epoch, nhưng effective sample size thấp hơn và estimator phụ thuộc batch partition. Historical sanity cho thấy `drop_last=False` đôi khi cải thiện/stabilize, nhưng đó chưa phải causal evidence. Đây là correction rủi ro thấp cần paired ablation.

### 4.5 Provenance

- Fair protocol ghi commit không chứa fair runner.
- Frozen selection manifest ghi commit trước các config/code fields thực tế trong artifact.
- Latest final bundle checksum đúng 32/32 file nhưng `checkpoints` rỗng.

Do đó không có headline artifact nào hiện vừa đạt estimator parity vừa bind exact code + checkpoint. Cần rerun sau correction thay vì diễn giải thêm số cũ.

## 5. Đánh giá các giả thuyết A–G

### Tổng hợp trạng thái

| Giả thuyết | Trạng thái | Kết luận |
|---|---|---|
| A — Training pipeline inconsistency | **Confirmed** | Có fair loss mismatch, selection/final mismatch, scheduler/refit mismatch, drop_last và SWA/BN gap |
| B — Architecture quá lớn/không phù hợp | **Strong, chưa chứng minh nhân quả** | N nhỏ, L=2, kernel 1, 13k params; cần matched ablation |
| C — Ordinal structure bị bỏ phí | **Strong rationale, unverified model effect** | Labels có thứ tự, Spearman cao; prototype chưa active |
| D — Residual quanh G2 | **Plausible prior, simple form bị diagnostic bác bỏ** | Residual nhỏ nhưng fixed Ridge/Huber không thắng G2 rule về Macro-F1 |
| E — Multitask classification–regression | **Plausible but unverified** | Có thể regularize, nhưng pipeline target/loss chưa tích hợp và có gradient conflict risk |
| F — Context branch | **Blocked under current contract** | Timing unknown; thêm ngay tạo scientific/leakage risk |
| G — Calibration/decision rule | **Plausible but unverified** | Active argmax/no calibration; phải fit inner-OOF và kiểm stability |

### 5.1 A — Training inconsistency

Được xác nhận và là ưu tiên số 0. Không được chạy architecture search lớn trước khi cùng một config sinh ra cùng criterion, preprocessing/resampling, epoch/refit policy và final estimator ở mọi tầng.

`epochs_ran`, `selected_epoch`, `refit_epochs`, `hit_epoch_cap` phải là bốn field riêng trong artifact. Current historical evidence không luôn phân biệt được chúng.

### 5.2 B — Capacity và ablation

Historical fixed ablations có hướng:

- CNN-only khoảng 0,8004;
- BiLSTM-only khoảng 0,2184;
- CNN–BiLSTM không imbalance khoảng 0,8422;
- 11-seed ensemble khoảng 0,8505.

Nhưng parameter budgets/configs không matched; repo cũng cảnh báo không suy luận causal từ bảng này. Không thể kết luận “CNN cần thiết” hoặc “BiLSTM vô dụng”. Cần các control sau trên cùng feature/folds/search budget:

- deterministic G2 rule;
- tiny nominal MLP;
- tiny ordered MLP;
- CNN-only;
- BiLSTM-only;
- corrected compact CNN–BiLSTM;
- tối đa một CNN-LSTM comparator nếu compute cho phép.

Model sizes nên matched theo một dải nhỏ, không cho một family vượt xa chỉ vì search space rộng hơn.

### 5.3 C — Ordinal learning

Low < Medium < High là cấu trúc thật của target. Nominal softmax phạt Low→Medium và Low→High như nhau, trong khi hậu quả ordinal khác nhau. `src/models/ordinal_v3.py` có ordered/ordinal primitives, nhưng không được export/call/test; đây là prototype, không phải result.

Candidate phù hợp nhất là tiny MLP với ordered thresholds/CORAL hoặc cumulative logits. Metrics bắt buộc:

- Macro-F1 làm primary để giữ mục tiêu hiện tại;
- QWK;
- ordinal MAE trên class index;
- one-step/two-step error;
- per-class recall;
- boundary error ở raw G3 9/10/14/15.

Monotonic thresholds phải được đảm bảo bằng parameterization và test, không kiểm bằng mắt.

### 5.4 D — Residual quanh G2

Ý tưởng `G3_hat = G2 + r(G1,G2)` hợp lý vì residual trung tâm gần 0. Nhưng diagnostic development-only, immutable five-fold, không tuning cho thấy:

| Model residual | Macro-F1 | Accuracy | Raw G3 MAE | RMSE |
|---|---:|---:|---:|---:|
| Zero residual / G2 rule | 0,898836 | 0,892405 | 0,95886 | 2,01183 |
| Ridge residual, α=1 | 0,873735 | 0,87025 | 1,14326 | 1,92802 |
| Huber residual mặc định | 0,866684 | 0,86392 | 1,00016 | 1,99528 |

Ridge cải thiện RMSE nhẹ nhưng làm classification và MAE xấu hơn; Huber cũng không thắng. Vì vậy bác bỏ giả thuyết “residual đơn giản tự động dễ hơn và tốt hơn”. Nếu thử sau này, chỉ dùng gated/zero-init residual + Huber như conditional branch, phải thắng G2 rule trên primary và ordinal guardrails. Không dùng G2 copy như một “deep improvement”.

### 5.5 E — Multitask

Joint classification + raw G3 regression có thể chia sẻ signal ordinal, nhưng rủi ro:

- regression tối ưu RMSE nhưng làm Macro-F1 giảm, như residual diagnostic gợi ý;
- target scale/lambda có thể chi phối gradient;
- raw target phải được truyền riêng, không lọt vào feature frame;
- scaler và loss weighting phải fit/tune inner-fold only.

Thứ tự đúng: ordinal single-task trước; chỉ thêm Huber auxiliary head khi ordered model vượt nominal control hoặc cho tín hiệu nhất quán. Dùng pre-registered λ grid nhỏ; theo dõi cosine/gradient norm nếu cần, không mở search vô hạn.

### 5.6 F — Context fusion

Không admissible hiện tại. Questionnaire timing là unknown, `unknown = forbidden`. `absences` đặc biệt có thể là tích lũy đến cuối kỳ; support/motivation fields không có snapshot timestamp. Thêm context sẽ:

- làm thay đổi prediction cutoff;
- có thể leakage theo thời gian;
- làm DL có feature advantage nếu ML vẫn chỉ G1/G2;
- giảm khả năng tái lập/triển khai.

Chỉ mở track này sau data contract mới có `available_at`, `snapshot_at`, reference window, freshness, source và derived-input DAG. Khi mở, ML và DL phải dùng đúng cùng context.

### 5.7 G — Calibration, thresholds và abstention

Active frozen policy là argmax, không calibration. Brier/ECE có trong một số runner nhưng chưa có nested multi-seed decision study. Hướng hợp lý:

1. Fit temperature hoặc ordinal cutpoints chỉ trên inner-OOF predictions.
2. Apply một lần cho outer validation.
3. Báo NLL, Brier, ECE, reliability, Macro-F1 và decision stability.
4. Ưu tiên selective prediction/abstention cho case uncertainty cao thay vì “squeezing” thresholds để lấy vài điểm Macro-F1.
5. Không chọn threshold bằng observed 79.

## 6. Imbalance handling

Development distribution 104/154/58 là imbalance vừa. Current evidence không cho thấy SMOTE/class weighting cải thiện ổn định; fair bug làm một phần evidence không dùng được.

Khuyến nghị:

- Default chính: unweighted CE/ordinal loss, không oversampling.
- Không dùng SMOTE/ADASYN làm default vì nội suy hai điểm có thể tạo trajectory điểm tổng hợp khó giải thích.
- Nếu High recall dưới guardrail, ablate **một** lựa chọn tại một thời điểm: class weight, focal loss hoặc random oversampling train-only.
- Không kết hợp class weight + oversampling nếu chưa có lý do và ablation riêng.
- Không chọn imbalance method theo một fold/seed đẹp.

## 7. Kiến trúc có cơ sở hơn

Candidate set cân bằng được đề xuất:

1. **G2 deterministic rule** — reference bắt buộc, 0 fit.
2. **RF và SVM** — strong ML baselines trên đúng G1/G2.
3. **Tiny nominal MLP** — kiểm tra liệu recurrent/convolution có cần thiết hay không.
4. **Tiny ordered MLP** — candidate neural chính nhờ ordinal inductive bias phù hợp hơn sequence inductive bias.
5. **Corrected compact CNN–BiLSTM** — thesis/control bắt buộc, thu nhỏ search space và estimator đúng.
6. **Matched CNN-only và BiLSTM-only** — ablation, không nhất thiết full search độc lập.
7. **Conditional gated residual/Huber auxiliary** — chỉ mở khi ordered candidate qua gate.

Champion phải được chọn theo pre-registered development nested-CV, không theo yêu cầu “phải là CNN”. Nếu ML thắng, kết quả khoa học đúng là ML champion + CNN–BiLSTM research/control; không được đổi bài toán hoặc feature để ép kết quả.

## 8. Trả lời độc lập 12 câu hỏi

1. **Repository đang giải quyết gì?** Late-stage 3-class prediction của G3 từ G1/G2, rồi sinh advice rule-based từ prediction và một số profile fields.
2. **Có khớp đề cương không?** Khớp một phần. Có CNN/BiLSTM và recommendation prototype, nhưng dữ liệu không hỗ trợ claim nhiều học kỳ và recommender chưa có vòng đời.
3. **Dữ liệu đủ cho CNN–BiLSTM không?** Đủ để thử nghiệm một architecture nhỏ trên two-step ordered input; không đủ để biện minh long-sequence deep model hoặc kết luận ưu thế.
4. **Có nên giữ sequence-only không?** Có, làm primary fair track vì contract rõ và triển khai được. Nhưng không mặc định sequence architecture là champion; tiny vector models phải cạnh tranh ngang hàng.
5. **Có nên thêm context branch không?** Chưa. Timing contract hiện cấm. Chỉ mở thành track riêng sau khi có dữ liệu timestamp/freshness hợp lệ.
6. **Có nên dùng ordinal/regression supervision không?** Ordinal: có, là candidate ưu tiên. Regression: chỉ auxiliary Huber sau ordinal gate, không dùng MSE mặc định.
7. **Có nên học residual quanh G2 không?** Không làm centerpiece. Simple residual đã giảm Macro-F1 trong diagnostic; chỉ thử conditional gated/zero-init nếu budget còn.
8. **Có kiến trúc tốt hơn ý tưởng ban đầu không?** Tiny ordered MLP là inductive bias phù hợp hơn với N=316 và L=2; corrected compact CNN–BiLSTM vẫn là control. Champion có thể là RF/SVM/G2 rule.
9. **Có cần thêm dữ liệu/thay đổi chuỗi không?** Có nếu muốn claim temporal/early-warning thực sự: cần student/entity ID, nhiều assessment với timestamp, cutoff-aware context và domain phù hợp. Không ghép dataset khác một cách cơ học.
10. **Recommendation hiện đủ gọi “lộ trình học” chưa?** Chưa; chỉ có snapshot advice + 3 template, không goal/follow-up/revision.
11. **Recommendation architecture phù hợp nhất?** Expert-guided, versioned decision support: calibrated risk → valid/modifiable signals → goals/actions → advisor approval → follow-up/revision → outcomes.
12. **Phương án cân bằng tốt nhất?** Sửa correctness/provenance; strict G1/G2 candidate pool gồm strong ML, tiny nominal/ordinal, corrected CNN controls; recommendation hardening + human lifecycle; external data mới cho final confirmation.

## 9. Kết luận chẩn đoán

Nguyên nhân CNN–BiLSTM chưa chứng minh vượt ML không phải một yếu tố đơn lẻ. Phần **đã xác nhận** là protocol/estimator inconsistency. Phần **có evidence mạnh** là inductive bias không khớp sequence dài 2 và capacity lớn so với N. Phần **hợp lý nhưng chưa kiểm chứng** là ordinal/calibration có thể giúp. Phần **đã bị diagnostic làm yếu đi** là residual tuyến tính/Huber đơn giản. Context fusion hiện bị data governance chặn.

Vì vậy, hành động đúng không phải “tune CNN mạnh hơn ngay”, mà là tạo một sân chơi estimator-correct, cùng feature contract, có G2 rule và matched small models. Chỉ evidence mới từ sân chơi đó được phép quyết định architecture.
