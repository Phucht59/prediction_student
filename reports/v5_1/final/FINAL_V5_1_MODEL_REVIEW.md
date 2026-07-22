# CNN–BiLSTM V5.1 — Final Model Review

Ngày khóa evidence: 2026-07-19
Branch: `codex/project-v5-1-cnn-bilstm-performance`

## Kết luận điều hành

V5.1 hoàn tất đầy đủ ba study với outer-fold evaluation, toàn bộ fixed seeds, checkpoint replay, paired bootstrap và ablation đã đăng ký. Hai UCI study cải thiện rõ so với V5 và vượt target định hướng: `student-mat` đạt Macro-F1 0.901460 và `student-por` đạt 0.862259. Tuy vậy, khoảng tin cậy paired bootstrap vẫn giao 0 khi so với Decision Tree và Random Forest mạnh nhất, nên kết luận hợp lệ là practical tie/chưa chắc chắn, không phải deep learning vượt trội.

OULAD đạt Macro-F1 0.827422, không đạt target 0.832 và thấp nhẹ hơn cả V5 (0.828003) lẫn XGBoost (0.828381). Chênh lệch đều nằm trong khoảng bất định. Hybrid vượt CNN-only chắc chắn hơn, nhưng chỉ practical-tie với BiLSTM-only. Kết quả âm này được giữ nguyên; không search thêm sau khi xem outer result.

Future OULAD vẫn `LOCKED_NOT_EXECUTED`. Không dùng outer-test để tuning. Validator V5.1 PASS toàn bộ correctness contract; directional target không phải correctness gate.

## Headline results

| Dataset | V5.1 pooled outer-OOF Macro-F1 | Target | Target met | Primary comparator | Comparator Macro-F1 | Delta V5.1 − comparator | Bootstrap 95% CI | Verdict |
| --- | ---: | ---: | --- | --- | ---: | ---: | --- | --- |
| `student-mat` | 0.901460 | 0.890 | Yes | Decision Tree | 0.906654 | -0.005194 | [-0.022199, 0.012311] | practical tie / uncertain |
| `student-por` | 0.862259 | 0.860 | Yes | Random Forest | 0.869244 | -0.006985 | [-0.029431, 0.014580] | practical tie / uncertain |
| OULAD F2 | 0.827422 | 0.832 | No | XGBoost V4/V5 | 0.828381 | -0.000959 | [-0.004713, 0.002702] | practical tie / uncertain |

So với V5, V5.1 tăng +0.021543 trên `student-mat`, +0.013107 trên `student-por`, và giảm -0.000581 trên OULAD. OULAD V5.1 so với V5 có CI 95% [-0.003814, 0.002668], do đó không có bằng chứng về thay đổi thật.

## Dataset-specific interpretation

### student-mat

Candidate cuối là `cnn_bilstm_v5_1_transfer_selected_ensemble`. Trên 395 outer-OOF records, accuracy 0.891139, balanced accuracy 0.902089, Macro-F1 0.901460 và Macro PR-AUC 0.944184. Recall theo lớp Low/Medium/High lần lượt là 0.915385, 0.859375 và 0.931507.

Transfer-selected hybrid vượt CNN-only +0.030668 với CI 95% [0.005483, 0.058189], và vượt BiLSTM-only +0.061742 với CI [0.028426, 0.095981]. Điều này hỗ trợ giá trị của kiến trúc kết hợp trong study này, nhưng không chứng minh nó vượt Decision Tree.

### student-por

Candidate cuối là `cnn_bilstm_v5_1_ensemble`. Trên 649 records, accuracy 0.889060, balanced accuracy 0.867576, Macro-F1 0.862259 và Macro PR-AUC 0.914679. Recall Low/Medium/High lần lượt 0.780000, 0.906699 và 0.916031.

Hybrid vượt CNN-only +0.015451, CI 95% [0.002505, 0.029520], và vượt BiLSTM-only +0.077980, CI [0.051832, 0.105795]. So với Random Forest, CI giao 0 nên không được claim superiority.

### OULAD F2_MIDDLE

Candidate cuối là `cnn_bilstm_full_ensemble`. Trên 15.378 records thuộc 14.687 student groups, Macro-F1 0.827422, balanced accuracy 0.821232, at-risk precision 0.836795, at-risk recall 0.737496, PR-AUC 0.893550 và ECE 0.015673. Threshold được chọn riêng trong inner training theo từng outer fold.

Hybrid vượt CNN-only +0.007024, CI 95% [0.003327, 0.010694]. So với BiLSTM-only, delta chỉ +0.000160, CI [-0.003175, 0.003359]. Vì vậy dữ liệu ủng hộ việc BiLSTM mang phần lớn tín hiệu OULAD, còn incremental gain của CNN chưa được chứng minh chắc chắn.

## OULAD search và protocol amendment

Amendment giảm ngân sách được ghi trước khi xem bất kỳ outer result nào. Architecture screening chỉ dùng outer-training fold 0. Optuna DB/study name và mọi COMPLETE/PRUNED row được giữ nguyên.

- 31 unique configurations đã hoàn tất đủ ba inner folds được tái sử dụng; trong đó 19 row từng bị MedianPruner gắn PRUNED muộn được phục hồi vào ranking vì đã có đủ evidence.
- Không huấn luyện lại các cấu hình hợp lệ và không xóa trial lịch sử.
- Top-2 được xác nhận trên ba registered screening seeds; architecture trial 12 thắng với confirmed inner Macro-F1 0.827489.
- Masked-week pretraining không mở vì architecture không đạt gate 0.8305.
- Augmentation/loss candidates không được giữ vì không đạt improvement cùng-fold tối thiểu đã đăng ký.
- Focused search dừng ở 16 total trials: 13 COMPLETE, 3 PRUNED. Sau multi-seed confirmation, trial 12 được khóa với inner Macro-F1 0.827241, 8 fixed epochs, không augmentation và không pretraining.
- Duy nhất cấu hình thắng được chạy full 3 outer folds × 5 fixed seeds. CNN-only và BiLSTM-only chỉ chạy theo ablation contract, không Optuna riêng.

Median pruning bị vô hiệu ở architecture stage vì pruner cũ chỉ ra quyết định sau khi đã trả toàn bộ chi phí ba folds và có thể loại một cấu hình có mean tốt. Đây là sửa accounting/evidence reuse, không thay đổi metric hay selection data.

## Technical review of the selected models

### Stability, complexity và runtime

| Dataset | Mean seed Macro-F1 | Seed SD | Seed min–max | Selected parameter count | Tổng runtime selected checkpoints |
| --- | ---: | ---: | --- | --- | ---: |
| `student-mat` | 0.897697 | 0.005370 | 0.888689–0.903452 | 30.724–86.164 theo outer fold | 245,2 s |
| `student-por` | 0.857453 | 0.005733 | 0.849064–0.864793 | 31.494–163.606 theo outer fold | 239,2 s |
| OULAD | 0.826016 | 0.000668 | 0.825163–0.827046 | 99.443 | 4.277,1 s |

Pooled five-seed ensemble khác mean của five seed metrics vì ensemble lấy trung bình probability theo record trước khi tính Macro-F1. OULAD ổn định nhất theo seed nhưng không đạt performance target.

### Context, fusion và multi-task trên UCI

Context branch thực sự đi vào mọi final UCI checkpoint, nhưng protocol không đăng ký final context-off ablation; do đó không thể gán một delta causal riêng cho context. Final fold configs không ép một fusion duy nhất: Math chọn 1 FiLM-residual, 2 concatenation và 2 gated folds; Portuguese chọn 1 FiLM-residual, 2 concatenation và 2 gated folds. Với các final gated checkpoints có diagnostic hợp lệ, Portuguese gate mean nằm trong [0.4543, 0.6498] và không checkpoint nào collapse. Transfer-selected Math checkpoints không xuất gate diagnostic đồng nhất, nên không được suy diễn gate effect cho candidate đó.

Multi-task cũng được chọn theo outer-training fold, không phải áp đặt toàn cục. Math có 2/5 folds classification-only, 2/5 classification + Huber regression và 1/5 thêm ordinal auxiliary; Portuguese có 2/5 classification-only và 3/5 classification + regression + ordinal. Vì không có controlled final head-off comparison trên cùng mọi fold/seed, evidence chỉ cho thấy objective phụ có thể được inner selection giữ lại, không chứng minh regression head tự nó cải thiện classification. Không có dấu hiệu loss lấn át làm class collapse trong final predictions.

### Transfer learning

Trên cả 5 Math outer-training partitions, screening chọn `shared_trunk_subject_specific_heads`. Inner screening score của phương án này nằm 0.7685–0.8423, cao hơn standalone 0.4864–0.5857; phương án pretrain/freeze/unfreeze riêng chỉ đạt 0.2686–0.2921 và bị loại như negative transfer. Final shared-trunk candidate đạt 0.901460, cao hơn point estimate 0.861380 của standalone V5.1 hybrid; phép paired bootstrap riêng cho cặp transfer-vs-standalone không được đăng ký nên không gán superiority verdict cho chênh lệch này. Vì selection chỉ dùng inner data và quasi-identity overlap control, transfer được giữ cho thesis model Math; nó không biến Portuguese thành external validation độc lập.

### OULAD representation, pretraining, augmentation và fusion

Selected OULAD model dùng residual multi-kernel CNN kernels [2, 3], BiLSTM hidden 64, masked mean/max pooling, compact aggregate/static branches và gated residual fusion. Masked attention không được chọn; `attention_padding_max = 0.0` trên checkpoints vẫn xác nhận padding contract. Model có 99.443 tham số, dưới xa giới hạn 1,5 triệu.

Masked pretraining không chạy vì confirmed architecture 0.827489 thấp hơn gate 0.8305. Augmentation screening chọn none: event thinning 0.826669, short-span masking 0.826737 và channel-group dropout 0.826549 đều thấp hơn none 0.827859. Loss screening chọn standard BCE 0.827859; weighted BCE 0.827117 và focal 0.827448 không đạt improvement threshold.

Không final controlled ablation nào cô lập riêng multi-kernel so với CNN cũ, đảo temporal order, hoặc tắt compact aggregate; vì vậy không được claim causal gain cho ba thành phần này. Full hybrid vượt CNN-only nhưng practical-tie với BiLSTM-only, nên BiLSTM còn đóng góp rõ trong kiến trúc, còn incremental CNN value trên OULAD chưa được chứng minh. Gated fusion không collapse: gate mean của 15 selected full checkpoints nằm [0.4609, 0.6487], saturation fraction bằng 0 và branch norms đều hữu hạn.

### Model-role decision và Future OULAD

V5.1 hybrid là **thesis research model** cho cả ba dataset vì đây là đối tượng nghiên cứu đã khóa và đánh giá đầy đủ. Operational model vẫn là Decision Tree cho Math, Random Forest cho Portuguese và XGBoost cho OULAD vì chúng có point estimate cao hơn, đơn giản hơn và không có evidence buộc phải thay thế. Tổng thể V5.1 phù hợp **Tier C — cải thiện nhưng chưa vượt ML**. Future OULAD phải `KEEP_LOCKED`; không có cơ sở khoa học để mở chỉ vì target 0.832 chưa đạt.

## Fairness và leakage controls

- Target, cohort, split manifest, grouped student isolation, cutoff, primary metric và fixed seeds giữ nguyên.
- Preprocessing chỉ fit trên training partition.
- Architecture và focused search chỉ nhìn outer-training fold 0; outer test không tham gia tuning.
- Final evaluation dùng đủ fixed seeds đã đăng ký, không chọn best seed.
- OOF probabilities được record-align trước ensemble và paired comparison.
- OULAD bootstrap resample theo global student group; UCI resample theo record/student.
- Future OULAD chưa được mở.

## Reproducibility và artefact integrity

Ba model registry ghi candidate, protocol fingerprint, seeds, checkpoints và SHA-256. Validator replay 25 UCI Math checkpoints, 25 UCI Portuguese checkpoints và 45 OULAD checkpoints với probability contract, record coverage, fold/seed coverage và checksum đều PASS. Release summary, paired bootstrap 5.000 replicates và consolidated registry đều COMPLETE.

Lệnh read-only chính:

```powershell
python project.py v5-1 status
python project.py v5-1 validate
python project.py v5-1 report
```

## Database audit

Audit read-only trên PostgreSQL hiện hữu PASS và xác nhận transaction read-only; không có write. Destructive migration/permission integration được ghi `SKIP_NO_DISPOSABLE_DSN` vì execution environment không chứa DSN có tên database đánh dấu disposable. Đây là transparent skip, không phải PASS giả và không cho phép suy rộng thành database integration đã được kiểm thử đầy đủ.

## Validation status

- Strict V5 validator: PASS.
- Strict V5.1 validator: PASS, gồm 30 checks về completion, Future lock, seed/fold coverage, checkpoint replay, checksum, bootstrap và V5 immutability.
- Full pytest suite: 221 PASS, 7 SKIP sau khi namespace contract được cập nhật cho V5.1.
- Historical frozen OULAD release check: 18/19 PASS trên máy này. Check duy nhất không thể tái xác nhận là `backup_gate` vì bốn PostgreSQL backup file nằm ở external `backup_root` lịch sử không còn trên máy. Các artifact checksum, fairness, migration evidence, test evidence và credential-redaction checks của bundle này vẫn PASS; V4/V5 evidence không bị sửa để che external dependency đó.

## Claims được phép và không được phép

Được phép nói V5.1 cải thiện mạnh hai UCI point estimates so với V5, vượt target định hướng UCI, và hybrid vượt CNN-only theo paired bootstrap trên cả ba dataset. Được phép nói OULAD là practical tie với XGBoost/V5 và full hybrid practical-tie với BiLSTM-only.

Không được nói CNN–BiLSTM chắc chắn vượt ML, OULAD đạt target, Future OULAD đã được kiểm thử, external generalization đã được chứng minh, hoặc recommendation effectiveness đã được thiết lập.

## Authoritative evidence

- `artifacts/v5_1/final/summary.json`
- `artifacts/v5_1/final/model_registry.json`
- `artifacts/v5_1/*/model_registry.json`
- `reports/v5_1/final/paired_bootstrap.json`
- `reports/v5_1/final/database_audit.json`
- `reports/v5_1/final/validation_report.json`
- `reports/v5_1/PROTOCOL_AMENDMENTS.md`
