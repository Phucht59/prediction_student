# 08 — Tóm tắt để phê duyệt

## Quyết định điều hành đề xuất

Phê duyệt **Strategy B — Balanced**: sửa correctness/provenance trước; giữ strict G1/G2; so G2/RF/SVM với tiny nominal MLP, tiny ordered MLP và corrected compact CNN–BiLSTM; residual/imbalance là conditional; context đóng; recommendation nâng lên expert-guided lifecycle. Không model mới hay full experiment nào đã được triển khai trong giai đoạn audit.

## A. Dự án hiện đã làm được gì

### Data pipeline

- Ingest `student-mat` 395 dòng, source row identity, target separation và fold-local preprocessing.
- Official loader/preprocessor loại target khỏi feature path.
- Data profile sạch tại thời điểm audit: 0 null, 0 duplicate, 0 checked domain violation.
- Feature scenarios và availability contract đã tồn tại; active track là late-stage G1/G2.

### Database

- Versioned dataset, content hash, source records, separate target table.
- Run/split/prediction/recommendation lineage và append-only triggers.
- Live raw/DB/V2 hash khớp; 395 source + 395 target, không orphan/mismatch.
- Multi-policy recommendation versioning hoạt động.

### Prediction models

- Active CNN–BiLSTM sequence-only, 13.059 parameters, input G1/G2.
- ML baselines và fair-comparison runner đã có.
- Nested-CV/refit framework hiện tại có preprocessing/resampling train-only.
- Prototype ordinal/multitask primitives tồn tại nhưng chưa active.

### Evaluation

- Immutable development outer folds: 316 records, 5 folds, 1.580 assignment rows.
- Macro-F1, accuracy, Brier/ECE và nhiều evaluation utilities.
- Artifact/evidence validation, DB persistence và recommendation structural metrics.

### Evidence

- Latest final bundle có 32/32 checksum khớp.
- Historical model-selection, ablation và fair-comparison artifacts phong phú.
- Protocol V2 tự ghi nhận contamination của 79 held-out records, giúp sửa claim boundary trung thực.

### Recommendation

- Deterministic policy v3 với 9 risk signals/actions, risk/confidence bands, explanation và 3 template 4 tuần.
- Recommendation snapshots được version hóa và lưu PostgreSQL.
- Có structural self-check và expert-review case template.

### Tests

- `pytest -q`: 87 passed, 5 skipped.
- 92 tests collected; recommendation subset 12 passed.

### Documentation

- README, PROJECT, scientific protocols, report-context và limitations/ethics documents đã mô tả nhiều phần của hệ thống.
- Một số docs đã tự giới hạn recommendation là rule-based/structural-only và expert evaluation pending.

## B. Dự án còn thiếu gì

### Thiếu về khoa học

- Không còn unseen test: 79 records đã được xem qua nhiều models.
- Fair deep comparison sai inner/outer estimator do constant config bị mất.
- Selection estimator khác final deployment estimator.
- Scheduler/SWA epoch-selection không khớp fixed refit.
- Một số headline artifacts không bind exact commit/config/checkpoint.

### Thiếu về mô hình

- Chưa có matched-capacity evidence cho CNN-only/BiLSTM-only/CNN–BiLSTM.
- Chưa có active tiny ordinal model.
- Calibration/abstention chưa nested, multi-seed.
- Seed stability của fair deep candidates chưa đủ.
- G2 rule chưa nằm trong cùng fair runner dù là baseline mạnh nhất về logic.

### Thiếu về dữ liệu

- Sequence chỉ hai assessment, không phải nhiều học kỳ.
- Không entity/student ID hoặc timestamped event grain.
- Context capture time/freshness chưa biết.
- Chưa có external/prospective dataset chưa quan sát.

### Thiếu về recommendation

- Missing/invalid semantics, confidence validation và feature governance chưa safe.
- Không policy registry/hash/approver.
- Không goal/action objects, advisor decision, follow-up, adherence, revision hay outcome.
- Không expert ratings; không evidence usefulness/effectiveness.
- Không dữ liệu để học recommender hoặc causal policy.

### Thiếu về evaluation

- Inner/early-stop record ledgers.
- QWK, ordinal MAE, one-/two-step trong fair output.
- Paired multi-seed comparisons và calibrated uncertainty.
- Expert agreement, shadow pilot và prospective evaluation.

### Thiếu về engineering

- DB target/prediction constraints còn yếu; target conflict có thể bị bỏ qua.
- Portable identity, migration ledger và stale-run recovery.
- Production prediction schema tách outcome.
- Final checkpoint checksums và exact source-tree provenance.

### Thiếu so với đề cương

- “Diễn biến qua nhiều học kỳ” và “early warning” chưa được data/active pipeline hỗ trợ.
- “Lộ trình học” hiện mới là advice snapshot/template, chưa có lifecycle.
- Chưa chứng minh CNN–BiLSTM cạnh tranh/vượt ML dưới protocol hợp lệ.

## C. Nguyên nhân CNN–BiLSTM chưa vượt ML

### Confirmed

1. Fair benchmark inner dùng unweighted CE nhưng outer DL rơi vào balanced class weight; output DL invalid theo intended protocol.
2. Model-selection và final-training estimators khác nhau.
3. Epoch chọn dưới scheduler/SWA nhưng full refit dùng fixed LR/no SWA.
4. Active input chỉ có G1/G2, sequence length 2; kernel 1 chỉ pointwise.
5. G2 là baseline cực mạnh: development G2 rule Macro-F1 0,898836.
6. Final artifacts/provenance chưa bind exact estimator/checkpoint đầy đủ.

### Strong evidence

- 13.059 parameters là lớn so với 316 development records và inner training subsets.
- `drop_last` làm mất tỷ lệ mẫu đáng kể mỗi epoch trong dataset nhỏ.
- Ordinal structure mạnh nhưng nominal softmax không tận dụng trực tiếp.
- Current deep model có variance/collapse risk; fair run chỉ một seed mỗi outer fold.

### Plausible but unverified

- Tiny ordered model có thể tốt/ổn định hơn CNN–BiLSTM.
- Inner-only calibration/abstention có thể cải thiện reliability và safety.
- Gated residual + Huber auxiliary có thể giúp một subset/tail sau khi ordinal model ổn định.
- Compact recurrent/convolution ablations có thể giữ được signal với variance thấp hơn.

### Rejected hoặc downgraded hypotheses

- “Residual đơn giản chắc chắn dễ hơn”: fixed Ridge/Huber residual đều giảm Macro-F1 so G2 rule trong diagnostic.
- “Thêm context ngay sẽ tốt hơn”: timing contract hiện cấm; leakage/fairness risk cao.
- “SMOTE/class weight chắc chắn cần vì imbalance”: imbalance chỉ vừa và historical evidence không cho gain ổn định; fair bug còn làm evidence mơ hồ.
- “CNN–BiLSTM phải là final champion để đúng đề cương”: không đúng khoa học; nó phải là candidate/control, champion do evidence quyết định.

## D. Chiến lược được đề xuất

### Prediction architecture

- References: G2 rule, RF, SVM.
- Neural: corrected compact CNN–BiLSTM nominal; tiny nominal MLP; tiny ordered MLP/CORAL.
- Fixed ablations: CNN-only, BiLSTM-only, CNN-LSTM/simplified recurrent.
- Residual/multitask chỉ conditional sau ordinal gate.

### Feature tracks

- Primary: late-stage G1/G2.
- Early warning G1-only: future separate track, chưa active.
- Context: đóng đến khi có capture-time contract.

### Loss và imbalance

- Nominal: unweighted CE.
- Ordered: cumulative/ordered loss với monotonic thresholds.
- Conditional auxiliary: Huber, fold-local scaling, inner-selected λ.
- Imbalance default `none`; chỉ mở một method nếu minority guardrail thất bại; không SMOTE/ADASYN default.

### Protocol

- Correctness fixes + tests trước fits.
- Nested 5 outer × 3 inner, 30 trials/family main study.
- 3 outer seeds cho neural selected configs, 5 new seeds ở final stability.
- Macro-F1 primary; ordinal/calibration/class-wise/stability/compute guardrails.
- Full provenance và paired OOF analysis.
- Không dùng 79 observed.

### Recommendation architecture

```text
calibrated prediction + abstention
→ governed observations/modifiability
→ expert-approved policy registry
→ structured goals/actions
→ explanation/non-causal scope
→ advisor approve/modify/reject
→ follow-up/adherence/revision
→ outcomes/safety evaluation
```

### Validation

Technical safety → expert review → shadow pilot → prospective evaluation → learned ranking chỉ khi có interaction/outcome data → causal only with causal design.

## E. Các quyết định cần phê duyệt

| Decision | Options | Recommended option | Reason | Risk nếu chọn khác |
|---:|---|---|---|---|
| 1 — Trạng thái 79 records | A. Vẫn gọi locked; B. Đổi thành observed | **B** | Protocol V2 xác nhận đã xem nhiều lần | A làm invalid final-test claim |
| 2 — Chiến lược tổng thể | A Conservative; B Balanced; C Ambitious | **B** | Tốt nhất giữa performance, validity, scope | A dễ bỏ lỡ model phù hợp; C scope/leakage cao |
| 3 — Primary feature track | G1/G2; hoặc context ngay | **G1/G2** | Contract rõ, cùng sân chơi, active pipeline hỗ trợ | Context timing unknown, comparison không fair |
| 4 — Context-fusion | Mở ngay; conditional; bỏ vĩnh viễn | **Conditional, hiện đóng** | Có tiềm năng nhưng cần snapshot contract | Mở ngay có temporal leakage/external validity risk |
| 5 — Estimator corrections | Optional theo score; mandatory | **Mandatory** | Defect không được giữ vì metric đẹp | Optional làm mọi comparison khó diễn giải |
| 6 — Evidence fair hiện tại | Dùng leaderboard; quarantine DL/cross-family | **Quarantine và rerun** | Inner/outer loss mismatch + provenance sai | Dùng tiếp dẫn tới conclusion không hợp lệ |
| 7 — Candidate pool | Chỉ CNN; ML+small ordinal+CNN controls | **Pool cân bằng** | N=316, L=2 cần simple/ordinal controls | Chỉ CNN tạo architecture bias |
| 8 — Vai trò CNN | Bắt buộc champion; bắt buộc candidate/control | **Candidate/control, không ép champion** | Đúng đề cương nhưng tôn trọng evidence | Ép champion đánh đổi scientific validity |
| 9 — Ordinal | Không; ordered candidate | **Có, tiny ordered MLP** | Labels có thứ tự, Spearman G2–G3 cao | Không thử có thể bỏ phí đúng inductive bias |
| 10 — Residual/multitask | Main mandatory; conditional; bỏ | **Conditional** | Simple residual diagnostic kém G2 rule | Main làm tăng compute/multiplicity không có signal |
| 11 — Imbalance | SMOTE+weights; none default + gate | **None default + conditional single method** | Imbalance vừa, evidence gain yếu | Double handling có thể tăng variance/synthetic artifacts |
| 12 — Hybrid ML–DL | Main ngay; conditional sau base | **Conditional sau G4** | Chỉ hợp lệ với inner-OOF stacking và clear complementarity | Mở ngay tăng overfit/multiplicity |
| 13 — Recommendation scope | Giữ snapshot; Level 0.5 lifecycle; learned/causal ngay | **Level 0.5** | Khả thi với data và có giá trị thực tế | Snapshot không đạt đề cương; learned/causal thiếu data |
| 14 — Search/seed budget | 10 trials/1 seed; 30 trials/3+5 seeds; >50 trials | **30 trials; 3 outer + 5 final seeds** | Cân bằng compute/stability; gần workload fair hiện có | Quá thấp thiếu stability; quá cao scope lớn |
| 15 — Phase F | Dùng lại 79; chỉ external/prospective mới | **Chỉ data mới** | 79 không còn unseen | Dùng 79 tạo confirmation bias |

Phê duyệt Strategy B nên bao gồm Decisions 1–15 như một gói. Nếu muốn thay một quyết định làm đổi population/features/primary metric/budget, cần cập nhật protocol trước implementation.

## F. Kết quả diagnostic đã chạy

| Command | Phạm vi | Kết quả | Ý nghĩa | Vì sao chưa phải full evidence |
|---|---|---|---|---|
| `git status --short; git branch --show-current; git rev-parse HEAD` | Repository | `main`, commit `6618fe4...`; README/PROJECT dirty sẵn | Đóng băng audit context | Không đánh giá model |
| `py -3.10 -m pytest -q` | Full test suite | 87 pass, 5 skip, 9,75s | Current tests xanh | Tests chưa bao phủ semantic estimator bugs; không phải model comparison |
| `py -3.10 scripts/materialize_scientific_protocol_v2.py --verify` | V2 artifacts | Immutable artifacts verified | Outer manifest/checksum hợp lệ | Không chạy model, không xác nhận performance |
| Inline `py -3.10 -` read-only DB profile | Source/target/run/split/schema aggregates | 395/395, hash/membership/targets khớp; 2 stale runs | DB integrity live tốt, gaps design rõ | Aggregate audit; không đọc/chọn metric 79 |
| Inline `py -3.10 -` raw/development profile | 395 raw + 316 development | No null/duplicate; class/stats/correlation/residual profile | Xác minh grain, G2 strength, headroom | Descriptive, không model selection |
| Inline five-fold fixed residual diagnostic | 316 development; Ridge/Huber, no tuning | G2 F1 0,898836; Ridge 0,873735; Huber 0,866684 | Downgrade simple residual | Fixed small diagnostic, chưa nested/tuned family |
| Inline one-outer-fold neural smoke | Fold 0, seed 42, frozen config | selected epoch 17; 5 LR reductions; drop 22/28 rows; refit fixed LR | Xác nhận scheduler/refit/drop_last mismatch | Một fold/seed; score không dùng ranking |
| Inline resolved-criterion probe | Synthetic labels/config | Missing class-weight key tạo weighted criterion | Xác nhận critical fair bug | Unit probe, không ước lượng metric effect |
| Checksum-only audit latest final bundle | 32 manifest files | 32 match; checkpoint map rỗng | Bundle bytes nhất quán nhưng không bind model | Không mở/recompute historical predictions |
| `py -3.10 -m pytest tests/test_recommendation_policy.py -q` | Recommendation unit tests | 12 pass | Current code contract ổn theo tests | Không có expert/content/outcome ground truth |
| Synthetic recommendation boundary probes | Missing features/confidence | Missing tạo false risk; confidence ngoài miền pass | Xác nhận safety gaps | Synthetic technical diagnostic, không quality evaluation |

Không diagnostic nào dùng 79 records để chọn mô hình. Không full nested-CV, full Optuna search, full multi-seed hay locked/external test được chạy.

## G. Kế hoạch sau khi được phê duyệt

| Phase | Thứ tự | Deliverable | Stop condition chính |
|---|---:|---|---|
| A — Reproduction | 1 | Provenance/quarantine registry, ledgers, 30-fit control | Hash/membership/config mismatch |
| B — Corrections | 2 | Resolved estimator parity, B1/B2 training policy | Criterion/refit parity hoặc reproducibility fail |
| C — Candidates | 3 | 5-family nested OOF + matched ablations | Leakage, invalid ledger, hard collapse; conditional gates đóng nếu không đạt |
| D — Recommendation | 4 | Governed policy, goal/action lifecycle, expert package | Safety/expert gate fail |
| E — Full validation | 5 | Multi-seed top models, calibration, frozen champion bundle | Không strict-validate hoặc unresolved correctness issue |
| F — External | 6 | One-shot external confirmation | Không có genuinely new data → báo limitation, không thay bằng 79 |

Budget base ước tính 4.061–4.160 fits; tối đa 4.280 nếu cả residual và imbalance conditional gates mở. Workload xấp xỉ một fair benchmark hiện có. Mỗi phase tạo immutable artifacts mới, không ghi đè evidence lịch sử.

## Yêu cầu phê duyệt

Đề nghị người dùng phê duyệt hoặc sửa rõ Decisions 1–15. Trước khi có phê duyệt, dự án dừng tại bộ tài liệu chiến lược này: không sửa model/source/README, không chạy full experiment và không truy cập 79 observed để đánh giá.
