# 06 — Kế hoạch thực nghiệm đề xuất sau khi được phê duyệt

> **Chưa có phase nào trong tài liệu này được chạy.** Đây là kế hoạch, budget và stop conditions để người dùng phê duyệt trước.

## 1. Protocol cố định

### 1.1 Population và split

- Selection population: 316 immutable development records của Protocol V2.
- Outer evaluation: 5 immutable stratified folds.
- Inner selection: 3 stratified folds, materialize record ledger/checksum trước run.
- Early-stop membership: cũng materialize theo outer/inner/seed.
- 79 historical observed records: **không truy cập để chọn, calibrate, xác nhận hay viết claim mới**.
- Final external confirmation: chỉ trên dữ liệu mới thực sự chưa quan sát, nếu có.

### 1.2 Feature contract

- Primary track: late-stage G1/G2.
- Mọi model trong fair comparison nhận cùng raw information.
- Không context cho main study.
- Early-warning G1-only là nghiên cứu riêng tương lai, không trộn metric với late-stage.
- Derived feature nếu có phải deterministic từ G1/G2, pre-registered và công bằng giữa families.

### 1.3 Outcome và metrics

**Primary selection metric:** outer OOF Macro-F1.

**Guardrails bắt buộc:**

- class-wise precision/recall/F1 và confusion matrix;
- QWK;
- ordinal MAE trên class index;
- one-step và two-step error;
- error tại raw boundaries 9/10/14/15;
- accuracy;
- NLL, multiclass Brier và ECE;
- seed mean/SD/range và class-collapse count;
- parameter count, fit time, peak memory nếu khả thi.

Không chọn model bằng một composite score tự tạo. Nếu Macro-F1 gần hòa trong uncertainty, ưu tiên model đơn giản, ổn định, calibrated và ít two-step error hơn theo tie rule đã freeze.

### 1.4 Config và estimator contract

Một `resolved_config` canonical phải chứa cả:

- suggested hyperparameters;
- constant/default policy values;
- feature contract hash;
- target contract hash;
- split manifest hash;
- loss/resampling/class-weight settings;
- scheduler/refit/drop-last/SWA semantics;
- seed list;
- source tree commit + dirty diff hash;
- environment lock hash.

Inner, outer và final estimator phải nhận cùng resolved object. Missing required key phải fail-fast.

### 1.5 Quy ước tính compute

- **Trial:** một đề xuất hyperparameter của một family.
- **Neural fit:** một lần train model. Một evaluation theo two-stage estimator gồm 2 fits: chọn epoch trên train/validation, rồi full-fold fixed-policy refit.
- **ML fit:** một lần `.fit()` của estimator classical.
- **Job:** một đơn vị orchestration theo model family × outer fold; bên trong có thể có nhiều fits tuần tự.

Các counts dưới đây là upper-bound planning, không phải kết quả đã chạy.

## 2. Phase A — Reproduction và evidence quarantine

### Mục tiêu

Tạo baseline kiểm soát có provenance đúng, xác minh dữ liệu/folds/config và tách rõ historical evidence khỏi evidence có thể dùng.

### Công việc

1. Freeze Git state trên branch mới sau approval; lưu commit, diff hash và environment.
2. Verify raw/DB/manifest checksums, target contract và outer membership.
3. Materialize inner/early-stop ledgers.
4. Đánh dấu metadata:
   - 79 → `legacy_heldout_observed`;
   - fair deep rows hiện tại → `invalid_protocol_config_resolution`;
   - nested-full 20260710 → `historical_old_estimator`;
   - old README/report metrics → historical snapshots.
5. Chạy fixed current-config control trên 5 outer folds × 3 seeds, development-only, để có paired reference cho corrections. Đây không phải full search.
6. Checkpoint/artifact hash cho control; không update README headline.

### Budget

| Mục | Jobs | Fits |
|---|---:|---:|
| Hash/ledger/test verification | 1 | 0 |
| Fixed neural control | 5 | `5 folds × 3 seeds × 2 stages = 30` |
| **Phase A total** | **6** | **30** |

### Deliverables

- Provenance manifest;
- resolved-config snapshot;
- split/inner/early-stop ledgers;
- OOF control + fold/seed metrics;
- evidence quarantine registry.

### Stop conditions

Dừng ngay nếu raw/DB/manifest hash lệch, record membership/target mismatch, target xuất hiện trong feature graph, hoặc current config không resolve deterministic. Không chuyển Phase B bằng cách “sửa tại chỗ” evidence.

## 3. Phase B — Pipeline corrections

### 3.1 Defect fixes không được quyết định bằng điểm số

Các correction sau là bắt buộc, kể cả nếu historical score xấu hơn:

- giữ constants trong resolved Optuna config;
- missing loss/class weight/oversample key phải raise;
- selection/final dùng cùng estimator factory;
- final model refit toàn development theo policy đã chọn;
- source/protocol/checkpoint hashes đầy đủ;
- verifier filter policy version đúng;
- semantic contract tests.

Không A/B-test bug rồi giữ bug vì score cao.

### 3.2 Training-design ablation tuần tự

Hai design hợp lệ được so paired, không trộn nhiều thay đổi mơ hồ:

- **B1:** fixed LR, không SWA, `drop_last=True`, cùng policy ở epoch-selection và refit.
- **B2:** B1 + `drop_last=False`.

Lựa chọn fixed LR/no SWA nhằm làm selection/refit replay được và bỏ dead complexity. Nếu muốn scheduler, phải thiết kế scheduler theo epoch có thể replay deterministic; không dùng validation-driven `ReduceLROnPlateau` rồi bỏ nó ở refit.

Mỗi design dùng frozen current architecture/config, 5 folds × 3 seeds × 2 stages.

### Budget

| Variant | Jobs | Fits |
|---|---:|---:|
| B1 | 5 | 30 |
| B2 | 5 | 30 |
| **Phase B total** | **10** | **60** |

### Decision rule

- Correctness tests phải pass tuyệt đối.
- Giữa B1/B2, ưu tiên `drop_last=False` trừ khi paired evidence cho thấy instability rõ và có nguyên nhân tái lập; không chọn theo một fold.
- Báo effect từng fold/seed, sample utilization và runtime.

### Stop conditions

Dừng nếu inner/outer/final criterion hoặc sample membership không parity, có nondeterministic config, NaN, target access, class collapse ở ≥2 folds mà không có root cause, hoặc checkpoint cannot reproduce prediction.

## 4. Phase C — Architecture candidates

### 4.1 Main candidate pool

| Family | Search | Outer seed policy | Fits ước tính |
|---|---:|---:|---:|
| G2 deterministic rule | 0 trial | Không seed | 0 |
| Random Forest | 30 trials × 3 inner | 1 outer fit/fold | `5 × (30×3 + 1) = 455` |
| SVM | 30 trials × 3 inner | 1 outer fit/fold | 455 |
| Corrected compact CNN–BiLSTM nominal | 30 trials × 3 inner × 2 stages | 3 outer seeds × 2 stages | `5 × (30×3×2 + 3×2) = 930` |
| Tiny nominal MLP | Như neural ở trên | 3 outer seeds | 930 |
| Tiny ordered MLP/CORAL | Như neural ở trên | 3 outer seeds | 930 |
| **Main pool** |  |  | **3.700 fits** |

Search dùng một fixed search seed trong inner trials để giữ budget; stochastic stability của selected config được đo bằng 3 outer seeds và 5 new seeds ở Phase E. Seeds không được coi là independent outer folds.

### 4.2 Fixed matched ablations

- CNN-only;
- BiLSTM-only;
- CNN-LSTM hoặc simplified recurrent control.

Mỗi variant: `5 folds × 3 seeds × 2 stages = 30 fits`; tổng 90. Parameter budget và training policy phải matched gần nhất có thể.

### 4.3 Low-cost sanity baselines

- Multinomial logistic;
- ordinal logistic/cumulative link nếu implementation ổn định;
- fixed Huber regression → bin.

Không full tune; 3 families × 5 folds = 15 fits. Mục tiêu là kiểm tra complexity floor, không làm rộng search space.

### 4.4 Phase C base budget

| Nhóm | Jobs | Fits |
|---|---:|---:|
| Main families | 25 | 3.700 |
| Neural matched ablations | 15 | 90 |
| Fixed classical sanity | 15 | 15 |
| **Phase C base** | **55** | **3.805** |

### 4.5 Conditional gates

#### Gate C-R — Residual/multitask

Chỉ mở nếu tiny ordered MLP:

- không thua nominal MLP về mean Macro-F1 quá pre-registered tolerance;
- cải thiện ordinal guardrail ở đa số outer folds;
- không có threshold monotonicity violation;
- không class collapse.

Khi mở, chạy tối đa hai fixed variants:

1. gated zero-init residual + Huber;
2. ordered classification + Huber auxiliary.

Budget tối đa: `2 × 5 × 3 × 2 = 60 fits`.

#### Gate C-I — Imbalance

Chỉ mở nếu selected neural family vi phạm pre-registered High-class recall/collapse guardrail. Chạy từng method riêng, tối đa hai alternatives trong `class weight`, `focal`, `random oversampling`; không SMOTE/ADASYN default và không double compensation.

Budget tối đa: 60 fits.

#### Conditional total

Tối đa 120 fits; nếu gates không mở thì 0.

### Selection procedure

1. Freeze search spaces trước khi outer scores có thể được xem tổng hợp.
2. Chạy mỗi family đủ 5 outer folds; không loại family vì một fold đầu xấu, trừ hard failure pre-registered.
3. Lưu all OOF predictions/probabilities, resolved configs, trial states và provenance.
4. So paired predictions trên cùng records; báo delta và bootstrap CI phù hợp.
5. Không tuyên bố superiority chỉ vì mean hơn vài phần nghìn hoặc CI rộng qua 0.
6. Tie → chọn model đơn giản/ổn định/calibrated hơn.

### Stop conditions

- Fail-fast nếu leakage/config/ledger invariant hỏng.
- Prune trial chỉ theo rule freeze trước; báo completed/pruned counts.
- Có thể dừng conditional branches nếu gate không đạt.
- Không dừng main family theo partial outer leaderboard.

## 5. Phase D — Recommendation improvement

Phase này không cần predictive model fits mới; dùng development OOF predictions của frozen candidate/control.

### D1 — Technical safety

- Canonical schema và payload builder;
- confidence/probability/domain validation;
- missing/stale → unknown/abstain;
- feature governance enforcement;
- bỏ invalid ratio;
- action catalog + duplicate/conflict/prerequisite/workload tests;
- policy registry/hash/approval status;
- split production prediction snapshot khỏi outcome;
- multi-policy-aware verifier.

### D2 — Goal/action lifecycle

- Structured learning goals;
- plan actions và schedule/owner/status;
- advisor approve/modify/reject;
- follow-up/adherence/adverse-event events;
- immutable revisions/supersedes;
- outcome linkage.

### D3 — Expert validation

- Tối thiểu 60 cases phân tầng từ development OOF, không dùng 12 rows đầu tiên và không iterate trên 79 observed.
- Ít nhất 2 chuyên gia độc lập.
- Rubric: relevance, safety, feasibility, specificity, workload, explanation, fairness.
- Adjudication + weighted kappa/Krippendorff alpha.

### Gate để sang shadow pilot

- 0 critical unsafe case;
- median safety và relevance ≥4/5;
- agreement ≥0,60;
- 100% missing/forbidden/confidence tests pass;
- 0 duplicate/incompatible action IDs;
- mọi recommendation cần human approval trước active.

### Deliverables

Schema/migrations, policy registry, action catalog, safety test report, expert casebook, raw ratings, agreement analysis, approved policy version và shadow-pilot protocol.

### Stop conditions

Dừng nếu có critical unsafe case, expert agreement dưới gate, action không localize được cho domain, hoặc policy phụ thuộc feature timing chưa xác nhận. Không tự chỉnh rubric/cutoff sau khi xem ratings mà không tạo policy version mới và review lại.

## 6. Phase E — Full validation và freeze

### 6.1 Multi-seed stability

Chọn trước một top neural và một top ML từ Phase C theo rule đã freeze.

- Neural: 5 folds × 5 **new predeclared seeds** × 2 stages = 50 fits.
- ML upper bound nếu stochastic: 5 folds × 5 seeds = 25 fits. Nếu deterministic SVM, report 1 fit/fold và unused budget.
- Tổng upper bound: 75 fits.

Đo mean/SD/range, worst seed, collapse count và paired OOF delta. Không chọn seed đẹp để deploy.

### 6.2 Calibration/abstention

- Fit temperature/cutpoints từ inner-OOF only.
- So uncalibrated vs calibrated bằng outer OOF NLL/Brier/ECE và Macro-F1.
- Pre-register abstention thresholds và coverage-risk curve.
- Recommendation chỉ nhận automatic policy output khi uncertainty qua gate; còn lại human review.

### 6.3 Chọn family và final development refit

- Freeze winner bằng selection rule, không bằng title/architecture preference.
- Nếu winner là ML: final inner search `30×3=90` + one full-development fit = 91 fits.
- Nếu winner là neural: final inner search `30×3×2=180` + 5-seed full-development ensemble, mỗi seed two-stage = 10 fits; total 190.
- Hash checkpoint(s), preprocessor, calibrator, resolved config, source tree, environment và all input manifests.

### Phase E budget

| Winner case | Stability | Final selection/refit | Total |
|---|---:|---:|---:|
| ML winner | 75 upper bound | 91 | 166 |
| Neural winner | 75 upper bound | 190 | 265 |

### Freeze gate

- Strict validator pass;
- no unresolved Critical/High correctness issue;
- reproducible predictions from bundle;
- all 5 outer folds + all declared seeds present;
- calibration/metric/provenance schema complete;
- claim language approved;
- no access to 79 observed for any decision.

Nếu neural không rõ ràng tốt hơn ML, chọn simpler stable champion và giữ CNN–BiLSTM như research model. Đây là kết quả hợp lệ, không phải thất bại.

## 7. Phase F — External locked confirmation, không dùng lại 79

Phase F theo yêu cầu khoa học ban đầu chỉ có thể thực hiện khi tồn tại dataset mới thực sự chưa quan sát. Existing 79 **không đáp ứng điều kiện này**.

### Điều kiện mở

1. Model family/config/seeds/preprocessor/calibrator/policy đã freeze.
2. Population, domain, collection period, inclusion/exclusion và target mapping được pre-register.
3. Raw external hash và acquisition timestamp được ghi trước khi đọc labels.
4. Không overlap entity/source với development; nếu có entity IDs phải kiểm group overlap.
5. Không dùng external labels để sửa model/threshold/policy sau evaluation.

### Thực hiện

- Một lần inference theo frozen bundle.
- Báo primary/guardrail metrics với CI và domain-shift diagnostics.
- Không re-open selection sau kết quả.
- Nếu thất bại, báo external validation failure; nghiên cứu tiếp theo phải tạo protocol/dataset version mới, không vá current claim.

### Nếu không có external data trước hạn khóa luận

Báo trung thực kết quả development nested-CV và giới hạn “chưa có unseen external confirmation”. Không thay thế bằng 79 observed và không gọi nó là locked test.

## 8. Tổng compute và dung lượng

### 8.1 Base plan

| Phase | Fits |
|---|---:|
| A | 30 |
| B | 60 |
| C base | 3.805 |
| D | 0 predictive fits |
| E, ML winner | 166 |
| E, neural winner | 265 |
| **Base total, ML winner** | **4.061** |
| **Base total, neural winner** | **4.160** |
| Conditional residual + imbalance max | **+120** |
| **Absolute planned range** | **4.061–4.280 fits** |

Main orchestration khoảng 81 jobs; tối đa khoảng 101 khi cả conditional gates mở. Đây là logical jobs, không phải số process chạy đồng thời.

### 8.2 Relative runtime

- Existing fair benchmark tương đương xấp xỉ 4.095 fit units theo cùng cách đếm; base Balanced plan khoảng 0,99–1,02 lần workload đó.
- One-fold two-stage smoke trên CPU mất khoảng 14,6 giây trong môi trường audit, nhưng không nên ngoại suy tuyến tính tuyệt đối vì trial architectures, pruning, DB I/O và parallelism khác nhau.
- Planning range thực tế: nhiều giờ đến khoảng 1–2 ngày CPU nếu chạy tuần tự/ít song song; cần pilot timing Phase A trước khi chốt wall-clock.
- Không hứa thời gian GPU vì môi trường audit là CPU build.

### 8.3 Disk

- Nếu chỉ lưu CSV/JSON/OOF/protocol và checkpoint finalists: mục tiêu <1 GB.
- Không lưu checkpoint của mọi trial; nếu lưu sẽ tăng lên hàng chục GB không cần thiết.
- Lưu Optuna DB, trial summaries, OOF probabilities và exact final checkpoints.
- Historical artifacts không bị ghi đè; run mới dùng immutable run IDs.

## 9. Deliverable map

| Phase | Deliverable quyết định |
|---|---|
| A | Reproduction/provenance registry và corrected control |
| B | Estimator parity tests + selected training policy |
| C | Full OOF candidate/ablation comparison và recommendation gate inputs |
| D | Governed policy, lifecycle schema, expert validation package |
| E | Frozen champion bundle, calibrated uncertainty, strict validator report |
| F | External confirmation report hoặc explicit “not available” limitation |

## 10. Nguyên tắc dừng sớm toàn chương trình

Dừng và quay về phê duyệt nếu:

- cần thay target bins, population, folds hoặc primary metric;
- muốn mở context/new dataset;
- compute thực tế vượt 1,5× budget mà không có root cause;
- candidate set phải mở rộng ngoài pre-registration;
- discovered bug làm invalid completed outer folds;
- expert review phát hiện critical safety issue;
- external data collection thay đổi domain/target contract;
- có yêu cầu dùng 79 observed để “xác nhận nhanh”.

Mỗi trường hợp trên là một thay đổi nghiên cứu, không phải implementation detail được phép tự quyết.
