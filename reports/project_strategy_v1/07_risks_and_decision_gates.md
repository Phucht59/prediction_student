# 07 — Risk register và decision gates

## 1. Quy ước

- **Severity:** Critical / High / Medium / Low.
- **Evidence strength:** Confirmed / Strong / Suggestive / Unverified.
- **Expected impact:** tác động nếu issue tồn tại trong claim/model active.
- **Fix cost:** tổng hợp code, validation, compute/data/human cost.
- “Confirmed” nghĩa là đã đối chiếu bằng code/artifact/DB/diagnostic; không đồng nghĩa đã đo chính xác effect size lên metric.

## 2. Risk register ưu tiên

| ID | Phân loại | Vấn đề | Severity | Evidence strength | Expected impact | Fix cost | Hành động bắt buộc |
|---|---|---|---|---|---|---|---|
| P01 | Critical correctness issue | Fair deep inner search mất constants và outer rơi vào balanced class weight, trái protocol `none` | **Critical** | Confirmed | High | Code Low; rerun High | Quarantine DL rows; resolved-config fail-fast; rerun |
| P02 | Scientific validity issue | 79 records đã được xem qua nhiều model, không còn untouched test | **Critical** | Confirmed | High | High/new data | Đổi nhãn observed; cấm selection/confirmation; external data mới |
| P03 | Training-estimator issue | Model selection full refit nhưng final pipeline giữ 15% validation, không refit full development | **High** | Confirmed | High | Medium | Một estimator factory cho inner/outer/final |
| P04 | Training-estimator issue | Epoch chọn dưới ReduceLROnPlateau/SWA nhưng refit fixed LR, bỏ scheduler/SWA | **High** | Confirmed | High | Medium | Fixed/replayable training policy; bỏ dead complexity |
| P05 | Training-estimator issue | `drop_last=True` bỏ 22–28 records/epoch trong smoke nhỏ | Medium | Confirmed behavior; effect unverified | Medium | Low | Paired ablation; default đề xuất false |
| P06 | Training-estimator issue | SWA với BatchNorm nhưng không thấy `update_bn`; SWA không đi vào refit estimator | Medium | Confirmed code | Medium | Low | Bỏ hoặc tích hợp đầy đủ + test |
| P07 | Reproducibility/engineering issue | Fair artifact commit không chứa runner; selection config không khớp commit ghi trong manifest | **High** | Confirmed | High | Medium + rerun | Source-tree/diff hash; immutable clean runs |
| P08 | Reproducibility issue | Final evidence checkpoint map rỗng | **High** | Confirmed | High | Low | Hash exact checkpoints/preprocessor/calibrator |
| P09 | Model-design limitation | Sequence input chỉ dài 2; kernel 1 pointwise; 13.059 params trên N nhỏ | **High** | Confirmed facts; performance mechanism Strong | High | Medium | Tiny/matched controls; hạ temporal claim |
| P10 | Evaluation issue | G2 deterministic rule không nằm trong current fair runner dù rất mạnh | **High** | Confirmed | High | Low | Bổ sung R0 baseline cùng folds/protocol |
| P11 | Evaluation issue | Current fair runner thiếu QWK/ordinal MAE/step errors dù protocol yêu cầu | Medium | Confirmed | Medium | Low | Recompute từ OOF và đưa vào runner |
| P12 | Evaluation issue | Fair neural outer chỉ một seed; chưa có seed stability | Medium–High | Confirmed | High | Medium–High compute | 3 outer seeds, 5 final stability seeds |
| P13 | Evaluation issue | Pruned neural trials và completed classical trials làm “30 trials” không đồng nghĩa equal compute | Medium | Confirmed | Medium | Low | Report states/fit time; claim budget chính xác |
| P14 | Scientific validity issue | Old HGB baseline dùng full context, không cùng G1/G2 | **High** | Confirmed | High | Low + rerun | Gắn nhãn exploratory; strict feature comparison riêng |
| P15 | Scientific validity issue | Operational target bins có nhiều samples sát biên, rationale/sensitivity chưa freeze | **High** | Confirmed | High | Medium | Versioned target contract + development-only sensitivity |
| P16 | Future-data limitation | Không student/entity ID; không group split/overlap proof khi ghép data | Medium | Confirmed absence; duplicate-person risk Suggestive | Medium–High | High | Không ghép nguồn trước entity resolution |
| P17 | Future-data/model limitation | Claim nhiều học kỳ/early warning không được two-assessment late-stage data hỗ trợ | **High** | Confirmed | High | High/new data | Hạ claim; timestamped multi-assessment data |
| P18 | Scientific validity issue | Context questionnaire timing unknown; adding branch có temporal leakage risk | **High** | Confirmed contract | High | Medium–High | Context gate + snapshot/freshness contract |
| P19 | Model-design limitation | Ordinal prototype không integrated/tested; không phải evidence | Medium–High | Confirmed | Medium | Medium | Tiny ordered integration + semantic tests |
| P20 | Model-design limitation | Simple residual hypothesis không thắng G2 rule về Macro-F1 | Medium | Strong diagnostic, chưa full tuned | Medium | Low | Demote to conditional gate |
| D01 | Data/engineering correctness | Target insert `ON CONFLICT DO NOTHING` rồi chỉ count; conflicting values có thể lọt | **High** | Confirmed design; live sạch | High | Low | Compare all existing target fields/hash |
| D02 | Data correctness | DB target constraints không chặn encoded >2/raw ngoài 0–20/contract inconsistency | **High** | Confirmed design; live sạch | High | Low | CHECK/trigger dùng central target contract |
| D03 | Data lineage | Prediction true label/probability invariants yếu ở DB | **High** | Confirmed design | High | Medium | FK/trigger/schema constraints |
| D04 | Reproducibility | Manifest identity phụ thuộc DB-local `dataset_version_id=1` | **High** | Confirmed | High | Medium | Portable content-hash identity |
| D05 | Evaluation/reproducibility | Inner/early-stop membership không có durable record ledger | Medium | Confirmed | Medium | Medium | Materialize + checksum |
| D06 | Leakage defense-in-depth | `raw_payload` chứa G3; official path chặn nhưng bypass có thể đọc | Medium | Confirmed | High | Medium | Target-free view/DB role/allowlist tests |
| D07 | Data contract | Ingestion contract thiếu dtype/nullability/enum/range/upstream provenance | Medium | Confirmed | Medium | Low | Mở rộng contract manifest |
| D08 | Engineering | Không migration ledger; 2 stale runs `running` | Medium | Confirmed live | Low–Medium | Low | Migration checksums + timeout/recovery |
| R01 | Recommendation safety | Missing/invalid features mặc định thành G1/G2=0, studytime=1 và tạo false risk | **High** | Confirmed | High | Low–Medium | Unknown/abstain + schema validation |
| R02 | Recommendation safety | Confidence ngoài `[0,1]` vẫn pass | **High** | Confirmed | High | Low | Fail-fast probability contract |
| R03 | Recommendation governance | Policy dùng `internet` dù feature inventory excluded/sensitive/timing unknown | **High** | Confirmed | High | Medium | Shared feature registry + policy approval |
| R04 | Recommendation validity | `absences/studytime` chia count cho ordinal category, sai đơn vị | **High** | Confirmed | High | Low | Bỏ ratio, expert-validated semantics |
| R05 | Recommendation evaluation | `no_contradiction` chỉ đo unique risk codes; bỏ sót duplicate/conflict/workload | **High** | Confirmed | High | Medium | Rename + semantic action checks |
| R06 | Recommendation-system gap | Không goals/actions có cấu trúc, follow-up, feedback, revision | **High** | Confirmed | High | High | Level 0.5 lifecycle schema/workflow |
| R07 | Recommendation evidence | 12 expert cases chưa có rating; structural metrics không chứng minh quality | **High** | Confirmed | High | High/human | ≥2 experts, stratified rubric + agreement |
| R08 | Recommendation engineering | Experiment prediction schema yêu cầu true label, không phù hợp production | **High** | Confirmed | High | High | Snapshot/outcome separation |
| R09 | Recommendation provenance | Không policy registry/hash/approver/evidence version | Medium–High | Confirmed | High | Medium | Immutable policy registry |
| R10 | Recommendation engineering | Verifier count mâu thuẫn append-only multi-policy | Medium–High | Confirmed | Medium | Medium | Filter by policy version/composite key |
| R11 | Recommendation consistency | Hai builder paths tạo envelope khác nhau dưới cùng policy string | Medium | Confirmed | Medium | Low–Medium | Canonical builder/schema version |
| R12 | External validity | Portuguese secondary-school data/actions chưa validate cho đại học Việt Nam | **High** | Confirmed domain mismatch; effect unverified | High | High | Localization experts + prospective domain data |
| R13 | Causal validity | Không treatment/exposure/propensity/adherence/outcome; learned/causal recommender không khả thi | **High** | Confirmed | High | High/future data | Giữ non-causal expert-guided scope |
| E01 | Documentation issue | README/PROJECT vẫn gọi 79 locked hoặc dễ dẫn tới claim này | **High** | Confirmed | High | Low sau approval | Reconcile claim language, không sửa ở audit |
| E02 | Documentation/evidence issue | `MODEL_IMPROVEMENT_PLAN_V3.md` dùng benchmark numbers không có supporting artifact hiện tại | Medium–High | Confirmed workspace | Medium | Low | Gắn historical/unsupported label hoặc regenerate |
| E03 | Reproducibility issue | Raw/fair artifacts bị gitignore; clone không có exact evidence bytes | Medium | Confirmed | Medium | Low–Medium | Release manifest/archive strategy |

## 3. Rủi ro hệ thống nếu chọn sai thứ tự

```text
Chạy search trước khi sửa P01/P03/P04
  → tốn compute trên estimator không thống nhất
  → leaderboard không thể diễn giải
  → chọn architecture sai
  → final model khác model được estimate
  → recommendation dùng uncertainty sai
  → claim/evidence phải rerun toàn bộ
```

Vì vậy correctness gates đứng trước model performance gates. Không có metric gain nào hợp thức hóa một estimator bug.

## 4. Decision gates tuần tự

### Gate G0 — Phê duyệt chiến lược

**Cần phê duyệt:** Strategy B, candidate families, budget, seeds, primary metric, 79 observed policy, context/residual gates và recommendation scope.

**Không pass nếu:** người dùng muốn đổi bins/folds/population, bắt buộc CNN champion, hoặc mở context mà chưa có data contract.

### Gate G1 — Data/protocol integrity

**Evidence cần có:**

- raw/DB/target/fold hashes;
- 316 membership và class counts;
- portable record IDs;
- inner/early-stop ledgers;
- target-free input graph;
- explicit `legacy_heldout_observed` registry.

**Pass:** tất cả invariants exact match.  
**Fail:** dừng mọi fit; sửa data/protocol và version lại.

### Gate G2 — Estimator correctness

**Evidence cần có:**

- resolved config chứa constants;
- inner/outer/final criterion/resampling parity tests;
- same estimator factory;
- scheduler/refit/drop-last semantics explicit;
- final full-development refit path;
- reproducible checkpoint hash.

**Pass:** semantic tests và fixed smoke đều pass; không silent default.  
**Fail:** không chạy Phase C.

### Gate G3 — Training policy

So B1 fixed-policy/drop-last true và B2 fixed-policy/drop-last false trên paired folds/seeds.

**Pass:** chọn một policy dựa trên pre-registered primary/stability/sample-utilization rule; không class collapse.  
**Fail:** thu nhỏ model/batch hoặc sửa loader; không mở architecture search.

### Gate G4 — Candidate comparison

**Evidence cần có:** full 5 outer folds, all declared seeds/trials, OOF probabilities, primary/guardrails, provenance và paired intervals.

**Pass:** chọn family theo frozen rule.  
**Fail:** nếu không có winner rõ, chọn simplest stable model; không mở thêm candidate tùy hứng.

### Gate G4-R — Residual/multitask

Chỉ mở khi ordered model có tín hiệu ordinal nhất quán và không làm giảm primary quá tolerance. Simple residual diagnostic hiện **không đủ để pass**.

### Gate G4-I — Imbalance

Chỉ mở khi selected model vi phạm minority recall/collapse guardrail. Không mở chỉ vì class counts không đều.

### Gate G4-C — Context

**Hiện đóng.** Chỉ mở khi có snapshot timing/freshness/source/modifiability/sensitivity contract và cùng feature contract cho ML/DL. Đây là change request cần phê duyệt mới.

### Gate G5 — Recommendation technical safety

**Pass conditions:**

- invalid probability/missing/stale/forbidden cases reject/abstain 100%;
- deterministic + idempotent dưới exact policy hash;
- zero duplicate/incompatible action IDs;
- action workload/prerequisites hợp lệ;
- canonical payload + policy registry;
- production snapshot tách outcome.

**Fail:** không đưa case cho user/student; chỉ tiếp tục test nội bộ.

### Gate G6 — Expert content validation

**Pass conditions đề xuất:**

- ≥60 stratified development OOF cases;
- ≥2 experts independent;
- no critical unsafe case;
- median relevance/safety ≥4/5;
- agreement ≥0,60;
- all adjudications resolved trong policy version mới.

**Fail:** revise policy và đánh giá lại; không tự tạo quality score thay expert evidence.

### Gate G7 — Full validation và freeze

**Pass conditions:**

- 5 new predeclared seeds cho top neural;
- paired top neural/top ML analysis;
- calibration/abstention inner-only;
- strict artifact validator;
- exact checkpoint/source/config/environment hashes;
- claim matrix được duyệt;
- không high correctness issue unresolved.

**Fail:** chọn simpler model hoặc giữ nghiên cứu ở development stage; không mở 79.

### Gate G8 — External confirmation

**Pass-to-open:** dữ liệu ngoài/prospective mới đã hash/pre-register, không overlap, model/policy frozen trước labels.

**Nếu không có:** khóa luận báo nested-CV development estimate và limitation. Không dùng 79 observed thay thế.

## 5. Claim matrix

### 5.1 Có thể nói ngay sau audit

- Repository có versioned PostgreSQL lineage, immutable outer-fold manifest và test suite 87 pass/5 skip.
- Active prediction dùng G1/G2, sequence length 2, late-stage cutoff.
- G2 là predictor mạnh trong development và deterministic G2 rule là baseline bắt buộc.
- Existing fair DL comparison vi phạm intended loss/imbalance protocol.
- 79 historical records không còn unseen.
- Recommendation v3 là deterministic rule-based snapshot có version/storage/explanation.
- Recommendation chưa có lifecycle/expert/effectiveness evidence.

### 5.2 Chỉ có thể nói sau G4/G7

- Candidate X tốt hơn/yếu hơn candidate Y trên immutable development nested-CV.
- Ordered inductive bias cải thiện error structure.
- CNN/BiLSTM component có hoặc không có incremental value dưới matched protocol.
- Model ổn định qua declared seeds.
- Calibration/abstention cải thiện metric tương ứng.

Mọi câu superiority phải kèm paired estimate, uncertainty, feature contract, folds và seed protocol.

### 5.3 Chỉ có thể nói sau G6

- Expert đánh giá policy phù hợp/safe theo rubric và population đã nêu.
- Advisor workflow khả thi trong shadow context nếu usability gate cũng pass.

Không được suy từ expert content rating sang “cải thiện điểm”.

### 5.4 Chỉ có thể nói sau G8/prospective study

- Generalization trên external population cụ thể.
- Recommendation cải thiện process/academic outcome nếu prospective comparator và CI hỗ trợ.
- Causal effect chỉ khi causal design thực sự hợp lệ.

## 6. Quy tắc khi evidence mâu thuẫn

1. Code + exact artifact provenance cao hơn README narrative.
2. Corrected nested-CV cao hơn historical observed holdout cho model selection.
3. Full OOF paired evidence cao hơn one-fold/smoke.
4. Multiple declared seeds cao hơn best seed.
5. Ground-truth expert/prospective ratings cao hơn structural self-score.
6. Khi provenance không bind exact source/config, hạ evidence xuống historical/suggestive.
7. Khi primary và auxiliary metrics mâu thuẫn, áp pre-registered primary + guardrail rule; không chọn metric thuận lợi sau khi xem kết quả.

## 7. Quyết định không được tự động hóa

Những thay đổi sau cần người dùng phê duyệt mới:

- đổi target bins, population, primary metric hoặc immutable folds;
- thêm context/new dataset;
- tăng candidate families/trials/seeds vượt budget;
- bắt buộc/loại CNN khỏi final candidate set;
- mở residual/multitask khi gate không đạt;
- dùng 79 observed;
- triển khai advice tự động không qua advisor;
- chuyển từ expert-guided sang learned/causal recommendation;
- thay claim domain sang đại học Việt Nam khi chưa có validation.

## 8. Kết luận quản trị rủi ro

Hai rủi ro lớn nhất không phải “model score thấp” mà là **chọn model từ comparison sai estimator** và **đưa ra final-test claim khi không còn unseen test**. Hai gate này phải được đóng trước mọi optimization. Ở recommendation, rủi ro lớn nhất là biến missing/correlation thành lời khuyên có vẻ chắc chắn mà chưa có expert/causal evidence. Strategy B giảm đồng thời ba rủi ro bằng estimator parity, candidate pool phù hợp dữ liệu và human-in-the-loop lifecycle.
