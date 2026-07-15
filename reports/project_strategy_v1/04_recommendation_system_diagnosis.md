# 04 — Chẩn đoán hệ thống khuyến nghị

## 1. Kết luận điều hành

Hệ thống hiện tại là một **policy tư vấn theo luật, deterministic, có explanation và lưu version**, không phải learned recommender và chưa đủ gọi là “lộ trình học thích ứng”. Điểm mạnh nhất là output tái lập, liên kết prediction–recommendation và lưu append-only trong PostgreSQL. Khoảng trống cốt lõi là:

- không có goal/action object với baseline, target, unit, deadline và owner;
- không có advisor approval/modify/reject được lưu;
- không có follow-up observation, adherence, feedback, plan revision hay outcome;
- chưa có expert ratings;
- structural metrics hiện không đo content validity, safety hay effectiveness;
- một số rule có lỗi semantic/safety cụ thể.

Tên đúng ở giai đoạn này là:

> **Evidence-informed, expert-guided rule-based advisory prototype; non-causal and human-in-the-loop.**

## 2. Inventory hiện trạng

| Thành phần | Hiện trạng | Mức trưởng thành | Evidence |
|---|---|---|---|
| Rule-based advice | 9 feature/risk codes, action mapping | Prototype active | `src/recommendation.py:22-32,115-278` |
| Policy | `student_mat_rule_policy_v3`, deterministic | Active nhưng chưa expert-approved | `src/recommendation.py:21,99-219` |
| Risk band | Đảo từ predicted performance class | Active heuristic | `src/recommendation.py:99-112` |
| Uncertainty | Ba band theo max-softmax 0,55/0,75 | Prototype | `src/recommendation.py:19-21,281-323` |
| Explanation | Prediction basis, evidence string, rationale, disclaimer, causal scope note | Prototype tốt | `src/recommendation.py:325-338` |
| Learning path | 3 template tĩnh theo risk band, mỗi template 4 tuần | Minh họa | `src/recommendation.py:294-314` |
| Goal planning | Một số câu văn “set measurable target” | Chưa có object | Không có schema/table tương ứng |
| Follow-up | Lời nhắc review trong text | Chưa có mechanism | Không có event/revision state |
| Expert validation | CSV 12 case, rating rỗng | Chưa thu thập | final evidence review CSV |
| Evaluation | Structural self-check | Prototype | `src/recommendation.py:431-477` |
| Storage | JSONB snapshot theo prediction/policy version | Active | migrations 001–002 |
| Policy provenance | Chỉ có string version | Thiếu registry | DB/schema inventory |
| Production inference | Không có service/schema riêng; prediction yêu cầu true label | Chưa có | `docs/PRODUCTION_INFERENCE_SCHEMA_EXTENSION.md` |
| Learned ranking | Không có interaction/item labels | Chưa phù hợp | Không có data entities cần thiết |
| Causal recommendation | Không có treatment/propensity/outcome design | Không khả thi hiện tại | Dataset/schema inventory |

## 3. Luồng active

```text
source record features
  → prediction class + probability
  → risk band
  → confidence band từ max probability
  → feature-specific rule checks
  → chọn top risk signals
  → map risk → action strings
  → chọn 1 trong 3 template 4 tuần
  → explanation + disclaimer
  → ml_recommendations JSONB snapshot
  → structural_validity_metrics
```

Luồng này tạo một snapshot có thể đọc và audit, nhưng không có state transition sau khi recommendation được sinh.

### 3.1 Live DB snapshot tại thời điểm audit

Kiểm tra bằng transaction read-only:

- 17 experiment runs: 13 completed, 2 failed, 2 stale `running`;
- 1.027 predictions;
- 1.106 recommendations;
- policy counts: v1 = 553, v2 = 237, v3 = 316;
- active final run: 79 predictions, 79 v3 recommendations;
- một historical run có 79 predictions và 158 recommendations thuộc hai policy versions, xác nhận append-only multi-policy hoạt động.

Những counts này chỉ chứng minh persistence/versioning, không chứng minh recommendation quality.

## 4. Khoảng cách với “lộ trình học”

| Năng lực mong muốn | Hiện có | Khoảng trống |
|---|---|---|
| Hiểu tình trạng hiện tại | Class/risk/explanation | Chưa calibrated/abstention đầy đủ; risk và rule discordance chưa explicit |
| Yếu tố có thể điều chỉnh | Rule dùng một số context | Không tách predictor, modifiable factor, sensitive/non-actionable factor |
| Mục tiêu ngắn hạn | Câu văn chung | Không baseline/target/unit/deadline/owner |
| Mục tiêu dài hạn | Không | Chưa có hierarchy/milestone |
| Hành động cụ thể | Action strings | Không dosage/schedule/prerequisite/workload/evidence grade |
| Ưu tiên | Có thứ tự/top risks | Không policy giải quyết conflict, redundancy hoặc total workload |
| Dự kiến tác động | Không có evidence causal | Không được tự tạo impact score |
| Theo dõi tiến độ | Không | Không event/adherence/status |
| Điều chỉnh theo dữ liệu mới | Không | Không revision/supersedes relation |
| Cá nhân hóa | Một vài rule từ profile | Timing/missing semantics yếu; chỉ 3 plan templates |
| Cảnh báo sớm | Input active sau G2 | Không phải early warning |
| Hỗ trợ cố vấn | Payload có thể đọc | Không review workflow/audit trail |

Kết luận: output hiện tại là advice snapshot, không phải learning path lifecycle.

## 5. Findings safety và correctness

### 5.1 Missing/invalid feature tạo false risk — High, Confirmed

Các helper mặc định missing/invalid numeric về giá trị như G1/G2 = 0, `studytime = 1` (`src/recommendation.py:74-79,118-149`). Điều này biến “không biết” thành “điểm rất thấp/thời gian học thấp”, rồi sinh recommendation tự tin.

**Yêu cầu sửa:** missing/stale/invalid phải thành explicit `unknown`; nếu feature required thì abstain hoặc chuyển human review. Không impute bằng giá trị có ý nghĩa policy.

### 5.2 Confidence ngoài miền vẫn pass — High, Confirmed

Input confidence `-0,1` hoặc `1,2` không bị reject. Điều này làm confidence band/explanation vô nghĩa.

**Yêu cầu sửa:** schema validation `[0,1]`, finite, probability vector sum≈1, class index hợp lệ; bind calibration version.

### 5.3 Feature governance không nhất quán — High, Confirmed

Policy dùng `internet`, trong khi `config/feature_availability.yaml` đánh dấu sensitive/policy-excluded và timing unknown. Đây là policy drift giữa recommender và prediction/data governance.

**Yêu cầu sửa:** một feature registry dùng chung; policy build fail nếu rule truy cập forbidden feature. Nếu muốn dùng `internet` cho resource-access support, cần approval fairness/privacy riêng và không diễn giải là nguyên nhân học kém.

### 5.4 Tỷ lệ `absences/studytime` sai đơn vị — High, Confirmed

`absences` là count, `studytime` là ordinal category, không phải số giờ liên tục. Chia hai biến tạo quantity không có đơn vị khoa học ổn định.

**Yêu cầu sửa:** bỏ ratio; dùng rule đã được expert xác nhận trên từng observation với semantics rõ.

### 5.5 “No contradiction” không kiểm contradiction — High, Confirmed

Implementation chỉ kiểm tra `risk_code` trùng (`src/recommendation.py:464-472`). Nó không phát hiện action trùng text, semantic conflict, excessive workload, contraindication hay inconsistent priorities.

Audit thấy 12/79 recommendation có duplicate action text nội bộ, thường vì `prior_grade_gap` và `failure_history` map cùng action. Artifact `no_contradiction_rate=1,0` đúng theo implementation hẹp, nhưng tên metric dẫn tới overclaim.

**Yêu cầu sửa:** đổi tên hiện tại thành `unique_risk_code_rate`; thêm semantic rule graph, duplicate action IDs, incompatible actions, workload và prerequisite checks.

### 5.6 Cùng policy version có payload envelope khác nhau — Medium, Confirmed

Main generator không truyền full probability vector; post-hoc materializer có truyền. Cùng string policy version có thể sinh explanation khác nhau.

**Yêu cầu sửa:** schema version + canonical builder + input snapshot hash; mọi đường sinh phải đi qua cùng factory.

### 5.7 Verifier mâu thuẫn append-only versioning — Medium–High, Confirmed

Migration 002 cho phép nhiều policy version cho một prediction, nhưng builder/verifier so tổng recommendation count với prediction count. Thêm policy version hợp lệ có thể làm verifier fail.

**Yêu cầu sửa:** filter `(run_id, policy_version)` hoặc kiểm uniqueness theo cặp; không giả định tổng rows = predictions.

### 5.8 Production schema không phù hợp — High, Confirmed

`ml_predictions.true_label NOT NULL` phù hợp experiment, không phù hợp inference khi outcome chưa đến. Không có endpoint/service hay snapshot/outcome separation.

**Yêu cầu sửa:** tách `prediction_snapshots` và `prediction_outcomes`; không tái dùng experiment table làm production table.

### 5.9 Domain mismatch — High khi triển khai

Dữ liệu là học sinh phổ thông Bồ Đào Nha; use case mong muốn là sinh viên/cố vấn đại học Việt Nam. Family/teacher/support actions không tự động phù hợp. Expert localization và prospective data đúng domain là bắt buộc.

## 6. Phân biệt các khái niệm bắt buộc

| Khái niệm | Định nghĩa áp dụng | Điều không được làm |
|---|---|---|
| Dự đoán | Ước lượng class/probability của G3 từ input tại cutoff | Không gọi là nguyên nhân |
| Giải thích prediction | Mô tả signal/model basis và uncertainty | Không đổi feature attribution thành intervention effect |
| Risk signal | Observation liên quan tới nguy cơ | Không mặc định modifiable |
| Modifiable factor | Yếu tố có thể thay đổi, đo đúng thời điểm và có policy approval | Không suy ra chỉ từ correlation |
| Recommendation | Advice được policy/expert tạo, có scope và safety guard | Không gọi là causal treatment nếu chưa có design |
| Correlation | Quan hệ quan sát | Không dùng để hứa tác động |
| Causal effect | Thay đổi outcome do intervention | Chỉ claim khi randomized/causal design hỗ trợ |
| Sensitive/non-actionable | Giới tính, tuổi, lịch sử cố định, điểm quá khứ... | Không khuyên người học “thay đổi” hoặc phạt theo thuộc tính này |

G1/G2 và failures có thể dùng làm context/risk history, nhưng không phải intervention target. Recommendation nên gắn với hành động có thể thực hiện, được expert phê duyệt, và ghi rõ evidence grade.

## 7. Kiến trúc recommendation phù hợp dữ liệu hiện có

Không nên xây learned recommender hoặc causal policy ngay. Kiến trúc cân bằng nhất:

```text
Production prediction snapshot
  → calibrated risk + uncertainty + abstention
  → temporally valid observations
  → feature governance: modifiable / contextual / forbidden / stale
  → expert-approved policy registry
  → risk signals + prediction–rule discordance
  → structured goal generator
  → versioned action catalog + workload/prerequisite checks
  → explanation with evidence grade and non-causal scope
  → advisor approve / modify / reject
  → student/advisor follow-up + adherence + adverse-event notes
  → immutable plan revision / new prediction
  → outcome and safety evaluation
```

### 7.1 Schema tối thiểu đề xuất

| Entity | Field cốt lõi | Mục đích |
|---|---|---|
| `prediction_snapshots` | model/version/input hash/probabilities/calibration/uncertainty/abstain | Production inference không cần true label |
| `prediction_outcomes` | snapshot_id/outcome/observed_at/source | Outcome đến sau |
| `recommendation_policies` | version/code commit/policy hash/feature-contract hash/domain/status/approver | Provenance và approval |
| `policy_rules` | prerequisites/missing behavior/contraindication/evidence grade | Rule governance |
| `action_catalog` | action ID/dosage/workload/prerequisite/domain/evidence | Tránh text tự do trùng/mâu thuẫn |
| `recommendation_instances` | snapshot/policy/input hash/status | Immutable instance |
| `risk_signals` | observation/freshness/modifiability/confidence/source | Tách fact khỏi advice |
| `learning_goals` | baseline/target/unit/deadline/owner/status | Goal có thể đo |
| `plan_actions` | goal/action/schedule/owner/status | Lộ trình thực thi |
| `advisor_reviews` | approve/modify/reject/reason/reviewer/time | Human gate |
| `followup_events` | adherence/new observation/adverse event/notes | Vòng phản hồi |
| `plan_revisions` | supersedes/version/reason | Adaptation có audit trail |
| `recommendation_outcomes` | goal/process/academic outcome | Evaluation |

### 7.2 Policy decisions bắt buộc

- Missing/stale input → unknown/abstain, không 0/1 mặc định.
- Confidence phải gắn calibration version, full vector và uncertainty metric.
- Tách risk indicator, modifiable factor và action.
- Feature forbidden không được policy truy cập.
- Prediction–rule discordance là state riêng cần human review.
- Action bắt buộc có ID, workload, prerequisite và contraindication.
- Không tự động active plan trước advisor approval trong giai đoạn pilot.
- Mọi explanation ghi `non_causal` trừ khi evidence design thực sự thay đổi.

## 8. Evaluation protocol

### Gate R0 — Contract freeze

- Freeze population, domain, cutoff, feature freshness/modifiability/sensitivity.
- Freeze policy/action catalog/hash trước evaluation.
- Không iterate rule bằng 79 historical observed cases.

### Gate R1 — Technical safety

Điều kiện bắt buộc:

- 100% reject/abstain confidence/probability ngoài miền;
- 100% missing/stale required inputs không bị biến thành risk giả;
- 0 target/forbidden-feature access;
- deterministic/idempotent dưới cùng input + policy hash;
- 0 duplicate action IDs, incompatible action pairs và prerequisite violation;
- workload trong giới hạn expert-defined.

### Gate R2 — Expert content validation

- Sample phân tầng theo risk band × uncertainty × risk code × discordance × missingness, không lấy 12 rows đầu tiên.
- Ít nhất 2 chuyên gia độc lập; blind true G3 khi đánh giá advice.
- Rubric 1–5: relevance, safety, feasibility, specificity, workload, explanation, fairness.
- Báo weighted kappa hoặc Krippendorff alpha.
- Gate đề xuất: không critical unsafe case; median safety và relevance ≥4; agreement ≥0,60; mọi contraindication được xử lý.

Không tự tạo “quality score” nếu expert ratings chưa có.

### Gate R3 — Shadow/usability pilot

Cố vấn xem recommendation nhưng hệ thống không tự kích hoạt hành động. Đo:

- accept/modify/reject và reason;
- review time;
- comprehension;
- action workload;
- subgroup disparity;
- recommendation stability khi input thay đổi nhỏ hợp lý.

### Gate R4 — Prospective effectiveness

Pre-register usual-care comparator, stepped-wedge hoặc RCT nếu đạo đức/khả thi. Process outcomes (approval, adherence, goal attainment, adverse event) là primary ban đầu; academic outcome là secondary. Chỉ claim causal khi design và confidence interval hỗ trợ.

### Gate R5 — Learned recommender

Chỉ mở khi có đủ:

- exposure logs;
- expert/student acceptance và modification;
- item/action catalog;
- adherence/follow-up;
- outcomes;
- known assignment propensity/overlap nếu đánh giá causal.

Khi đó mới cân nhắc supervised ranking, rồi IPS/DR/uplift/contextual bandit dưới safety constraints. Dataset UCI hiện tại không đủ.

## 9. Metrics phù hợp theo giai đoạn

| Giai đoạn | Metrics hợp lệ |
|---|---|
| Unit/structural | Schema validity, coverage, determinism, idempotency, missing/forbidden rejection, duplicate/conflict/workload rate |
| Expert review | Relevance, safety, feasibility, specificity, workload, explanation, fairness, agreement |
| Shadow pilot | Accept/modify/reject, review time, comprehension, subgroup disparity, stability |
| Prospective | Adherence, goal attainment, adverse events, retention/process outcomes, academic outcome với uncertainty |
| Learned ranking tương lai | Ranking metrics + off-policy IPS/DR chỉ khi logging propensity hợp lệ |

“Diversity” không tự động là tốt: một action catalog nhỏ nhưng đúng/safe có thể tốt hơn advice đa dạng nhưng vô căn cứ. “Counterfactual plausibility” không đồng nghĩa causal effect.

## 10. Lộ trình trưởng thành

1. **Level 0 — current:** deterministic policy snapshot.
2. **Level 0.5 — đề xuất khóa luận:** governed expert-guided policy + structured goals/actions + human review + follow-up/revision.
3. **Level 1 — supervised relevance:** học ranking từ labels độc lập của expert, chưa claim outcome effect.
4. **Level 2 — feedback learning:** học từ exposure/acceptance/adherence/outcome logs.
5. **Level 3 — causal/bandit:** chỉ sau randomized/known propensity design và ethics/safety approval.

Phạm vi cân bằng cho khóa luận là Level 0.5. Nó có giá trị thực tế, kiểm chứng được và không đòi dữ liệu tương tác chưa tồn tại.

## 11. Kết luận

Recommendation hiện có là nền prototype tốt về deterministic output, storage và explanation, nhưng “lộ trình” mới là 3 text templates. Ưu tiên không phải thêm AI sinh lời khuyên, mà là sửa safety semantics, thiết kế policy registry, goal/action lifecycle, human review và evaluation thật. Chỉ sau khi có interaction/outcome data mới hợp lý gọi hệ thống là learned recommender; chỉ sau causal design mới được gọi advice là intervention có hiệu quả.
