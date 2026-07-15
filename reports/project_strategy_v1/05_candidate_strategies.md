# 05 — Các chiến lược ứng viên

## 1. Nguyên tắc chung không thay đổi giữa các chiến lược

Mọi chiến lược chỉ được triển khai sau phê duyệt và phải tuân thủ:

1. 79 historical records là `legacy_heldout_observed`, không dùng để chọn/xác nhận.
2. Selection chỉ dựa trên immutable 316-record development set và outer folds V2.
3. ML và DL trong một comparison phải dùng cùng feature contract, folds, metrics và search budget có thể giải thích.
4. Resolved config phải giống nhau ở inner search, outer evaluation và final estimator.
5. Preprocessing, calibration, threshold, resampling và loss tuning chỉ fit trong training/inner-OOF.
6. Macro-F1 là primary; ordinal, calibration, class-wise, stability và compute là guardrails.
7. CNN–BiLSTM không bị loại trước ablation, nhưng cũng không được ép làm champion.
8. Recommendation chỉ là non-causal expert-guided advisory cho đến khi có intervention evidence.

## 2. So sánh nhanh

| Tiêu chí | A — Conservative | B — Balanced | C — Ambitious |
|---|---|---|---|
| Trọng tâm | Sửa estimator, rerun control gần hiện tại | Sửa pipeline + small ordinal candidates + fair ML + lifecycle recommendation | Context/multitask/hybrid và mở rộng dữ liệu |
| Thay đổi prediction | Thấp | Trung bình, có kiểm soát | Cao |
| Thay đổi recommendation | Safety hardening tối thiểu | Structured goals/actions + human lifecycle | Có thể learned ranking sau khi có data |
| Compute | Thấp–trung bình | Khoảng một fair benchmark hiện có | >2 lần Balanced, chưa tính thu thập dữ liệu |
| Scientific risk | Thấp sau correction | Thấp–trung bình | Cao |
| Leakage/external validity risk | Thấp | Thấp nếu giữ strict G1/G2 | Cao do context/new data/domain alignment |
| Khả năng neural vượt ML | Thấp | Thấp–trung bình, phải kiểm chứng | Không rõ; complexity không bảo đảm cải thiện |
| Khả năng tìm champion tốt nhất | Trung bình | Cao trong phạm vi dữ liệu hiện có | Có thể cao hơn nếu có data mới, nhưng không chắc trong thesis |
| Khả năng bảo vệ hội đồng | Cao nhưng novelty vừa | **Cao nhất** | Trung bình nếu scope/protocol trượt |
| Khuyến nghị | Fallback | **Phương án chính** | Chỉ conditional sau gates/data mới |

## 3. Chiến lược A — Conservative

### 3.1 Mục tiêu

Khôi phục tính đúng đắn và tái lập với thay đổi nhỏ nhất; giữ compact CNN–BiLSTM gần current architecture làm research model, đồng thời so với G2/RF/SVM trên strict G1/G2.

### 3.2 Thiết kế prediction

| Hạng mục | Thiết kế |
|---|---|
| Architecture | Corrected compact CNN–BiLSTM; cùng một training factory cho inner/outer/final |
| Dữ liệu | 316 development rows; 5 immutable outer folds |
| Feature contract | Late-stage `[G1,G2]` duy nhất |
| CNN | Giữ kernel 1 như historical control; một paired ablation kernel hợp lý nếu budget cho phép |
| Loss | Unweighted cross-entropy |
| Imbalance | `none` mặc định; không SMOTE/ADASYN |
| Training | `drop_last=False`; bỏ dead SWA hoặc replay đầy đủ; epoch/refit policy nhất quán |
| Decision | Argmax; calibration chỉ nếu inner-OOF test cho tín hiệu ổn định |
| Model selection | Nested 5×3; budget thấp hơn Balanced; resolved config serialized |
| Baselines | G2 rule, RF, SVM; có thể thêm multinomial logistic như sanity |
| Ablations | Bug-compatible vs fixed config; scheduler/refit policy; drop_last; CNN-only/BiLSTM-only nếu parameter-matched |
| Metrics | Macro-F1, class-wise recall/F1, QWK, ordinal MAE, one/two-step, Brier/ECE, params/runtime |

### 3.3 Recommendation

Giữ policy v3 concept nhưng sửa ngay:

- missing/invalid → unknown/abstain;
- confidence/probability validation;
- bỏ `absences/studytime` ratio;
- cấm feature không có trong governance contract;
- action IDs và duplicate/conflict checks;
- canonical payload builder;
- policy registry tối thiểu.

Chưa xây follow-up schema đầy đủ; expert review vẫn là deliverable bắt buộc trước claim usability.

### 3.4 Lợi ích và rủi ro

**Lợi ích:** ít thay đổi, dễ cô lập effect, nhanh có một benchmark đúng, khả năng bảo vệ protocol cao.

**Rủi ro khoa học:** vẫn dùng inductive bias không thật sự phù hợp L=2; có thể chỉ xác nhận CNN–BiLSTM thua baseline đơn giản. Novelty về model thấp.

**Leakage risk:** thấp vì không mở context. Rủi ro chính là tái sử dụng 79 hoặc calibration sai tầng; cả hai bị cấm bằng gate.

**Khả năng vượt ML:** thấp. Strategy này tối ưu correctness, không tối ưu hypothesis space.

**Implementation complexity:** thấp–trung bình. **Compute:** khoảng 0,3–0,5 workload Balanced tùy search budget.

### 3.5 File dự kiến phải sửa sau approval

- `src/model_selection.py`
- `src/train_pipeline.py`
- `scripts/run_pipeline.py`
- `scripts/run_fair_model_comparison.py`
- `tests/test_model_selection_contract.py`
- `tests/test_fair_model_comparison.py`
- `tests/test_scientific_protocol_v2.py`
- `src/recommendation.py`
- recommendation tests và policy registry migration mới

### 3.6 Thứ tự

Correctness tests → resolved config → estimator parity → fixed control → fair baselines → recommendation safety → expert review package.

## 4. Chiến lược B — Balanced (đề xuất chính)

### 4.1 Mục tiêu

Tìm model tốt nhất có cơ sở trong giới hạn N=316, L=2; giữ CNN–BiLSTM đúng vai trò research/control; đưa ordinal inductive bias phù hợp hơn vào candidate pool; hoàn thiện recommendation đến mức expert-guided learning-plan lifecycle có thể đánh giá.

### 4.2 Candidate architecture

| ID | Model | Vai trò |
|---|---|---|
| R0 | Deterministic G2 threshold rule | Reference mạnh, không fit |
| M1 | Random Forest | Strong nonlinear ML baseline |
| M2 | SVM | Strong margin baseline |
| N0 | Corrected compact CNN–BiLSTM nominal | Thesis/control architecture |
| N1 | Tiny nominal MLP trên vector G1/G2 | Kiểm tra giá trị của sequence machinery |
| N2 | Tiny ordered MLP/CORAL | Neural candidate chính, bias phù hợp ordinal target |
| A1 | Parameter-matched CNN-only | Fixed ablation |
| A2 | Parameter-matched BiLSTM-only | Fixed ablation |
| A3 | CNN-LSTM hoặc simplified recurrent control | Fixed ablation nếu budget |
| C1 | Gated residual + Huber auxiliary | Conditional, chỉ mở sau gate ordinal |

Không thêm engineered features riêng cho một family. Nếu dùng `G2−G1`, mean hoặc slope, đây là deterministic transforms của cùng G1/G2 và phải được cung cấp/cho phép công bằng cho mọi vector model; vì chúng đại số dư thừa với hai điểm, default là dùng raw G1/G2 để tránh feature theater.

### 4.3 Loss và output

- N0/N1: unweighted nominal cross-entropy.
- N2: cumulative/ordered binary loss với monotonic thresholds by construction.
- C1 nếu mở: ordered/classification loss + λ·Huber(raw G3 residual), target scaler fold-local, residual head zero-init/gated.
- Không MSE default vì negative tail và boundary classification là mục tiêu chính.
- λ chỉ được chọn trong inner CV từ grid nhỏ pre-registered.
- Final decision argmax/ordered class; calibration/temperature chỉ inner-OOF.

### 4.4 Feature tracks

**Track P — primary:** strict late-stage G1/G2. Tất cả candidates và fair conclusions dùng track này.

**Track E — early warning:** chưa chạy trong chiến lược này; chỉ thiết kế contract G1-only, cần model/data path riêng và không so số trực tiếp với late-stage.

**Track C — context:** đóng. Chỉ mở bằng change request khi capture-time contract được duyệt.

### 4.5 Imbalance

- Default `none` cho mọi model.
- Báo class-wise recall/F1, đặc biệt High.
- Nếu High recall không qua guardrail, mở một conditional ablation: class weight **hoặc** focal **hoặc** random oversampling, từng phương pháp riêng.
- Không SMOTE/ADASYN default; không double compensation.

### 4.6 Model-selection protocol

- Immutable outer 5 folds; inner stratified 3 folds.
- 30 trials/family trong main run; search spaces có complexity budget tương đương và được freeze.
- Neural correction study dùng 3 seeds; final stability dùng 5 new predeclared seeds.
- Một resolved-config object điều khiển inner, outer và final training.
- Fit preprocessing/resampling/calibration train-only.
- Report mọi trial state/pruning; không gọi “equal budget” nếu completed trials/compute khác nhau đáng kể.
- Selection rule pre-register: primary mean outer Macro-F1; paired delta + CI; guardrail class collapse/ordinal/calibration; tie ưu tiên model đơn giản/ổn định hơn.
- Không dùng fold mean như 5 independent observations để overstate significance.

### 4.7 Baselines, ablations và metrics

**Baselines:** G2 rule, RF, SVM; optional multinomial/ordinal logistic và Huber→bin sanity models nếu compute nhỏ.

**Mandatory ablations:** N0 vs CNN-only vs BiLSTM-only; nominal MLP vs ordered MLP; `drop_last` correction; scheduler/refit semantics; no imbalance vs conditional method nếu gate mở.

**Primary:** Macro-F1.

**Guardrails:** per-class F1/recall, confusion, QWK, ordinal MAE, one-/two-step error, raw boundary error, accuracy, NLL, Brier, ECE, seed SD/range, collapse count, params, fit time.

### 4.8 Recommendation architecture

Triển khai Level 0.5:

```text
calibrated prediction/abstention
→ governed risk observations
→ expert-approved policy registry
→ structured goals + action catalog
→ explanation/evidence grade
→ advisor approve/modify/reject
→ follow-up/adherence
→ immutable revision
→ outcome/safety evidence
```

Không học recommendation từ UCI. Không gắn causal impact score. Scope deliverable là schema + policy + safety tests + expert validation package + shadow-pilot design.

### 4.9 Lợi ích và rủi ro

**Expected benefit:** candidate ordinal nhỏ khớp dữ liệu hơn; candidate pool đủ rộng để tìm champion thực tế; CNN contribution được kiểm chứng thay vì giả định; recommendation đạt đúng mục tiêu hỗ trợ cố vấn ở mức có thể audit.

**Scientific risk:** nhiều candidate làm tăng multiplicity; residual/multitask có thể không cải thiện; ordinal có thể chỉ đổi error structure chứ không tăng Macro-F1. Giảm rủi ro bằng pre-registration, gate và fixed family count.

**Leakage risk:** thấp nếu strict G1/G2 và inner-only calibration. Context bị khóa.

**Khả năng neural vượt ML:** thấp–trung bình; G2/RF rất mạnh. **Khả năng chọn đúng model tổng thể:** cao hơn A vì không ép neural champion.

**Implementation complexity:** trung bình. **Compute:** xấp xỉ workload fair benchmark hiện có; chi tiết ở kế hoạch fit.

**Khả năng bảo vệ:** cao nhất vì mỗi component có hypothesis, control, gate và claim boundary rõ.

### 4.10 File dự kiến phải sửa sau approval

Prediction:

- `src/model_selection.py`
- `src/train_pipeline.py`
- `scripts/run_pipeline.py`
- `scripts/run_fair_model_comparison.py`
- `src/models/models.py`
- `src/models/ordinal_v3.py` hoặc module candidate mới
- `src/models/__init__.py`
- `src/data_pipeline.py` cho auxiliary raw target contract nếu conditional branch mở
- `src/evaluation/*`
- model/protocol/evidence tests

Recommendation:

- `src/recommendation.py`
- canonical recommendation schema/registry module mới
- migrations mới cho policy/goals/actions/reviews/follow-up/revisions
- materializer/bundle verifier
- recommendation safety/integration tests

Documentation chỉ cập nhật sau khi evidence mới đã validate; không ghi đè historical artifacts.

### 4.11 Thứ tự

Protocol freeze → correctness fixes/tests → fixed correction study → nested candidate comparison → conditional residual/imbalance gates → multi-seed full validation → freeze champion → recommendation safety/schema → expert review/shadow plan → external/prospective confirmation khi có data mới.

## 5. Chiến lược C — Ambitious

### 5.1 Mục tiêu

Mở rộng tín hiệu và architecture: multi-assessment sequence, temporally valid context fusion, ordinal-regression/residual multitask, uncertainty và OOF hybrid ML–DL. Recommendation có thể tiến tới learned relevance ranking sau khi thu thập feedback.

### 5.2 Điều kiện tiên quyết

Strategy C **không admissible trên current contract** nếu chưa có:

- entity/student ID;
- nhiều assessment timestamps hoặc semester events;
- prediction cutoff rõ;
- context snapshots với freshness/reference windows;
- domain phù hợp đại học Việt Nam hoặc validation transfer rõ;
- data governance/sensitivity approval;
- đủ sample size cho capacity tăng;
- interaction/exposure/adherence/outcome logs nếu muốn learned recommender.

Không được ghép `student-por`, xAPI hay nguồn khác chỉ để tăng N. Cần schema harmonization, entity resolution, domain shift study và group split.

### 5.3 Prediction architecture khả dĩ

```text
timestamped assessment sequence
  → compact temporal encoder (TCN/GRU/BiLSTM, ablated)

cutoff-valid context snapshot
  → small MLP with feature governance

fusion
  → ordered head
  + optional Huber residual/regression head
  + calibrated uncertainty/abstention
```

Một hybrid option là convex blend/stacking của best ML và best neural, nhưng meta-learner chỉ được fit từ inner-OOF predictions. Không fit stacking weights trên outer validation hoặc observed 79. Monotonic constraints có thể áp cho G2 nếu policy/empirical validation ủng hộ, nhưng không ép monotonic khi dữ liệu có legitimate declines.

### 5.4 Loss, protocol và metrics

- Ordered loss + optional Huber auxiliary; uncertainty objective chỉ khi calibration protocol rõ.
- Nested group/time-aware CV phù hợp grain mới; không tái dùng V2 folds nếu population/grain thay đổi.
- External/domain validation bắt buộc.
- Baselines mở rộng: strong tabular boosting, temporal naive/last-value, ordinal logistic, sequence models, champion Balanced.
- Ablations: sequence, context, fusion, ordinal, auxiliary, ensemble, uncertainty.
- Metrics thêm domain-shift, subgroup calibration/fairness, early-warning lead time và coverage-risk.

### 5.5 Recommendation

Sau Level 0.5 và khi có labels/logs:

- supervised action relevance ranking từ expert labels;
- feedback model từ accept/modify/adherence;
- causal/off-policy optimization chỉ khi treatment assignment propensity được biết và ethics approval có hiệu lực.

LLM/free-text generation không thay thế action catalog và safety policy.

### 5.6 Lợi ích và rủi ro

**Lợi ích kỳ vọng:** dữ liệu nhiều thời điểm/context hợp lệ mới thật sự tạo headroom cho temporal/context architecture và early warning; feedback data mở đường cho personalization thật.

**Scientific risks:** domain mismatch, entity leakage, temporal leakage, overfitting, multiple comparisons, causal overclaim, scope explosion. Complexity cao không bảo đảm vượt G2/RF trên current dataset.

**Compute:** >2× Balanced cho prediction, cộng chi phí data engineering/expert/prospective study. **Disk:** lớn hơn do nhiều OOF/calibration/model snapshots. **Implementation:** cao.

**Khả năng vượt ML:** không thể ước lượng trước data audit mới. Trên current N/L, không có cơ sở cho rằng cao hơn Balanced.

**Khả năng bảo vệ:** tốt chỉ nếu data mới và protocol được hoàn thiện; nếu không, thấp hơn B vì quá nhiều moving parts.

### 5.7 File dự kiến

Ngoài toàn bộ file của B: schema/data ingestion mới, group/time split protocol, context feature registry, fusion models, interaction/outcome tables, privacy/ethics docs và deployment/shadow infrastructure.

### 5.8 Thứ tự

Data acquisition protocol → ethics/governance → entity resolution → temporal contract → new split protocol → simple baselines → fusion candidates → OOF hybrid → prospective recommendation logs → learned ranking gate.

## 6. Lựa chọn đề xuất

**Đề xuất phê duyệt Strategy B — Balanced**, với hai giới hạn cứng:

1. Residual/multitask là conditional, không phải mandatory centerpiece.
2. Context/hybrid-data Strategy C đóng cho đến khi data contract mới được phê duyệt.

Lý do:

- A sửa correctness nhưng có nguy cơ chỉ tái xác nhận một architecture không khớp dữ liệu.
- B đưa vào inductive bias ordinal và small-model controls phù hợp N/L, nhưng vẫn giữ CNN–BiLSTM để đáp ứng đề cương bằng evidence.
- B cho phép champion là model thật sự tốt nhất, nên tối ưu đồng thời điểm số và scientific validity.
- Recommendation Level 0.5 có giá trị thực tế và bảo vệ được mà không giả vờ có learned/causal data.
- C chỉ có giá trị khi dữ liệu thay đổi; triển khai ngay sẽ tăng risk nhiều hơn expected gain.

Phê duyệt B không đồng nghĩa phê duyệt sẵn model cuối. Nó chỉ phê duyệt **candidate set, protocol, budgets và gates**; champion vẫn phải do evidence sau này quyết định.
