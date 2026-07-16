# 02 — Đối chiếu yêu cầu khóa luận và khoảng trống triển khai

## 1. Cách chấm mức hoàn thành

| Mức | Ý nghĩa |
|---|---|
| Hoàn thành | Có code active, test và evidence phù hợp claim |
| Hoàn thành một phần | Có nền tảng đúng nhưng thiếu một hoặc nhiều gate khoa học/kỹ thuật |
| Prototype | Có code/artifact minh họa, chưa tích hợp hoặc chưa validation |
| Chưa có | Không có implementation/evidence cần thiết |
| Evidence không hợp lệ | Có output nhưng protocol/provenance không cho phép dùng cho claim tương ứng |

Severity đánh giá theo tác động đến khả năng bảo vệ khóa luận, không chỉ theo độ khó kỹ thuật.

## 2. Ma trận đối chiếu tổng thể

| Yêu cầu đề tài | Dự án đã triển khai | Mức hoàn thành | Evidence | Vấn đề còn thiếu | Severity | Đề xuất khắc phục |
|---|---|---|---|---|---|---|
| Dataset được xác định và tái lập | `student-mat`, raw/DB/V2 hash khớp, 395 records | Hoàn thành một phần | DB live; V2 manifest; raw SHA-256 | Raw bị gitignore; upstream DOI/acquisition/license chưa được freeze đầy đủ | Medium | Đóng gói source provenance manifest, archive checksum và acquisition metadata |
| Data quality cơ bản | 0 null, 0 exact/profile duplicate, 0 enum/range violation đã kiểm | Hoàn thành | Read-only profile | Không có entity ID để xác minh cùng người qua nguồn/dataset | Medium | Ghi giới hạn; bắt buộc entity resolution + group split nếu ghép nguồn |
| Dataset versioning | `source_dataset_versions` và content hash | Hoàn thành một phần | migration 001; live DB | Identity ngoài artifact phụ thuộc DB-local ID | High | Dùng `(dataset_code, content_hash, source_row_number)` làm portable identity |
| Source-record lineage | Source → target → run → split → prediction → recommendation | Hoàn thành một phần | migrations 001–003 | Prediction true label chưa FK trực tiếp tới source target; raw payload còn G3 | High | Thêm DB constraints/trigger và target-free inference view |
| Target contract | Low 0–9, Medium 10–14, High 15–20 | Hoàn thành một phần | `src/config.py:59-62`; ingestion/migration | Bins là operational choice; rationale/sensitivity chưa pre-register; DB range check yếu | High | Centralize versioned target contract, rationale, boundary sensitivity trên development |
| Train/test separation | Immutable 316 development + historical 79 | Evidence không hợp lệ cho “unseen test” | `SCIENTIFIC_PROTOCOL_V2.md:5-7` | 79 đã được quan sát nhiều lần | Critical | Đổi nhãn `legacy_heldout_observed`; chỉ nested-CV cho selection; cần external/prospective test mới |
| Immutable outer folds | 5 stratified folds, manifest/checksum/ledger | Hoàn thành | Protocol V2; 316 records, 1.580 rows | Không có inner/early-stop record ledger | Medium | Materialize inner and early-stop memberships + checksum |
| Fold-local preprocessing | Scaler/selector/resampling fit train-only | Hoàn thành một phần | `src/model_selection.py`; tests | Actual context selector tồn tại nhưng model bỏ context; estimator parity còn lỗi | Medium | Contract-test toàn input DAG và resolved estimator |
| Không target leakage active | Official loader/drop path loại target | Hoàn thành một phần | loader/preprocessor code | Raw payload vẫn mang G3; consumer mới có thể bypass | Medium | Target-free DB view/role; explicit allowlist và integration assertion |
| Scenario pre-assessment | Config không cho feature | Prototype contract | `config/features_pre_assessment.yaml` | Active `StudentDataset` yêu cầu ≥2 sequence feature | High | Không claim implemented; chỉ xây khi có dữ liệu phù hợp |
| Scenario early warning | Config chỉ G1 | Prototype contract | `config/features_early_warning.yaml` | Active pipeline hard-code G1/G2, không chạy G1-only | High | Tách model/data contract theo scenario; cần evaluation riêng |
| Scenario late stage | G1/G2 input sau kỳ 2 | Hoàn thành | feature config + active model | Giá trị cảnh báo sớm thấp vì G2 sát outcome | Medium | Gọi đúng là late-stage prediction, không overclaim early warning |
| CNN trích xuất đặc trưng điểm | Conv1D kernel 1 | Prototype/giới hạn | `src/models/models.py:56-63` | Kernel 1 chỉ pointwise, không trộn G1/G2; contribution chưa được matched ablation | High | Giữ như control; tune/ablate matched-capacity CNN-only và kernel hợp lệ |
| BiLSTM học diễn biến thời gian | BiLSTM sau Conv | Prototype/giới hạn | `src/models/models.py:95-107` | Chỉ 2 timestep, không timestamp; không chứng minh long-term dynamics | High | Hạ claim xuống “two-assessment ordered signal”; cần dữ liệu nhiều thời điểm cho claim chuỗi mạnh |
| CNN–BiLSTM cạnh tranh ML | Có fair runner và historical artifacts | Evidence không hợp lệ | Fair artifact + config-loss bug | Inner/outer loss mismatch; provenance sai; G2 rule thiếu | Critical | Sửa resolved config, estimator parity, rerun cùng feature/folds/search |
| Strong ML baselines | RF, DT, SVM, GB, XGB; G2 rule artifact khác | Hoàn thành một phần | fair summary; historical ablation | Fair cross-family invalid; HGB cũ full-context không like-for-like; G2 rule chưa cùng runner | High | Pre-register G2, RF, SVM và neural candidates trên strict G1/G2 |
| Estimator selection = estimator final | Hai đường train khác nhau | Chưa đạt | `src/model_selection.py:248-289`; `scripts/run_pipeline.py:404-489` | Selection full refit; final giữ 15% và không refit | High | Dùng một estimator factory/training contract duy nhất |
| Scheduler/early-stop/refit consistency | Có scheduler/SWA và fixed trainer | Chưa đạt | `src/train_pipeline.py:122-190` | Epoch chọn dưới adaptive LR/SWA, refit fixed LR, SWA bị bỏ | High | Chọn policy nhất quán; test replay semantics |
| Imbalance handling hợp lệ | Class weight, oversampling, SMOTE/ADASYN paths | Hoàn thành một phần | `src/model_selection.py`; historical ablations | Fair resolved-config bug; synthetic grade trajectories khó biện minh; imbalance chỉ vừa | High | Default `none`; chỉ ablate từng phương pháp train-only khi guardrail minority thất bại |
| Ordinal modeling | `ordinal_v3.py` có head/prototype | Prototype | `src/models/ordinal_v3.py:9-87` | Không caller/export/test/artifact; chưa nested validation | Medium–High | Tích hợp tiny ordered MLP trước; QWK/ordinal MAE/step errors bắt buộc |
| Regression/multitask | Có primitives thử nghiệm | Prototype | `ordinal_v3.py`; current dataset target path | Raw G3 auxiliary target chưa được nối đúng; loss/scaler chưa fold-local end-to-end | High | Chỉ triển khai sau ordinal đơn; fold-local scaler, Huber, inner-CV lambda |
| Residual learning quanh G2 | Chưa có active model; đã chạy diagnostic nhỏ | Chưa có | Development diagnostic | Fixed Ridge/Huber residual làm Macro-F1 giảm so zero-residual | Medium | Không chọn làm centerpiece; chỉ conditional gated/zero-init Huber ablation |
| Context fusion | Không có active branch | Chưa có và đang bị data contract chặn | `feature_availability.yaml`; model sequence-only | Capture timing unknown; comparison fairness và leakage risk | High | Chỉ mở sau contract timestamp/freshness; track riêng, cùng feature contract cho ML/DL |
| Calibration/uncertainty | Có Brier/ECE code, argmax active, heuristic confidence bands | Hoàn thành một phần | selection manifest; recommendation policy | Chưa nested multi-seed calibration/abstention; confidence chỉ max softmax | Medium–High | Fit temperature/cutpoints inner-OOF; đánh giá NLL/Brier/ECE/coverage-risk |
| Metrics phù hợp ordinal | Macro-F1/accuracy/Brier/ECE một phần | Hoàn thành một phần | fair runner | Fair output thiếu QWK, ordinal MAE, one-/two-step và boundary errors | Medium | Bổ sung OOF metric suite pre-registered |
| Multi-seed stability | Một số historical ensemble/ablation | Hoàn thành một phần | final deep ablation artifacts | Fair deep outer chỉ một seed; seed không thể coi là independent folds | High | Predeclare ≥3 seeds khi search/correction, 5 seeds ở full validation |
| Reproducible model artifact | Checksummed final bundle | Hoàn thành một phần | 32/32 checksums match | Checkpoint map rỗng; commit/provenance mismatch ở selection/fair runs | High | Hash checkpoint, preprocessor, source tree/diff, environment và protocol |
| Unit/integration tests | 87 pass, 5 skip | Hoàn thành một phần | `pytest -q` | Không bắt estimator/config parity và recommendation safety/lifecycle | High | Bổ sung semantic contract + provenance tests trước rerun |
| Rule-based advice | 9 risk signals/actions, deterministic v3 | Hoàn thành ở mức prototype | `src/recommendation.py` | Một số rule thiếu semantic validity, missing behavior nguy hiểm | High | Hardening safety trước expert review |
| Recommendation policy versioning | String policy version + append-only DB | Hoàn thành một phần | migration 002; live v1/v2/v3 | Không có policy registry/hash/status/approver/evidence version | High | Thêm policy registry và immutable approval metadata |
| Recommendation storage | JSONB snapshot gắn prediction | Hoàn thành một phần | migrations; 1.106 rows live | Schema experiment yêu cầu true label, không phù hợp production inference; verifier mâu thuẫn multi-policy | High | Tách production prediction snapshot/outcome; filter verifier theo policy version |
| Recommendation explanation | Basis/evidence/rationale/disclaimer | Prototype tốt | `src/recommendation.py:325-338` | Cùng policy version có hai envelope probability khác nhau; noncausal scope cần enforce | Medium | Chuẩn hóa payload schema, input snapshot/hash và evidence grade |
| Goal planning | Chỉ câu văn/template 4 tuần | Prototype rất hạn chế | `src/recommendation.py:294-314` | Không baseline, target, unit, deadline, owner, status | High | Tạo `learning_goals` + `plan_actions` có cấu trúc |
| Follow-up/adaptation | Chỉ lời nhắc review | Chưa có | Không có table/code event | Không adherence, observation mới, feedback, revision state | High | Thêm follow-up events và immutable plan revisions |
| Expert validation | CSV 12 case để trống | Chưa có | final evidence review CSV | Không rating, không independent reviewers/agreement | High | Ít nhất 2 expert, stratified cases, rubric và agreement gate |
| Recommendation evaluation | Structural self-check | Prototype | `recommendation_evaluation.json`; policy code | Không ground truth/usefulness/safety; `no_contradiction` đặt tên quá rộng | High | Semantic safety tests + expert/shadow/prospective protocol |
| Learned recommender | Không có interaction/exposure/outcome data | Chưa có và chưa phù hợp | DB/schema inventory | Không item catalog, propensity, adherence, repeated outcome | High/Future | Giữ expert-guided policy; chỉ học ranking khi có feedback labels, causal khi có design phù hợp |
| Hỗ trợ cố vấn | Advice payload có thể đọc | Prototype | explanation + plan text | Không review/approve/modify/reject state và audit trail | High | Human-in-the-loop workflow và advisor review table |
| Phù hợp domain đại học Việt Nam | Dữ liệu Portuguese secondary school | Chưa được chứng minh | Dataset/docs limitations | External validity và tính phù hợp action chưa có | High | Expert localization + dữ liệu prospective đúng domain; giới hạn claim rõ |
| Documentation nhất quán | README/docs/report context phong phú | Hoàn thành một phần | docs inventory | Một số tài liệu gọi 79 locked, dùng số benchmark stale/unsupported, test counts lịch sử khác nhau | High | Sau approval, reconcile theo evidence hierarchy; không sửa hồi tố artifacts |

## 3. Đánh giá riêng theo hai mục tiêu đề tài

### 3.1 Mục tiêu mô hình dự đoán

**Đã có:** pipeline end-to-end, G1/G2 late-stage contract, CNN–BiLSTM active, ML baselines, nested-CV framework, DB evidence và test suite.

**Chưa đạt:** một comparison hợp lệ cho current estimator; unseen confirmation set; evidence CNN/BiLSTM contribution; ordinal model tích hợp; estimator parity; multi-seed stability và provenance đủ mạnh.

**Kết luận:** mục tiêu kỹ thuật đã có prototype mạnh, nhưng mục tiêu khoa học “cạnh tranh hoặc vượt ML” vẫn là giả thuyết. Không được xem fair summary hiện tại là câu trả lời cuối.

### 3.2 Mục tiêu hệ thống khuyến nghị

**Đã có:** deterministic risk/action policy, explanation, 3 template 4 tuần, policy version và PostgreSQL snapshots.

**Chưa đạt:** lifecycle của lộ trình, human approval, goals/action objects, follow-up, adaptation, expert validation, usefulness/safety evidence và production inference schema.

**Kết luận:** hiện tại đúng tên nhất là **“evidence-informed, rule-based advisory prototype”**. Không nên gọi là learned recommender, causal intervention engine hoặc lộ trình học thích ứng.

## 4. Khoảng cách giữa đề cương và dữ liệu thực tế

| Tuyên bố có thể bảo vệ | Tuyên bố chưa thể bảo vệ |
|---|---|
| Dùng hai điểm có thứ tự G1, G2 để dự báo late-stage G3 class | Học diễn biến qua nhiều học kỳ |
| So sánh CNN–BiLSTM với baseline dưới cùng feature contract sau khi sửa protocol | CNN–BiLSTM hiện đã vượt ML |
| G2 là predictor mạnh và là baseline bắt buộc | Deep residual chắc chắn tốt hơn G2 |
| Policy luật sinh advice có giải thích và lưu version | Recommendation cải thiện kết quả học tập |
| Advice mang tính hỗ trợ quyết định, không causal | Feature importance chứng minh yếu tố can thiệp |
| Kết quả mới được lựa chọn bằng immutable development nested-CV | 79 record là untouched final test |

Nếu đề cương bắt buộc cụm “CNN–BiLSTM”, có thể bảo vệ bằng cách giữ nó như kiến trúc nghiên cứu/control, chạy ablation công bằng, và báo cáo trung thực champion. Không nên ép CNN–BiLSTM thành final champion khi evidence không ủng hộ.

## 5. Thứ tự đóng khoảng trống

1. Quarantine 79 observed và fair DL rows; sửa claim boundary.
2. Sửa resolved-config bug, estimator parity, scheduler/refit semantics và provenance.
3. Bổ sung test chặn hồi quy các lỗi trên.
4. Rerun development-only comparison với G2, RF, SVM, tiny nominal/ordinal models và CNN controls.
5. Chỉ mở residual/multitask nếu ordinal đơn có tín hiệu; context chỉ khi timing contract được duyệt.
6. Hardening recommendation safety và tạo policy registry/lifecycle schema.
7. Expert validation → shadow pilot → prospective evaluation.
8. Chỉ xác nhận ngoài mẫu trên dữ liệu mới thực sự chưa quan sát.

Chiến lược cụ thể được trình bày tại [05_candidate_strategies.md](./05_candidate_strategies.md), kế hoạch fit tại [06_proposed_experimental_plan.md](./06_proposed_experimental_plan.md), và các quyết định cần phê duyệt tại [08_approval_summary.md](./08_approval_summary.md).
