# 01 — Kiểm toán repository, dữ liệu, lineage và evidence

**Ngày audit:** 2026-07-14  
**Phạm vi:** repository `prediction_student` tại `C:\Huflit\kltn`  
**Chế độ:** chỉ đọc đối với source code, database, artifacts và dữ liệu; chỉ thư mục báo cáo này được tạo  
**Kết luận trạng thái:** audit chiến lược, chưa phải nghiệm thu mô hình hay kết quả thực nghiệm mới

## 1. Kết luận điều hành

Repository đã có nền tảng kỹ thuật đáng kể: dữ liệu được version hóa trong PostgreSQL, có source-record lineage, outer-fold manifest bất biến, pipeline tiền xử lý theo fold, nested-CV framework, nhiều baseline, artifact bundle, recommendation policy có version và test suite đang xanh. Tuy nhiên, evidence hiện tại **chưa đủ để tuyên bố mô hình cuối đã được lựa chọn và xác nhận khoa học** vì bốn lý do ưu tiên:

1. Tập 79 bản ghi từng gọi là `locked test` đã được xem qua nhiều mô hình và phân tích. Chính `SCIENTIFIC_PROTOCOL_V2.md:5-7` xác nhận điều này. Từ nay nó chỉ là `legacy_heldout_observed`, không phải test chưa quan sát.
2. Fair benchmark làm rơi các tham số cố định của nhánh deep learning; inner search dùng CE không trọng số nhưng outer evaluation rơi vào class weight mặc định. Hai hàng CNN và kết luận ML–DL của run đó không hợp lệ theo protocol công bố.
3. Estimator dùng trong model selection hiện khác estimator tạo final model: selection chọn epoch rồi refit toàn fold, còn `scripts/run_pipeline.py` giữ lại 15% validation và không refit toàn development. Ngoài ra scheduler/SWA ở giai đoạn chọn epoch không được replay trong fixed-epoch refit.
4. Hệ thống khuyến nghị là policy theo luật, có version và lưu trữ, nhưng chưa có goal có cấu trúc, human approval được lưu, follow-up, feedback, revision, expert validation hay evidence hiệu quả can thiệp. Vì vậy chưa đủ gọi là “lộ trình học” hoàn chỉnh hay recommender học từ tương tác.

Định hướng khoa học phù hợp nhất là: **sửa protocol và estimator trước; giữ CNN–BiLSTM như một control bắt buộc phải kiểm chứng, nhưng cho phép một mô hình nhỏ/ordinal hoặc ML mạnh trở thành champion nếu evidence paired nested-CV ủng hộ; đồng thời nâng recommendation thành expert-guided decision support có vòng đời, không gắn nhãn causal.**

## 2. Trạng thái Git và môi trường

| Mục | Kết quả audit |
|---|---|
| Branch | `main` |
| Commit tại thời điểm audit | `6618fe48f8314a73fa0188a1fe8ee3bec5823045` |
| Commit message | `feat: add fair model comparison and remove legacy benchmarks` |
| Working tree trước audit | Đã dirty sẵn: `PROJECT.md`, `README.md` |
| Thay đổi do audit | Không sửa hai file trên; chỉ tạo `reports/project_strategy_v1/` |
| Python | 3.10.8 |
| PyTorch | 2.12, CPU build |
| scikit-learn | 1.7.2 |
| pandas / NumPy | 2.3.3 / 2.2.6 |
| Optuna / XGBoost | 4.8 / 3.2 |
| Full unit test | 87 passed, 5 skipped, 9,75 giây |

Hai thay đổi sẵn có trong `PROJECT.md` và `README.md` được xem là tài sản của người dùng và được bảo toàn. Audit không stage, commit hay reset bất kỳ nội dung nào.

## 3. Bản đồ thành phần chính

| Lớp | Thành phần chính | Vai trò thực tế |
|---|---|---|
| Cấu hình | `src/config.py`, `config/features_*.yaml`, `config/feature_availability.yaml` | Bins, scenario và feature timing contract |
| Dữ liệu | `src/data_pipeline.py`, `src/postgres_data_source.py` | Load DB, loại target, xử lý theo fold, tạo tensor G1/G2 |
| Lineage DB | `database/migrations/001_*.sql` đến `003_*.sql` | Version dữ liệu, source records, target table, run/split/prediction/recommendation ledger |
| Mô hình active | `src/models/models.py` | Conv1D → BatchNorm → BiLSTM → nominal softmax |
| Mã ordinal thử nghiệm | `src/models/ordinal_v3.py` | Prototype ordered/ordinal/multitask chưa được tích hợp |
| Huấn luyện | `src/train_pipeline.py` | Early stopping, scheduler, SWA và fixed-epoch trainer |
| Model selection | `src/model_selection.py` | Inner search, outer refit, preprocessing/resampling train-only |
| Pipeline cuối | `scripts/run_pipeline.py` | Train ensemble, evaluate historical holdout, sinh recommendation, persist DB |
| Fair comparison | `scripts/run_fair_model_comparison.py` | So sánh ML/DL trên G1/G2, nested 5×3, 30 trials |
| Evaluation | `src/evaluation/*` | Metrics, protocol, persistence, artifact validation |
| Recommendation | `src/recommendation.py`, materializer và migrations | Policy luật v1–v3, explanation, JSONB snapshots |
| Tests | `tests/` | 92 collected; 87 pass, 5 skip |
| Evidence | `artifacts/`, `reports/`, `docs/report_context/` | Snapshot lịch sử, final bundle, protocol và diễn giải |

## 4. Luồng dữ liệu và ranh giới target

```text
student-mat.csv
  → source_dataset_versions
  → source_records(raw_payload, source_row_number)
  → source_record_targets(G3 raw + class encoded)
  → immutable development membership / outer folds
  → fold-local preprocessing + model selection
  → OOF predictions / historical evaluation
  → ml_experiment_runs
  → ml_run_record_splits
  → ml_predictions
  → ml_recommendations(policy_version, learning_path JSONB, explanation JSONB)
```

Official application paths loại target khỏi feature frame tại `src/postgres_data_source.py:299-313` và `src/data_pipeline.py:273-275,408-413`. Không tìm thấy active target leakage trong đường huấn luyện được audit. Tuy nhiên, `source_records.raw_payload` vẫn chứa `G3` cho 395/395 record; ranh giới least-privilege còn yếu nếu một consumer mới bypass allowlist.

### 4.1 Grain và tính chất chuỗi

- Một dòng là một hồ sơ học sinh–môn Toán trong `student-mat`, không phải event theo thời gian.
- Không có `student_id`, `course_id`, `term_id`, assessment timestamp hoặc duration.
- `G1`, `G2`, `G3` có thứ tự kỳ 1 → kỳ 2 → cuối kỳ, nhưng đầu vào active chỉ là `[G1,G2]`, tức chuỗi dài 2.
- Chuỗi này có thể biểu diễn mức hiện tại, chênh lệch và một bước xu hướng. Nó không hỗ trợ claim “diễn biến qua nhiều học kỳ” hay long-term temporal dynamics.
- Dự báo sau G2 là late-stage prediction, không phải cảnh báo sớm.

### 4.2 Hồ sơ dữ liệu

| Kiểm tra | Toàn bộ 395 dòng | Development 316 dòng |
|---|---:|---:|
| Null cells | 0 | 0 |
| Exact duplicate rows | 0 | 0 |
| Duplicate profile bỏ G1/G2/G3 | 0 | 0 |
| Class Low / Medium / High | 130 / 192 / 73 | 104 / 154 / 58 |
| G1 range | 3–19 | 4–19 |
| G2 range | 0–19 | 0–19 |
| G3 range | 0–20 | 0–20 |
| Corr(G2,G3) | 0,9049 | 0,9048 |

Target bins là lựa chọn vận hành của dự án, không phải chuẩn phổ quát:

- Low: 0–9;
- Medium: 10–14;
- High: 15–20.

Có 144/395 bản ghi toàn bộ, và 114/316 bản ghi development, nằm đúng các điểm biên 9, 10, 14 hoặc 15. Vì vậy classification nhạy với bin policy; cần pre-register lý do sư phạm và báo thêm ordinal/regression metrics, không tối ưu biên bằng 79 mẫu đã quan sát.

### 4.3 Feature timing contract

Contract hiện tại là bảo thủ và đúng hướng:

| Scenario | Feature được phép | Trạng thái thực thi |
|---|---|---|
| Pre-assessment | Không có | Chỉ là contract; active dataset/model không chạy được với 0 sequence feature |
| Early warning | G1 | Chỉ là contract; `StudentDataset` yêu cầu ít nhất 2 sequence columns |
| Late stage | G1, G2 | Track active duy nhất |

`studytime`, `failures`, `schoolsup`, `famsup`, `absences` và các biến questionnaire chưa có capture timestamp/reference window đáng tin cậy. Theo `config/feature_availability.yaml`, trạng thái `unknown` là forbidden. Vì thế context branch hiện **bị chặn bởi data contract**, không phải chỉ thiếu code.

## 5. PostgreSQL lineage và integrity

### 5.1 Điểm mạnh đã xác minh

- Một dataset version live: `student-mat`, `dataset_version_id=1`.
- 395 source records liên tục từ row 0 đến 394 và 395 target rows.
- Raw SHA-256, DB dataset hash và V2 manifest cùng giá trị `e47f9ee225e1ee6e69b7564e6dac7123e80b8486677fe111f351964cef5dec80`.
- Không orphan source/target/split; không mismatch target raw ↔ target table ↔ encoded.
- 316 development records, 1.580 outer assignment rows; mỗi record làm validation đúng một lần.
- Outer folds có kích thước 64/63/63/63/63 và phân bố lớp gần cân bằng theo stratification.
- Tái tạo độc lập split 80/20, `random_state=42`, stratified cho đúng membership 316/79.
- DB recommendation là append-only theo `(prediction_id, policy_version)`.

### 5.2 Khoảng trống lineage

| Vấn đề | Evidence | Ý nghĩa |
|---|---|---|
| Identity dùng `dataset_version_id=1` cục bộ | `src/evaluation/protocol.py:56-58` | Manifest dễ gãy khi ingest vào DB mới theo thứ tự khác |
| Inner folds không được materialize | `src/model_selection.py:125-128,367-405`; manifest có `inner_fold:null` | Chỉ tái tạo gián tiếp từ code/seed, chưa có ledger record-level |
| Early-stop split không có ledger | Training code | Khó audit estimator chính xác ở mức record |
| Target ingest `ON CONFLICT DO NOTHING`, sau đó chỉ đếm dòng | `src/postgres_data_source.py:118-132` | Target sai nhưng đủ số dòng có thể lọt qua trên DB không sạch |
| DB chỉ check encoded target ≥0 | migration 003 | Không ràng buộc ≤2, raw G3 0–20 và consistency contract |
| `ml_predictions.true_label` không FK tới source target | migration 001 | Application kiểm tra tốt hơn DB, nhưng DB invariant còn yếu |
| Không có migration ledger | Live schema | Khó chứng minh migration nào đã chạy và checksum nào |
| Hai run còn `running` từ 2026-07-05 | Live DB read-only profile | Thiếu timeout/recovery lifecycle |
| Ingestion contract còn mỏng | `src/evaluation/evaluation.py:66-77` | Chưa freeze dtype, nullability, enum/range, upstream provenance |

## 6. Evaluation protocol và tình trạng evidence

### 6.1 Thứ bậc evidence áp dụng cho báo cáo này

| Cấp | Có thể dùng để làm gì | Evidence hiện có |
|---|---|---|
| A — Unseen external/prospective | Xác nhận cuối sau freeze | **Chưa có** |
| B — Corrected nested-CV trên immutable development | Chọn và so sánh model | Framework có, nhưng headline runs chưa khớp estimator/provenance hiện tại |
| C — Historical observed holdout | Bối cảnh lịch sử, sanity only | 79 mẫu; không được dùng lại để chọn/xác nhận |
| D — Small diagnostic/smoke | Xác minh bug/khả thi | One-fold, residual stats, shape/criterion probes |
| E — Documentation claim | Chỉ dùng khi reconcile với code/artifact | Một số claim đang stale hoặc mâu thuẫn |

### 6.2 Tập 79 không còn là locked test

`SCIENTIFIC_PROTOCOL_V2.md:5-7` ghi rõ predictions của 79 record đã được xem cho CNN, G2 rule, LR, HGB, ablations, imbalance và multi-seed. Do đó:

- Không được gọi nó là untouched/locked test trong claim mới.
- Không chạy lại nó để quyết định architecture, seed, threshold, calibration, policy hoặc dừng sớm.
- Không dùng nó làm Phase F hiện tại.
- Phase F chỉ được mở khi có một dataset bên ngoài/prospective mới, được đăng ký và khóa trước khi xem nhãn.
- Các tài liệu hiện vẫn gọi 79 là locked test cần được sửa **sau khi chiến lược được phê duyệt**, không sửa ở giai đoạn audit này.

### 6.3 Headline model-selection artifact

`artifacts/model_selection/nested-full-20260710/selected_config.json` ghi outer Macro-F1 `0,878089 ± 0,044829`, 13.059 tham số, G1/G2, seed 42, argmax, không calibration/SMOTE/class weighting theo manifest.

Giá trị này chỉ là **historical estimate**, không phải con số chính thức cho estimator V2 hiện tại:

- Run ghi commit `74e43fc...`; commit đó chưa có full-fold refit correction hiện tại.
- Selection artifact có config thuộc giai đoạn code muộn hơn commit đã ghi; provenance không khớp source tree.
- Current model selection chọn epoch rồi refit full outer train; historical run train trên khoảng 85% outer train.
- Final pipeline vẫn dùng cách 85/15 và không full refit, nên estimator selection/deployment tiếp tục khác nhau.

### 6.4 Fair ML–DL artifact

Artifact `artifacts/baseline_comparison/fair-model-comparison-full/summary.csv` có hướng kết quả: RF/DT/SVM/GB khoảng 0,887–0,891, CNN–BiLSTM khoảng 0,838, CNN–LSTM khoảng 0,797. Không được dùng các con số này để kết luận chính thức vì:

1. `student_search_space(fair_comparison=True)` đặt `class_weight_mode="none"` như constant, không phải trial suggestion.
2. Search chỉ trả `study.best_params`, làm rơi constants.
3. Outer refit nhận params thiếu key; `_criterion` mặc định thành `balanced`.
4. Protocol artifact lại tuyên bố không imbalance handling cho mọi model.

Đây là **Critical correctness issue**. Các hàng ML riêng lẻ vẫn là evidence kỹ thuật của chính model đó, nhưng cross-family comparison và hai hàng DL phải quarantine cho đến khi sửa và rerun. Artifact còn ghi commit `743b4ac...`, trong khi runner fair chưa tồn tại trong tree của commit đó; run nhiều khả năng được tạo từ working tree chưa commit.

### 6.5 Final bundle

- Latest pointer: `final-5a0b5041-5216-4a48-9e46-b0c16ab14866`.
- Audit checksum-only: 32 file được kiểm, 0 mismatch.
- `model_checksums.json` có `checkpoints: {}` và chỉ ghi parameter count 13.059.

Bundle nhất quán theo checksum nhưng không bind checkpoint bất biến, nên chưa thể tái tạo exact inference model từ bundle một cách độc lập.

## 7. Test audit

| Kiểm tra | Kết quả | Diễn giải |
|---|---|---|
| `py -3.10 -m pytest -q` | 87 passed, 5 skipped | Code hiện tại không có test failure |
| Recommendation policy subset | 12 passed | Bao phủ contract hiện có, không bao phủ lifecycle/content validity |
| Test collection | 92 tests | Khớp 87 + 5 |
| Initial 5-second harness | Timeout của harness | Không phải test failure; rerun hoàn chỉnh đã pass |

Các khoảng trống test quan trọng:

- Không test resolved Optuna config giữ lại constants và inner/outer criterion parity.
- Không test selection estimator = final training estimator.
- Không test scheduler/SWA/refit policy parity.
- Không test target conflict khi `ON CONFLICT DO NOTHING`.
- Không test missing recommendation feature phải abstain thay vì mặc định 0/1.
- Không test confidence nằm trong `[0,1]`.
- Không test feature-policy alignment với sensitive/excluded inventory.
- Metric `no_contradiction` chỉ test trùng `risk_code`, không test mâu thuẫn/redundancy ngữ nghĩa.
- Một test có tên “loader drops target” chỉ kiểm tra source string, không thực sự gọi loader và assert frame đầu ra.

## 8. Nhật ký diagnostic nhỏ

| Diagnostic | Phạm vi và dữ liệu | Kết quả chính | Locked 79? | Vai trò |
|---|---|---|---|---|
| Git/environment inventory | Repository | Branch/commit/env như mục 2 | Không | Audit |
| Full pytest | Unit/integration tests | 87 pass, 5 skip | Không chạy inference/selection trên 79 | Engineering evidence |
| Raw + DB profile | 395 source rows; transaction read-only | Hash khớp, 0 null/duplicate/orphan/mismatch | Không đọc metric/prediction của 79 | Integrity diagnostic |
| V2 manifest verify | 316 development memberships | 316 records, 1.580 assignments, hash hợp lệ | Không | Protocol diagnostic |
| Development grade stats | 316 development rows | Corr G2–G3 0,9048; G2 rule Macro-F1 0,898836 | Không | Feasibility diagnostic |
| Residual sanity | 5 immutable outer folds, fixed Ridge/Huber, không tuning | Không cải thiện Macro-F1 so zero-residual/G2 rule | Không | Hypothesis diagnostic |
| One-fold neural smoke | Outer fold 0, seed 42, frozen config | Xác nhận scheduler/refit/drop_last behavior; score không dùng ranking | Không | Estimator diagnostic |
| Fair criterion probe | Synthetic label vector/config resolution | Missing constant kích hoạt class weights ngoài ý muốn | Không | Bug confirmation |
| Final bundle checksum-only | 32 files | 0 mismatch; checkpoint map rỗng | Không mở/recompute prediction | Provenance audit |
| Recommendation rule probes | Synthetic missing/confidence cases + structural aggregate | Missing bị đổi thành risk; confidence ngoài miền pass | Không dùng để chọn model | Safety diagnostic |

Không diagnostic nào ở trên là full model-selection evidence; không diagnostic nào được dùng để xếp hạng hay chốt model.

## 9. Issue register rút gọn

| Ưu tiên | Phân loại | Severity | Evidence | Phát hiện | Expected impact | Fix cost |
|---:|---|---|---|---|---|---|
| 1 | Critical correctness | Critical | Confirmed | Fair comparison sai estimator công bố | High | Code thấp, rerun cao |
| 2 | Scientific validity | Critical | Confirmed | Không còn unseen test | High | High/new data |
| 3 | Training-estimator | High | Confirmed | Selection không ước lượng final estimator | High | Medium |
| 4 | Training-estimator | High | Confirmed | Scheduler/SWA và refit không đồng nhất | High | Medium |
| 5 | Reproducibility | High | Confirmed | Artifact commit không bind exact code | High | Medium + rerun |
| 6 | Model-design limitation | High | Confirmed | Sequence length 2, kernel 1 pointwise | High | Medium |
| 7 | Evaluation | High | Confirmed | G2 rule chưa nằm trong fair runner hiện tại | High | Low |
| 8 | Future-data limitation | High | Confirmed | Context timing và entity ID chưa đủ | High | High |
| 9 | Recommendation-system gap | High | Confirmed | Không có goal/follow-up/expert evidence | High | High |
| 10 | Recommendation safety | High | Confirmed | Missing/confidence/feature governance lỗi | High | Low–Medium |
| 11 | Engineering/lineage | High | Confirmed design; live sạch | Target/DB invariants có thể trôi | High | Low–Medium |
| 12 | Documentation | High | Confirmed | README/PROJECT có thể dẫn tới claim sai | High | Low, sau approval |

## 10. Kết luận audit repository

Repository không phải dự án “chưa có gì”: data lineage, outer-fold protocol, DB append-only và test coverage là nền tốt. Nhưng tầng claim khoa học đang vượt trước evidence ở ba chỗ: gọi 79 là locked, coi fair comparison là hợp lệ, và coi recommendation snapshot là lộ trình học. Mọi cải tiến phải bắt đầu bằng **quarantine evidence sai, đồng nhất estimator và định nghĩa lại claim boundary**. Chỉ sau các gate đó mới hợp lý chi compute cho candidate architecture.

Các phân tích chi tiết tiếp theo:

- [Đối chiếu yêu cầu khóa luận](./02_thesis_requirement_gap_analysis.md)
- [Chẩn đoán mô hình dự đoán](./03_prediction_model_diagnosis.md)
- [Chẩn đoán recommendation](./04_recommendation_system_diagnosis.md)
- [Chiến lược ứng viên](./05_candidate_strategies.md)
- [Kế hoạch thực nghiệm](./06_proposed_experimental_plan.md)
- [Risk và decision gates](./07_risks_and_decision_gates.md)
- [Tóm tắt cần phê duyệt](./08_approval_summary.md)
