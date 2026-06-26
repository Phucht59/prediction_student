# 05. Thesis Ready Handoff

## PROJECT IDENTITY

Repository được phân tích là `prediction_student`, remote hiện tại trỏ tới `https://github.com/Phucht59/prediction_student.git`, nhánh làm việc là `main`. Tên đề tài dự kiến phù hợp với nội dung repository là: "XÂY DỰNG MÔ HÌNH HỌC KẾT HỢP ĐỂ DỰ ĐOÁN THÀNH TÍCH HỌC TẬP SINH VIÊN VÀ ĐỀ XUẤT LỘ TRÌNH HỌC TẬP CÁ NHÂN HÓA". Dự án kết hợp hai lớp chức năng: mô hình dự đoán thành tích học tập theo ba lớp `Low`, `Medium`, `High`, và module khuyến nghị học tập cá nhân hóa RA-HLPR. Phần dự đoán dùng CNN-BiLSTM hoặc biến thể có gated fusion tùy dataset và artifact final. Phần khuyến nghị nhận xác suất dự đoán từ mô hình, chẩn đoán rủi ro học tập, xếp hạng can thiệp và lập lộ trình 4 tuần.

Nguồn sự thật mạnh nhất cho kết quả final là `reports/final/`, đặc biệt `reports/final/final_model_manifest.json`, `reports/final/final_deep_results_table.csv`, `reports/final/final_baseline_comparison.csv`, `reports/final/final_prediction_model_report.md`, `reports/final/FINAL_PROJECT_STATUS.md`, `reports/final/final_recommender_report.md` và `reports/final/final_recommender_thesis_summary_vi.md`. Source code chính nằm trong `src/`, các script vận hành nằm trong `scripts/`, test nằm trong `tests/`, recommender output nằm trong `outputs/recommender/`, intervention catalog nằm trong `data/recommender/intervention_catalog.csv`. Có một số artifact cũ hoặc không khớp final trong `models/final/`; khi viết khóa luận cần ưu tiên `reports/final/` và ghi rõ các phần chỉ partially verified.

## RESEARCH PROBLEM

Bài toán nghiên cứu thực tế của repository là phân loại thành tích học tập sinh viên thành ba mức rủi ro/thành tích: `Low`, `Medium`, `High`. Đây không phải bài toán hồi quy chính. Source có một số head hoặc metric liên quan hồi quy trong các mô hình thử nghiệm, ví dụ `reg_head` trong `src/models_v27.py` và archive experiment, nhưng final report loại trừ việc claim regression head là đóng góp final. Vì vậy khi viết báo cáo, cần mô tả bài toán chính là classification ba lớp. Mọi nội dung về hồi quy, nếu nhắc đến, chỉ nên coi là thử nghiệm phụ hoặc code tồn tại trong repository, không phải hướng đánh giá final.

Với dataset Student Performance (`student-mat`, `student-por`), biến mục tiêu là cột `G3`. Trong `src/config.py`, bins cho ba lớp là `[0,9,14,20]`; trong `src/data_pipeline.py`, `G3` được chuyển thành nhãn 0, 1, 2 bằng `pd.cut(... labels=[0,1,2], include_lowest=True)`. Có thể diễn giải là `Low` tương ứng vùng điểm thấp, `Medium` vùng trung bình, `High` vùng cao theo bins này. Source cũng giữ `G3_raw` để phục vụ một số logic nội bộ nhưng `FeatureSelector` và preprocessing loại `G3_raw` khỏi input để tránh leakage. Với xAPI, biến mục tiêu là `Class`, được map bởi `XAPI_CLASS_MAPPING = {"L":0,"M":1,"H":2}`.

Repository có khái niệm scenario cho Student datasets. Archive scenario code trong `archive/experiments/src_experiments/current_src_experiments.20260617_170850/common.py` định nghĩa `early`, `midterm`, `late`. `early` loại cả `G1` và `G2`, không dùng sequence; `midterm` dùng `G1` và loại `G2`; `late` dùng cả `G1` và `G2`. Code áp dụng scenario trước feature engineering để tránh tạo feature dẫn xuất từ các điểm chưa được phép dùng tại thời điểm dự đoán. Với xAPI, final dùng scenario `default` và sequence gồm `raisedhands`, `VisITedResources`, `AnnouncementsView`, `Discussion`.

Mục tiêu đánh giá không chỉ là accuracy. Artifact final nhấn mạnh Macro F1, Recall Low và F1 Low. Lý do là bài toán giáo dục quan tâm tới việc phát hiện nhóm sinh viên có nguy cơ thấp/khó khăn để can thiệp sớm. Prediction mode `low_f1_tuned` tồn tại để điều chỉnh threshold cho lớp Low dựa trên xác suất đầu ra. Source archive `deep_debug.py` có hàm chọn threshold theo OOF probabilities, trong đó `low_f1_tuned` tối ưu F1 Low, có tie-break nhẹ bằng Macro F1. Một hàm khác trong archive scenario code chọn threshold theo điểm tổng hợp `0.65*recall_low + 0.35*macro_f1`. Final manifest xAPI ghi rõ threshold tuning dựa trên CV/OOF probabilities và locked test chỉ dùng cho final evaluation. Tuy nhiên numeric threshold cụ thể cho từng final row chưa tìm được trong artifact final, nên không được ghi số threshold nếu chưa bổ sung evidence.

## OBJECTIVES

Mục tiêu tổng quát của đề tài là xây dựng và đánh giá một hệ thống dự đoán thành tích học tập sinh viên kết hợp với module đề xuất lộ trình học tập cá nhân hóa. Hệ thống dùng dữ liệu học tập dạng tabular và một số biến có thể xem như chuỗi theo thời gian/ngữ cảnh để dự đoán xác suất thuộc ba lớp `Low`, `Medium`, `High`, sau đó sử dụng xác suất này để chẩn đoán rủi ro và đề xuất can thiệp phù hợp.

Mục tiêu cụ thể nên viết gồm: xây dựng pipeline tiền xử lý dữ liệu cho `student-mat`, `student-por` và `xAPI`; định nghĩa nhãn ba lớp từ `G3` hoặc `Class`; thiết kế mô hình CNN-BiLSTM cho feature dạng sequence và context; kiểm tra biến thể gated fusion cho xAPI; so sánh mô hình deep learning với baseline truyền thống khi có artifact; đánh giá theo Macro F1, Recall Low và F1 Low; xây dựng RA-HLPR gồm risk diagnosis, candidate generation, hybrid scoring và path planning; đánh giá offline RA-HLPR bằng risk diagnosis metrics, ranking metrics và path quality metrics.

Câu hỏi nghiên cứu có thể đặt theo bằng chứng hiện có: mô hình CNN-BiLSTM/gated fusion dự đoán ba mức thành tích với hiệu quả như thế nào trên từng dataset/scenario; điều chỉnh threshold cho lớp Low ảnh hưởng thế nào tới khả năng phát hiện sinh viên nguy cơ thấp; baseline truyền thống so với deep model ra sao trên artifact xAPI; xác suất đầu ra của mô hình dự đoán có thể được sử dụng như đầu vào cho quy trình RA-HLPR để sinh lộ trình học tập cá nhân hóa offline hay không. Không đặt câu hỏi theo hướng "RA-HLPR cải thiện điểm số thật" vì repository không có user study hoặc outcome sau can thiệp.

## DATASETS

Repository cấu hình ba dataset chính: `student-mat`, `student-por` và `xAPI`. `src/config.py` khai báo raw file tương ứng là `student-mat.csv`, `student-por.csv` và `xAPI-Edu-Data.csv`. Tuy nhiên trong workspace hiện tại, `data/raw/` chỉ có `.gitkeep`; raw CSV không có. `data/processed/final/` cũng trống. Do đó không thể kiểm trực tiếp số missing values, duplicate, exact schema từ raw data hoặc exact split indices. Mọi mô tả về missing/duplicate trong khóa luận cần được bổ sung bằng thống kê chạy lại sau khi khôi phục raw CSV, hoặc ghi là chưa xác minh.

Số mẫu có thể suy ra một phần từ recommender checkpoints và outputs. Với xAPI, checkpoint risk head ghi `training_rows=384`, output recommender có 96 hồ sơ locked test, tổng cộng 480 mẫu. Với student-por, checkpoint risk head ghi `training_rows=519`, output recommender có 130 hồ sơ locked test, tổng cộng 649 mẫu. Với student-mat, checkpoint risk head ghi `training_rows=316`, stale archived locked output có 79 hồ sơ, tổng 395 mẫu, nhưng final recommender cho student-mat đang pending, nên số này chỉ nên dùng như inference phù hợp dataset chuẩn chứ không phải evidence raw trực tiếp. Nếu báo cáo cần bảng dataset, phải đánh dấu rõ nguồn số mẫu là checkpoint/output inference khi raw file thiếu.

Nhãn của Student datasets tạo từ `G3` theo bins `[0,9,14,20]`. `src/data_pipeline.py` lưu `G3_raw` rồi chuyển `G3` sang mã lớp. Nhãn của xAPI tạo từ `Class` theo mapping `L`, `M`, `H`. `src/data_pipeline.py` cũng định nghĩa sequence columns: Student dùng `G1`, `G2`; xAPI dùng `raisedhands`, `VisITedResources`, `AnnouncementsView`, `Discussion`. Các categorical features được label encode, numerical features được MinMax scale trong `DataPreprocessor`. Source có xử lý oversampling và cảnh báo về ADASYN trên categorical label encoding; final report nói ADASYN trực tiếp với categorical label encoding bị loại trừ, vì vậy không claim ADASYN là kỹ thuật final.

Dataset thực sự có result final mạnh nhất là xAPI, vì có manifest, final deep table và baseline comparison CSV. Student final rows có trong `FINAL_PROJECT_STATUS.md`, README và cleanup log, nhưng thiếu final manifest/per-run CSV/checkpoint exact, nên chỉ partially verified. Recommender final có output cho xAPI và student-por; student-mat recommender final đang pending do thiếu prediction metadata/checkpoint tương thích.

## PREPROCESSING

Pipeline dữ liệu nằm chủ yếu trong `src/data_pipeline.py`. Hàm xử lý target tạo nhãn phân loại và lưu một số cột raw cần thiết. Locked test split dùng `train_test_split` với `test_size=0.2`, `random_state=42`, stratify theo target, theo hằng số trong `src/config.py`. Feature selection loại target khỏi input, loại `G3_raw` để tránh leakage, và bảo toàn các cột sequence bắt buộc. `StudentDataset` tạo sequence tensor dạng `(batch, seq_len, 1)` và tách context numerical/categorical.

Với Student scenario, archive code cho thấy nguyên tắc quan trọng: loại các grade chưa được phép dùng trước khi feature engineering. Đây là điểm cần nhấn mạnh trong khóa luận vì nếu tạo feature từ `G1` hoặc `G2` rồi mới drop cột gốc, có thể gây leakage thời điểm. Scenario `late` dùng `G1`, `G2`; `midterm` chỉ dùng `G1`; `early` không dùng cả hai. Final status chỉ có Student `late` và `student-por midterm`, không có `early` final row.

Với xAPI, sequence default được tạo từ bốn biến tương tác học tập. Các biến context gồm categorical và numerical còn lại sau khi bỏ target `Class`. Recommender risk rules cho xAPI dùng các biến như `StudentAbsenceDays`, `VisITedResources`, `raisedhands`, `Discussion`, `AnnouncementsView`, `ParentAnsweringSurvey`, `ParentschoolSatisfaction`, nhưng không dùng nhãn thật `Class` để sinh khuyến nghị vận hành. Tương tự, risk rules Student không dùng `G3` thật để chẩn đoán rủi ro trong vận hành.

## MODEL ARCHITECTURE

Mô hình chính trong source hiện tại là `StudentHybridModel` tại `src/models/models.py`. Kiến trúc gồm nhánh sequence CNN-BiLSTM và nhánh context. Nhánh sequence nhận tensor `(batch, seq_len, 1)`, chuyển sang dạng phù hợp cho `Conv1d`, đi qua `Conv1d`, `BatchNorm1d`, `ReLU`, `Dropout`, sau đó qua BiLSTM hai chiều. Output BiLSTM được gom bằng attention pooling (`AttentionPooling1D`). Nhánh context dùng embedding cho categorical features, numerical features dạng tensor, rồi qua MLP. Hai vector sequence và context được concat và đưa vào classifier. Đây là kiến trúc CNN-BiLSTM kết hợp context theo cách nối vector.

Archive `deep_debug.py` có model `SequenceCNNBiLSTMOnly`, dùng sequence only với Conv1d, BatchNorm, ReLU, Dropout, BiLSTM bidirectional, attention pooling và các head dự đoán. Tên `sequence_cnn_bilstm_only` khớp ba dòng final Student trong `FINAL_PROJECT_STATUS.md`. Tuy nhiên exact checkpoint/manifest cho ba dòng này không có trong final artifacts, vì vậy chỉ nên mô tả là final status ghi model này, còn cấu hình chi tiết exact là partially verified từ archive source.

Gated fusion nằm trong `src/models_v27.py`. Class `GatedFusion` chiếu vector sequence và vector context về cùng không gian, học một gate sigmoid từ concat input, rồi trộn `gate*h_seq + (1-gate)*h_ctx`. `StudentHybridV27` dùng CNN, BiLSTM, attention pooling, context MLP và gated fusion. Final manifest xAPI ghi `model_variant = gated_fusion_v28` và architecture là `CNN-BiLSTM with gated context fusion`. Source class exact tên `gated_fusion_v28` không tìm thấy; source gần nhất là `models_v27.py`. Vì vậy trong khóa luận có thể nói xAPI final artifact dùng gated fusion v28, và mô tả nguyên lý gated fusion dựa trên source v27, nhưng phải tránh nói đã tìm thấy class v28 nếu chưa bổ sung evidence.

Training source nằm trong `src/train_pipeline.py`, có EarlyStopping theo validation metric, scheduler ReduceLROnPlateau, class weights, Optuna search, RepeatedStratifiedKFold, ensemble probabilities và logic lưu model vào `models/saved/final`. Loss functions trong `src/losses_v27.py` gồm focal loss, class-balanced focal loss, ordinal loss và joint hybrid loss. Tuy nhiên exact batch size, epoch, optimizer, learning rate của final Student rows chưa có manifest. Với xAPI final, manifest cung cấp kết quả và guardrails nhưng không đủ để tái lập checkpoint exact.

## FINAL MODEL SELECTION

Bảng final prediction dự kiến trong user prompt được kiểm chứng như sau. Dòng xAPI `default`, model `gated_fusion_v28`, prediction mode `low_f1_tuned`, Macro F1 `0.7541`, Recall Low `0.8846`, F1 Low `0.8214` được verified bằng `reports/final/final_model_manifest.json`, `reports/final/final_deep_results_table.csv` và `reports/final/FINAL_PROJECT_STATUS.md`. Ba dòng Student được partially verified: `student-mat late sequence_cnn_bilstm_only low_f1_tuned` với Macro F1 `0.9365`, Recall Low `0.9615`, F1 Low `0.8929`; `student-por late sequence_cnn_bilstm_only low_f1_tuned` với `0.8783`, `0.9000`, `0.8182`; `student-por midterm sequence_cnn_bilstm_only argmax` với `0.8228`, `0.6500`, `0.7429`. Các số Student có trong `reports/final/FINAL_PROJECT_STATUS.md`, README và cleanup log, nhưng thiếu matching final manifest/checkpoint/per-run CSV.

Một điểm quan trọng là `models/final/final_model_manifest.json` không phải nguồn final hiện tại. File này mô tả strict-validation v23 với metrics thấp hơn và không khớp bảng final trong `reports/final`. Một số checkpoint Student trong `models/final` có classifier output 5 lớp, không phù hợp nhãn final ba lớp. Vì vậy không được dùng `models/final` để thay thế cho `reports/final` nếu viết kết quả final.

Locked test được thiết kế để giữ riêng cho final evaluation. `reports/final/final_model_manifest.json` ghi evaluation protocol: locked test only final; threshold tuning dùng CV/OOF probabilities. Đây là guardrail quan trọng để tránh overfit vào test. Tuy nhiên vì processed splits hiện không có, chưa thể tự kiểm chứng exact locked split rows.

## BASELINE COMPARISON

Baseline source trong archive gồm Logistic Regression, Random Forest, XGBoost hoặc CatBoost/HistGradient fallback, và MLP. Nhưng artifact final đầy đủ chỉ có baseline xAPI trong `reports/final/final_baseline_comparison.csv`. File này ghi deep model xAPI `gated_fusion_v28` có Macro F1 `0.7541`, còn baseline `RandomForestClassifier` có Macro F1 `0.8465`. Recall Low và F1 Low của baseline trong final CSV là `not_available`. Vì vậy kết luận hợp lệ là: trên xAPI, baseline Random Forest có Macro F1 cao hơn deep model trong artifact final; deep model vẫn được chọn trong luồng chính do mục tiêu Low-class probability và tích hợp RA-HLPR, nhưng không được viết rằng deep learning vượt baseline trên xAPI về Macro F1.

Không có baseline final cho student-mat hoặc student-por trong `final_baseline_comparison.csv`. Do đó khóa luận không được claim deep model tốt hơn baseline trên Student datasets nếu không bổ sung artifact. Có thể mô tả baseline là đối chứng thử nghiệm có source code, nhưng bảng kết quả final chỉ nên đưa xAPI baseline hoặc đánh dấu Student baseline missing.

Final reports cũng ghi rõ baseline không dùng làm teacher model, không dùng distillation, không dùng pseudo-label, không dùng feature importance cho mô hình chính. Đây là non-negotiable guardrail. Nếu viết về feature importance, chỉ dùng khi có artifact hợp lệ và không gán cho baseline như cơ sở giải thích mô hình chính.

## RA-HLPR DESIGN

RA-HLPR trong repository là "Risk-Aware Hybrid Learning Path Recommender", được mô tả trong `reports/final/final_recommender_report.md` và triển khai qua các module trong `src/recommender/`. Đây không phải collaborative filtering. Không có user-user similarity, item-item similarity, matrix factorization hoặc implicit feedback history. RA-HLPR là pipeline prediction-aware, rule/metadata/hybrid scoring dựa trên rủi ro và intervention catalog.

Luồng RA-HLPR gồm: xác suất dự đoán từ mô hình classification; risk diagnosis; candidate generation; hybrid scoring; path planning 4 tuần. `RiskDiagnosisHead` trong `src/recommender/risk_head.py` là MLP ba tầng, huấn luyện bằng weak labels từ domain rules, dùng BCEWithLogitsLoss và Adam. `risk_rules.py` sinh weak labels theo dataset mà không dùng nhãn thật `G3` hoặc `Class` trong vận hành recommendation. `candidate_generator.py` dùng class probabilities để điều chỉnh threshold rủi ro và lọc intervention theo `applicable_kind`. `hybrid_scorer.py` tính điểm dựa trên risk match, performance need, difficulty fit, time fit, prerequisite fit, expected effect và rule adjustment. `path_planner.py` lập kế hoạch 4 tuần với các pha Stabilize, Practice, Reinforce, Evaluate & Adjust.

Intervention catalog nằm trong `data/recommender/intervention_catalog.csv`, có 20 dòng với các cột `item_id`, `intervention_name`, `description`, `target_risks`, `difficulty_level`, `estimated_hours_per_week`, `recommended_phase`, `expected_effect`, `prerequisite_level`, `applicable_kind`. Điều này cho phép lọc và xếp hạng can thiệp theo dataset và rủi ro. Với xAPI, risks final gồm các nhóm liên quan attendance, engagement, high failure probability; với Student, report liệt kê thêm low prior performance, declining trend, insufficient study time.

Final recommender status: xAPI và student-por có refreshed final outputs và metrics. student-mat pending do thiếu `models/saved/final/student-mat_3class_ensemble_features.json` và shape mismatch/prediction metadata. Không được trình bày student-mat recommender là hoàn tất final nếu chỉ dựa trên stale archive output.

## EXPERIMENTAL DESIGN

Thiết kế thực nghiệm gồm tách locked test 20% bằng stratification, train pool dùng CV/OOF để chọn/tune model hoặc threshold, locked test chỉ dùng final. `src/config.py` ghi `LOCKED_TEST_SIZE = 0.2`, `CV_FOLDS = 5`, `DEFAULT_SEED = 42`. `src/train_pipeline.py` có RepeatedStratifiedKFold, EarlyStopping, scheduler và ensemble support. Archive threshold code cho thấy threshold Low được tune trên OOF probabilities, không tune trực tiếp trên locked test theo thiết kế final.

Metric prediction gồm Accuracy, Precision, Recall, F1, Macro F1, Recall Low, F1 Low; archive code có thêm RMSE/R2 cho các thử nghiệm phụ, nhưng final thesis nên tập trung classification metrics. Confusion matrix và ROC/AUC final không tìm thấy artifact tương ứng. Statistical significance test không có. Ablation final chưa có bằng chứng đủ mạnh, dù có source variant trong archive. Khi viết Chương 4, nên chia rõ: kết quả đã xác minh; kết quả partially verified; phần chưa đủ dữ kiện.

Recommender evaluation là offline. `outputs/recommender/xapi/recommender_metrics.json` và `outputs/recommender/student-por/recommender_metrics.json` chứa risk diagnosis metrics, ranking metrics và path quality metrics. Không có user feedback, không có A/B test, không có longitudinal outcome sau can thiệp. Vì vậy không claim RA-HLPR cải thiện thành tích học tập thực tế, chỉ claim hệ thống sinh lộ trình cá nhân hóa và có đánh giá offline theo tiêu chí nội bộ/weak supervision.

## VERIFIED RESULTS

Kết quả xAPI final prediction: `gated_fusion_v28`, scenario `default`, prediction mode `low_f1_tuned`, Macro F1 `0.7541`, Recall Low `0.8846`, F1 Low `0.8214`. Evidence mạnh: `reports/final/final_model_manifest.json`, `reports/final/final_deep_results_table.csv`, `reports/final/final_baseline_comparison.csv`. Baseline xAPI Random Forest có Macro F1 `0.8465`, cao hơn deep model về Macro F1 trong CSV final.

Kết quả Student final prediction: `student-mat late sequence_cnn_bilstm_only low_f1_tuned` đạt Macro F1 `0.9365`, Recall Low `0.9615`, F1 Low `0.8929`; `student-por late sequence_cnn_bilstm_only low_f1_tuned` đạt Macro F1 `0.8783`, Recall Low `0.9000`, F1 Low `0.8182`; `student-por midterm sequence_cnn_bilstm_only argmax` đạt Macro F1 `0.8228`, Recall Low `0.6500`, F1 Low `0.7429`. Evidence chỉ là `reports/final/FINAL_PROJECT_STATUS.md`, README và cleanup log, nên trạng thái là partially verified.

Recommender xAPI: risk diagnosis F1 macro `0.9831`, F1 micro `0.9813`; ranking P@3 `0.6840`, R@3 `0.4720`, NDCG@3 `0.8229`, coverage@3 `0.65`; path risk coverage `0.8958`, difficulty progression `0.7153`, prerequisite violation `0.0`. Recommender student-por: risk diagnosis F1 macro `0.9359`, F1 micro `0.9094`; ranking P@3 `0.6641`, R@3 `0.3185`, NDCG@3 `0.7455`, coverage@3 `0.55`; path risk coverage `0.9508`, difficulty progression `0.6`, prerequisite violation `0.0449`. Evidence là recommender metrics JSON và final recommender report.

Test suite hiện tại pass bằng `py -3.10 -m pytest -q`: `31 passed in 20.57s`. Default `python` là 3.14 và thiếu pytest, nên không dùng runtime đó để đánh giá repo.

## FIGURES AVAILABLE

Visual pack đã được tạo trong `report_context/figures/` từ artifact thật. Có 10 hình: prediction metrics, xAPI baseline comparison, Low-class focus, Macro F1 ranking, recommender offline metrics, risk diagnosis metrics, ranking metrics, path quality metrics, pipeline overview và RA-HLPR flow. Script tạo là `report_context/figures/create_verified_figures.py`; manifest là `report_context/figures/figure_manifest.csv`; README figure pack ghi chú nguồn dữ liệu và giới hạn xác thực.

Khi dùng hình prediction metrics, cần chú thích Student rows là partially verified. Khi dùng hình baseline comparison, chỉ nói về xAPI. Khi dùng hình recommender metrics, phải ghi là offline metrics, không phải bằng chứng cải thiện học lực thật. Không dùng ảnh AI hoặc biểu đồ minh họa không có dữ liệu.

## LIMITATIONS

Hạn chế lớn nhất là thiếu raw datasets và processed splits trong workspace. Điều này làm cho việc kiểm missing values, duplicate, schema và exact split không thể thực hiện trực tiếp. Hạn chế thứ hai là thiếu final checkpoint/metadata trong `models/saved/final`, khiến full rerun prediction/recommender theo script hiện tại không thể làm ngay. Hạn chế thứ ba là Student final metrics chỉ partially verified, thiếu artifact final chi tiết. Hạn chế thứ tư là exact source/checkpoint `gated_fusion_v28` không được tìm thấy dưới tên đó, dù source gated fusion v27 và final manifest xAPI có tồn tại.

Hạn chế thực nghiệm gồm thiếu statistical significance test, thiếu ablation final, thiếu confusion matrix final, thiếu latency benchmark, thiếu calibration metrics, thiếu full Student baseline final. Hạn chế khuyến nghị gồm đánh giá offline, weak supervision cho risk diagnosis, không có user feedback, không có thử nghiệm can thiệp thật, không có bằng chứng cải thiện thành tích sau khi áp dụng lộ trình.

## ETHICAL AND PRACTICAL CONSIDERATIONS

Vì hệ thống xử lý dữ liệu giáo dục và dự đoán rủi ro học tập, báo cáo nên nhấn mạnh rằng kết quả dự đoán chỉ hỗ trợ giảng viên/cố vấn, không nên dùng như quyết định tự động mang tính trừng phạt hoặc loại trừ sinh viên. Lớp `Low` cần được hiểu là tín hiệu can thiệp hỗ trợ, không phải nhãn cố định về năng lực cá nhân. Cần bảo mật dữ liệu sinh viên, hạn chế truy cập thông tin cá nhân, và minh bạch với người học nếu hệ thống được triển khai thực tế.

Do RA-HLPR dùng rule-based/weak supervision và metadata intervention, recommendation nên được xem là gợi ý tham khảo. Cần có giảng viên hoặc cố vấn kiểm tra tính phù hợp của lộ trình, đặc biệt với các sinh viên có hoàn cảnh cá nhân, sức khỏe hoặc điều kiện học tập đặc thù. Không nên claim hệ thống thay thế cố vấn học tập hoặc chứng minh cải thiện điểm số nếu chưa có nghiên cứu người dùng.

## MISSING INFORMATION

Các thông tin cần bổ sung trước khi viết bản thesis hoàn chỉnh gồm: raw CSV và thống kê dữ liệu; processed split files; exact final checkpoints; Student final per-run logs; exact threshold values; OOF probabilities; confusion matrices; baseline final cho Student; Low metrics của xAPI baseline; statistical tests; ablation; calibration; latency; dashboard evidence; user feedback cho recommender; dataset citation/version; training hardware. Danh sách chi tiết có trong `report_context/04_UNKNOWN_OR_MISSING_INFORMATION.md`.

Nếu thời gian hạn chế, ưu tiên bổ sung ba nhóm: raw/processed data statistics, exact final prediction artifacts, và confusion matrix/prediction outputs. Ba nhóm này ảnh hưởng trực tiếp tới Chương 2, Chương 3 và Chương 4 của khóa luận.

## SUGGESTED 5-CHAPTER THESIS OUTLINE

Chương 1 nên giới thiệu bài toán dự đoán thành tích học tập, nhu cầu phát hiện sớm sinh viên nguy cơ thấp, và mục tiêu kết hợp mô hình dự đoán với lộ trình học tập cá nhân hóa. Trình bày câu hỏi nghiên cứu và phạm vi: classification ba lớp trên `student-mat`, `student-por`, `xAPI`; RA-HLPR đánh giá offline.

Chương 2 nên trình bày cơ sở lý thuyết: educational data mining, classification, CNN, BiLSTM, attention pooling, gated fusion, threshold tuning, recommender systems. Với recommender, nhấn mạnh RA-HLPR không phải collaborative filtering mà là risk-aware hybrid rule/metadata/prediction-based recommender.

Chương 3 nên mô tả hệ thống: data pipeline, target construction, scenario design, leakage control, model architecture, training/evaluation protocol, locked test policy, RA-HLPR components và intervention catalog. Sơ đồ pipeline nên dùng Mermaid hoặc FIG-09/FIG-10 từ visual pack.

Chương 4 nên trình bày thực nghiệm: môi trường Python 3.10, dependency, test pass, dataset table, final prediction table, xAPI baseline comparison, Low-class metrics, recommender offline metrics. Cần đánh dấu Student prediction rows là partially verified nếu chưa bổ sung artifact. Không claim deep model thắng baseline mọi dataset.

Chương 5 nên tổng kết đóng góp và hạn chế. Đóng góp hợp lệ: pipeline classification ba lớp, CNN-BiLSTM/gated fusion, threshold focus cho Low, RA-HLPR offline path generation. Hạn chế: thiếu raw/processed artifacts trong workspace, thiếu statistical tests, thiếu user feedback, missing final checkpoints, recommender chưa chứng minh cải thiện điểm thật. Hướng phát triển: bổ sung artifact tái lập, calibration, ablation, user study, dashboard evidence, mở rộng dataset.

## NON-NEGOTIABLE FACTS

1. Bài toán final là classification ba lớp `Low`, `Medium`, `High`, không phải regression chính.
2. Student label được tạo từ `G3`; xAPI label được tạo từ `Class`.
3. Student bins trong source final là `[0,9,14,20]`; xAPI mapping là `L=0`, `M=1`, `H=2`.
4. Không dùng `G3` hoặc `Class` thật để sinh recommendation vận hành.
5. `student-combine` không được dùng làm dataset chính nếu không có artifact final chứng minh.
6. Không claim final model có regression head nếu chỉ dựa vào source thử nghiệm.
7. Không claim ADASYN trực tiếp với categorical label encoding là kỹ thuật final.
8. Baseline chỉ là đối chứng; không dùng baseline để distillation, pseudo-label, teacher model hoặc feature importance của mô hình chính.
9. Locked test được thiết kế chỉ dùng final evaluation; threshold tuning dựa trên CV/OOF probabilities.
10. xAPI final prediction result là verified: Macro F1 `0.7541`, Recall Low `0.8846`, F1 Low `0.8214`.
11. Student final prediction rows là partially verified cho đến khi có manifest/checkpoint/per-run artifact exact.
12. xAPI Random Forest baseline có Macro F1 `0.8465`, cao hơn deep model xAPI về Macro F1 trong final baseline CSV.
13. RA-HLPR không phải collaborative filtering.
14. RA-HLPR final output refreshed có cho xAPI và student-por; student-mat recommender final đang pending.
15. Recommender evaluation là offline; không có bằng chứng user feedback hoặc cải thiện thành tích thật.
16. `models/saved/final` trống trong workspace hiện tại, nên không rerun được full final pipeline theo đường dẫn mặc định.
17. `data/raw` và `data/processed/final` thiếu dữ liệu, nên không kiểm trực tiếp missing values, duplicate hoặc exact splits.
18. `models/final/final_model_manifest.json` không khớp `reports/final` final metrics và không nên dùng làm nguồn final results hiện tại.
19. Test suite pass bằng Python 3.10: `31 passed in 20.57s`.
20. Mọi bảng/hình dùng trong khóa luận phải có evidence path; nếu không có, ghi là missing hoặc limitation.
