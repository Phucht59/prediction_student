# Recommendation Engine and Pipeline Analysis

## Executive Summary
This analysis details the student performance prediction and recommendation engine in the project codebase. The system predicts student academic performance across three classes (`Low`, `Medium`, `High`) and generates tailored weekly learning path recommendations based on student features and predictions. 
The analysis covers:
1. **RuleBasedLearningPathEngine**: The mechanism mapping input features to academic risks and structured learning path actions.
2. **Datasets**: Detailed documentation of features, values, and class binning for `student-mat`, `student-por`, and `xapi` datasets.
3. **Model Orchestration & Pipeline**: The end-to-end training, Optuna optimization, ensemble prediction, evaluation, explainability, and PostgreSQL persistence workflow.
4. **Verification & Testing**: Test suite execution results, including an architectural mismatch finding.

---

## 1. Recommendation Engine (`RuleBasedLearningPathEngine`)
The recommendation logic is defined in `src/explainability.py` inside the `RuleBasedLearningPathEngine` class. It maps student risk factors to staged, action-oriented learning paths.

### A. Inputs and Outputs of `generate`
*   **Method Signature**: `generate(features: dict, predicted_class: int, confidence: float) -> dict`
*   **Outputs**: A dictionary containing:
    *   `predicted_class` (int: 0, 1, 2)
    *   `predicted_class_name` (str: "Low", "Medium", "High")
    *   `confidence` (float)
    *   `risk_band` (str: "stable", "moderate", "high")
    *   `headline` (str)
    *   `risk_factors` (list of dictionaries representing identified risk factors)
    *   `learning_path` (list of dictionaries representing staged goals and actions)

### B. Risk Identification Rules
The engine defines dataset-specific rules for identifying risks:

#### For Student Performance (`student` mode)
Using features from student-mat or student-por:
*   **Attendance Risk (`attendance`)**: Triggered if `absences >= 10` or the ratio of `absences / max(studytime, 0.5) >= 5`. Priority 1.
*   **Failure History Risk (`failure_history`)**: Triggered if `failures > 0` (past class failures). Priority 1.
*   **Grade Gap Risk (`grade_gap`)**: Triggered if current midterm results show `G2 < 10` or a downward trend (`g2 < g1` when `g1 > 0`). Priority 1.
*   **Low Study Time Risk (`study_time`)**: Triggered if `studytime <= 1` (less than 2 hours/week). Priority 2.
*   **Wellbeing Risk (`wellbeing`)**: Triggered if workday + weekend alcohol consumption `Dalc + Walc >= 6`. Priority 3.
*   **Time Management Risk (`time_management`)**: Triggered if going out frequency `goout >= 4`. Priority 3.

#### For xAPI Educational Data (`xapi` mode)
*   **Attendance Risk (`attendance`)**: Triggered if `StudentAbsenceDays` is "Above-7". Priority 1.
*   **Low Resource Usage Risk (`resource_usage`)**: Triggered if `VisITedResources < 40`. Priority 1.
*   **Low Class Engagement Risk (`class_engagement`)**: Triggered if `raisedhands < 30` or `Discussion < 30`. Priority 2.
*   **Course Updates Risk (`course_updates`)**: Triggered if `AnnouncementsView < 30`. Priority 2.
*   **Lack of Parent Support Risk (`parent_support`)**: Triggered if `ParentAnsweringSurvey` is "No". Priority 3.
*   **School Connection Risk (`school_support`)**: Triggered if `ParentschoolSatisfaction` is "Bad". Priority 3.

### C. Staged Actions Mapping
Based on identified risk codes, the engine recommends staged actions:

#### For Student Performance (`student` mode)
*   `attendance` $\rightarrow$ **Tuần 1**: Goal: "Khôi phục chuyên cần", Action: "Lập lịch đi học đủ; đăng ký lớp bù; cố vấn kiểm tra..."
*   `failure_history` or `grade_gap` $\rightarrow$ **Tuần 1-2**: Goal: "Bù lỗ hổng kiến thức", Action: "Làm bài chẩn đoán; học lại 2 chủ đề yếu nhất; 3 bài luyện tập/tuần."
*   `study_time` or `time_management` $\rightarrow$ **Tuần 2-4**: Goal: "Ổn định nếp tự học", Action: "Tăng $\ge$ 3 giờ tự học/tuần; chia thành phiên 45 phút; giảm 1 buổi đi chơi."
*   `wellbeing` $\rightarrow$ **Tuần 2-4**: Goal: "Điều chỉnh thói quen sinh hoạt", Action: "Giảm đồ uống có cồn; duy trì giấc ngủ/lịch học; gặp cố vấn."
*   **Always Appended**: **Mỗi cuối tuần**: Goal: "Theo dõi tiến bộ", Action: "Cập nhật điểm và tỷ lệ chuyên cần; nếu điểm < 60% hai tuần liên tiếp, chuyển sang phụ đạo trực tiếp."

#### For xAPI Educational Data (`xapi` mode)
*   `attendance` $\rightarrow$ **Tuần 1**: Goal: "Khôi phục chuyên cần", Action: "Xác nhận nguyên nhân vắng; làm gói bài bù; giáo viên kiểm tra tiến độ."
*   `resource_usage` or `course_updates` $\rightarrow$ **Tuần 1-2**: Goal: "Tăng sử dụng học liệu", Action: "Truy cập hệ thống $\ge$ 4 ngày/tuần; đọc thông báo; hoàn thành 2 tài nguyên trọng tâm."
*   `class_engagement` $\rightarrow$ **Tuần 2-4**: Goal: "Tăng tương tác học tập", Action: "Đặt câu hỏi/phản hồi; tham gia thảo luận; giáo viên ghi nhận."
*   `parent_support` or `school_support` $\rightarrow$ **Trong 2 tuần**: Goal: "Phối hợp gia đình - nhà trường", Action: "Gửi báo cáo tiến độ ngắn cho phụ huynh; thống nhất mục tiêu."
*   **Always Appended**: **Mỗi cuối tuần**: Goal: "Đánh giá lộ trình", Action: "So sánh mức truy cập/thảo luận/bài tập; nếu không cải thiện sau 2 tuần, bố trí kèm trực tiếp."

### D. Risk Band and Headline Generation
*   **Stable**: If predicted class is High (2) and zero risks are identified. Headline: "Duy trì lộ trình học tập hiện tại".
*   **High Risk**: If predicted class is Low (0) or any identified risk has a Priority 1. Headline: "Lộ trình can thiệp ưu tiên 4 tuần".
*   **Moderate Risk**: Default case. Headline: "Lộ trình củng cố để tiến lên nhóm High".

---

## 2. Dataset Exploration and Features

The raw data files are stored in `data/raw/` and parsed using specific delimiters.

### A. Student Performance Datasets (`student-mat.csv`, `student-por.csv`)
*   **CSV Delimiter**: Semicolon (`;`)
*   **Target column**: `G3` (final year grade, range 0-20)
*   **Target Binning**:
    *   **Low (0)**: $0 \le G3 \le 9$
    *   **Medium (1)**: $10 \le G3 \le 14$
    *   **High (2)**: $15 \le G3 \le 20$
*   **Sequential Features**: `G1` and `G2` (term 1 and term 2 grades, range 0-20)
*   **Categorical Features**: `school`, `sex`, `address`, `famsize`, `Pstatus`, `Mjob`, `Fjob`, `reason`, `guardian`, `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`, `romantic`.
*   **Numerical Features**: `age`, `Medu`, `Fedu`, `traveltime`, `studytime`, `failures`, `famrel`, `freetime`, `goout`, `Dalc`, `Walc`, `health`, `absences`.
*   **Engineered Features (in `apply_feature_engineering`)**:
    *   `grade_growth`: `G2 - G1`
    *   `grade_avg`: `(G1 + G2) / 2`
    *   `absence_study_ratio`: `absences / (studytime + 0.1)`
    *   `failure_risk`: `failures + absence_study_ratio`
    *   `alcohol_risk`: `Dalc + Walc`
    *   `social_risk`: `goout + freetime`

### B. xAPI Educational Dataset (`xAPI-Edu-Data.csv`)
*   **CSV Delimiter**: Comma (`,`)
*   **Target column**: `Class` (categorical values: `L`, `M`, `H`) mapped to integer classes:
    *   **Low (0)**: `L`
    *   **Medium (1)**: `M`
    *   **High (2)**: `H`
*   **Sequential Features**: `raisedhands`, `VisITedResources`, `AnnouncementsView`, `Discussion` (all range 0-100)
*   **Categorical Features**: `gender`, `NationalITy`, `PlaceofBirth`, `StageID`, `GradeID`, `SectionID`, `Topic`, `Semester`, `Relation`, `ParentAnsweringSurvey`, `ParentschoolSatisfaction`, `StudentAbsenceDays`.
*   **Numerical Features**: None (all numeric indicators are designated as sequential features).
*   **Engineered Features (in `apply_feature_engineering`)**:
    *   `engagement_score`: `raisedhands + VisITedResources + AnnouncementsView + Discussion`
    *   `absence_risk`: `1` if `StudentAbsenceDays == "Above-7"` else `0`
    *   `parent_support_signal`: `1` if `ParentAnsweringSurvey == "Yes"` else `0`
    *   `hands_resource_ratio`: `raisedhands / (VisITedResources + 1)`
    *   `active_participation`: `raisedhands * Discussion`
    *   `resource_engagement_ratio`: `VisITedResources / (engagement_score + 1)`

---

## 3. Model Orchestration and Pipeline Flow

The entire training, evaluation, prediction, and database logging are orchestrated in `scripts/run_pipeline.py`.

### A. Pre-split & Holdout Split
*   The raw dataset is split into **80% training pool** and **20% locked test set** via `create_and_save_locked_test()`.
*   Stratification is based on the target class (`G3` bins or `Class`). This holdout set is stored in `data/processed/final/` and remains completely untouched during optimization.

### B. Hyperparameter Optimization (Optuna)
*   Optuna runs trial searches to maximize the **validation F1-Macro** using a **Repeated Stratified 5-Fold Cross-Validation** (3 repeats).
*   Parameters searched include learning rate, weight decay, batch size, SMOTE oversampling ratio, CNN channels, LSTM hidden dimensions, context MLP hidden dimensions, fusion hidden dimensions, and dropout rates.
*   *Leakage Prevention*: Resampling (SMOTE/ADASYN), feature scaling, and feature selection are performed **strictly within each cross-validation fold**.

### C. Ensemble Seed Training
To guarantee robust predictions, an ensemble of models is trained using **5 fixed seeds** (`42, 123, 155, 156, 2025`).
*   For each seed, the training pool is split into 85% train and 15% val.
*   SMOTE/ADASYN oversampling is applied on the train split to resolve class imbalance.
*   A statistical feature selection (Pearson Correlation for numerical, Chi-Square for categorical) selects features with $p < 0.1$, while forcing sequential features to be retained.
*   **Averaged Model (SWA)** is evaluated during training. If the SWA model outperforms the standard early-stopped model on validation F1-Macro, its weights are saved.
*   Predictions on the locked test set are generated by averaging the output class probabilities from all 5 seeds.

### D. Prediction Schemes
*   **Student performance prediction**: Modeled as standard 3-class classification outputting logits for Low, Medium, High.
*   **xAPI performance prediction**: Modeled using an **ordinal classification** scheme. The model classifier outputs 2 logits representing decision boundaries: Class 0 vs 1+2 and Class 0+1 vs 2. The probabilities for Low, Medium, and High are reconstructed using sigmoid arithmetic:
    *   $P(\text{Low}) = 1.0 - \sigma(\text{logit}_0)$
    *   $P(\text{Medium}) = \max(0.0, \sigma(\text{logit}_0) - \sigma(\text{logit}_1))$
    *   $P(\text{High}) = \sigma(\text{logit}_1)$

### E. Database Logging
Evaluation results are persisted to PostgreSQL (configured in `.env`) into these tables:
*   `paper_runs`: metadata for the run (timestamp, counts).
*   `paper_predictions`: true labels, predictions, confidences, probabilities, and original features.
*   `paper_learning_recommendations`: recommendations, risk bands, features, and learning paths.
*   `paper_evaluation_metrics`: accuracy, precision, recall, F1-macro, RMSE, and R2.

---

## 4. Verification and Testing

### A. Test Execution Result
The test suite was run in the Conda environment `kltn` using the command `python -m pytest -v`. 
**Results Summary**: 9 passed, 1 failed.

Verbatim output:
```text
tests/test_thesis_pipeline.py::test_model_is_cnn_bilstm_mlp_and_outputs_three_class_probabilities PASSED [ 10%]
tests/test_thesis_pipeline.py::test_xapi_model_supports_independent_branch_dropouts PASSED [ 20%]
tests/test_thesis_pipeline.py::test_xapi_optuna_space_matches_high_trial_configuration PASSED [ 30%]
tests/test_thesis_pipeline.py::test_resampling_neighbor_count_is_configurable PASSED [ 40%]
tests/test_thesis_pipeline.py::test_forbidden_architectures_and_losses_are_removed FAILED [ 50%]
tests/test_thesis_pipeline.py::test_weighted_cross_entropy_supports_imbalanced_classes PASSED [ 60%]
tests/test_thesis_pipeline.py::test_feature_selector_keeps_required_sequence_columns PASSED [ 70%]
tests/test_thesis_pipeline.py::test_learning_path_engine_returns_staged_roadmap_not_variable_tweaks PASSED [ 80%]
tests/test_thesis_pipeline.py::test_learning_path_report_has_one_row_per_student PASSED [ 90%]
tests/test_thesis_pipeline.py::test_postgres_schema_stores_features_confidence_and_learning_paths PASSED [100%]

================================== FAILURES ===================================
_____________ test_forbidden_architectures_and_losses_are_removed _____________

    def test_forbidden_architectures_and_losses_are_removed():
        source = (PROJECT_ROOT / "src" / "models.py").read_text(encoding="utf-8")
        for forbidden in (
            "DeepFM",
            "DCNv2",
            "FTTransformer",
            "TabularTokenizer",
            "HybridLoss",
            "FocalLoss",
        ):
>           assert forbidden not in source
E           assert 'FocalLoss' not in '"""CNN-BiLS...im,\n    )\n'
E             
E             'FocalLoss' is contained here:
E               """CNN-BiLSTM + MLP model approved for the student-performance thesis."""
E               
E               from __future__ import annotations
E               
E               from typing import Any...

tests\test_thesis_pipeline.py:106: AssertionError
=========================== short test summary info ===========================
FAILED tests/test_thesis_pipeline.py::test_forbidden_architectures_and_losses_are_removed
========================= 1 failed, 9 passed in 6.41s =========================
```

### B. Analysis of Test Failure
The test `test_forbidden_architectures_and_losses_are_removed` failed because `FocalLoss` is defined and used in `src/models.py`.
According to `README.md` and the test itself, the thesis rules require that the model only uses standard loss functions (like weighted CrossEntropyLoss or BCEWithLogitsLoss) and **must not use Focal Loss**. However, `FocalLoss` class is defined in `src/models.py` (lines 12-28) and is integrated into the model loss selection in `src/train_pipeline.py` (lines 264-265 and 317-318) for student datasets when `"focal_gamma" in best_params` is true.

This architectural mismatch needs to be resolved by the implementation phase.
