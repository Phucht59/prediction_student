# Handoff Report — RA-HLPR Exploration & Codebase Analysis

This report presents a read-only investigation of the student academic performance prediction and Risk-Aware Hybrid Learning Path Recommender (RA-HLPR) system.

---

## 1. Observation

### A. Recommender Implementation and Pipeline
The recommender module is implemented in the `src/recommender/` directory, consisting of:
* **`src/recommender/rules.py`**: Heuristic rules generating 6 weak labels from raw student features.
  * *Student dataset rules* (lines 22-41):
    ```python
    absences = _number(record, "absences")
    study_time = _number(record, "studytime", 1.0)
    failures = _number(record, "failures")
    g1 = _number(record, "G1")
    g2 = _number(record, "G2")
    alcohol = _number(record, "Dalc", 1.0) + _number(record, "Walc", 1.0)
    goout = _number(record, "goout", 1.0)
    ratio = absences / max(study_time, 0.5)
    
    targets.append(
        [
            float(absences >= 10 or ratio >= 5),          # R1: attendance
            float(failures > 0),                          # R2: failure_history
            float(g2 < 10 or (g1 > 0 and g2 < g1)),       # R3: grade_gap
            float(study_time <= 1),                       # R4: study_time
            float(alcohol >= 6),                          # R5: wellbeing
            float(goout >= 4),                            # R6: time_management
        ]
    )
    ```
  * *xAPI dataset rules* (lines 43-52):
    ```python
    targets.append(
        [
            float(str(record.get("StudentAbsenceDays", "")).strip().lower() == "above-7"), # R1: attendance
            float(_number(record, "VisITedResources") < 40),                                # R2: resource_usage
            float(_number(record, "raisedhands") < 30 or _number(record, "Discussion") < 30), # R3: class_engagement
            float(_number(record, "AnnouncementsView") < 30),                               # R4: course_updates
            float(str(record.get("ParentAnsweringSurvey", "")).strip().lower() == "no"),     # R5: parent_support
            float(str(record.get("ParentschoolSatisfaction", "")).strip().lower() == "bad"),  # R6: school_support
        ]
    )
    ```
* **`src/recommender/risk_head.py`**: A 3-layer MLP (`RiskDiagnosisHead`) predicting 6 academic risks.
  * Inputs: Normalized student features concatenated with 3-class academic performance probabilities (`[Low, Medium, High]`).
  * Loss: `nn.BCEWithLogitsLoss` using class-imbalance positive weighting (`pos_weight`).
* **`src/recommender/knowledge_base.py`**: Manages intervention items and their mappings.
  * `DEFAULT_CATALOG` defines 12 interventions (e.g. `attendance_monitoring`, `time_planning`, `counselor_meeting`, etc.).
  * `DEFAULT_MAPPINGS` maps risks (e.g. `attendance`, `time_management`) to specific intervention IDs.
* **`src/recommender/hybrid_scorer.py`**: Implements `HybridScorer.score_student` combining 6 weighted criteria:
  * `risk_match` (0.30)
  * `performance_need` (0.20)
  * `difficulty_fit` (0.15)
  * `time_fit` (0.15)
  * `prerequisite_fit` (0.10)
  * `expected_effect` (0.10)
* **`src/recommender/path_planner.py`**: Schedules top-6 scored interventions into a 4-week study plan:
  * Weeks: Week 1 (Stabilize), Week 2 (Practice), Week 3 (Reinforce), Week 4 (Evaluate & Adjust).
  * Risk bands: Stable, Moderate, High (based on performance predictions and max risk score).

The pipeline script is at **`scripts/run_recommender_pipeline.py`** and coordinates:
1. Loading split datasets (train pool and locked test).
2. Generating 11-seed ensemble class probabilities for performance classification.
3. Constructing weak-supervision risk labels and extracting features.
4. Training the `RiskDiagnosisHead` MLP.
5. Evaluating risk predictions, ranking, and path quality.
6. Writing outputs to `outputs/recommender/`.

---

### B. Dataset Feature Risk Mapping
The mapping between the 6 required academic risks and raw features in `student-mat`/`student-por` (separator: `;`) and `xapi` (separator: `,`) was observed:

| Required Risk | Dataset | Raw Features Mapped | Logical Definition / Rule in Code |
| :--- | :--- | :--- | :--- |
| **R1_LOW_PRIOR_PERFORMANCE** | `student` | `failures` | `failures > 0` |
| | `xapi` | *None* | **Not Available** (No past academic records/failures exist in xAPI schema). |
| **R2_DECLINING_TREND** | `student` | `G1`, `G2` | `G2 < 10` or `(G1 > 0 and G2 < G1)` |
| | `xapi` | *None* | **Not Available** (xAPI is cross-sectional; no historical grades recorded). |
| **R3_ATTENDANCE_RISK** | `student` | `absences`, `studytime` | `absences >= 10` or `(absences / studytime) >= 5` |
| | `xapi` | `StudentAbsenceDays` | `StudentAbsenceDays == "Above-7"` |
| **R4_LOW_ENGAGEMENT** | `student` | `goout`, `freetime` *(Proxy)* | Mapped in code to `wellbeing` (`Dalc+Walc >= 6`) and `time_management` (`goout >= 4`). No direct LMS interaction metrics are available. |
| | `xapi` | `raisedhands`, `VisITedResources`, `Discussion`, `AnnouncementsView` | `VisITedResources < 40`, `raisedhands < 30` or `Discussion < 30`, `AnnouncementsView < 30` |
| **R5_INSUFFICIENT_STUDY_TIME**| `student` | `studytime` | `studytime <= 1` (equivalent to <= 2 hours/week) |
| | `xapi` | *None* | **Not Available** (Study time hours are not recorded in xAPI schema). |
| **R6_HIGH_FAILURE_PROBABILITY**| `student` | *Derived* | Inferred via the downstream classification's `Low` performance probability, or proxy features like `failures > 0` and low grades. |
| | `xapi` | *Derived* | Inferred via the downstream classification's `Low` performance probability. |

---

### C. Structure of `generate_doc.py`
The script `generate_doc.py` creates `Bao_cao_cuoi_cung.docx` (the final thesis report) using `python-docx`.
* **Section 3.5 Generation (lines 151-155)**:
  ```python
  doc.add_paragraph("\n3.5. Hệ thống Khuyến nghị Lộ trình Học tập Hỗn hợp Thích ứng Rủi ro (RA-HLPR)")
  doc.add_paragraph("Hệ thống Khuyến nghị Lộ trình Học tập Hỗn hợp Thích ứng Rủi ro (Risk-Aware Hybrid Learning Path Recommender - RA-HLPR)...")
  # (Paragraphs on Risk Diagnosis Head, Hybrid Scorer, and Learning Path Planner)
  ```
  It is followed by `doc.add_page_break()` at line 156.
* **Section 4.4 Generation (lines 169-278)**:
  * Line 169: Adds header: `doc.add_paragraph("4.4. Kết quả đánh giá hệ thống khuyến nghị RA-HLPR")`
  * Line 170-171: Explains evaluation across Risk Diagnosis, Ranking, and Path Quality.
  * Lines 172-184: Loads metrics from JSON files `outputs/recommender/recommender_metrics_<dataset>.json`.
  * Lines 191-235: Formats and inserts **Bảng 4.1**: "Kết quả chẩn đoán rủi ro và xếp hạng can thiệp của RA-HLPR" (Micro F1, Macro F1, Precision@3, NDCG@3, Catalog Coverage).
  * Lines 237-277: Formats and inserts **Bảng 4.2**: "Đánh giá chất lượng lộ trình học tập 4 tuần" (Risk Coverage, Difficulty Progression, Prerequisite Violation, Workload Balance).

#### Insertion Locations:
1. **New Sections 3.5.1 to 3.5.5 and Weak Labels Description**:
   * Must be inserted between line 154 (the last paragraph of Section 3.5) and line 156 (`doc.add_page_break()`).
   * Specifically:
     * **3.5.1**: Đầu chẩn đoán rủi ro (Risk Diagnosis Head MLP architecture, loss, and training).
     * **3.5.2**: Cơ sở tri thức can thiệp (Intervention Knowledge Base catalogs and mappings).
     * **3.5.3**: Cơ chế gán nhãn yếu (Weak Supervision Rule Engine rules for `student` and `xapi` datasets).
     * **3.5.4**: Bộ chấm điểm hỗn hợp (Hybrid Scorer weighted multi-criteria ranking formula and student capacity adjustment).
     * **3.5.5**: Bộ lập kế hoạch lộ trình học tập (Learning Path Planner 4-week thematic scheduling and risk band assessment).
2. **Limitations, Evaluation Metrics Definitions**:
   * **Evaluation Metrics Definitions**: Should be placed in a new sub-section **3.5.6** or **4.4.1** (before the results tables) to define mathematically how F1, P@K, NDCG, risk coverage, difficulty progression, prerequisite violations, and workload standard deviation are computed.
   * **Limitations**: Can be appended at the end of Section 3.5 (before the page break) or included in **5.2 (Hướng phát triển)**. The limitations must address the lack of longitudinal or causal verification (fidelity evaluation against weak supervision rather than actual educational outcomes).

---

### D. Location and Format of Artifacts

#### 1. Recommender Outputs
All outputs of the recommendation pipeline are located in `outputs/recommender/`:
* **`risk_predictions_<dataset>.csv`**:
  * Columns: `student_index`, followed by the 6 risk labels (e.g. `attendance`, `failure_history`, `grade_gap`, `study_time`, `wellbeing`, `time_management` for student).
  * Format: Floating point probabilities (0.0 to 1.0).
* **`recommendation_results_<dataset>.csv`**:
  * Columns: `student_index`, `rank`, `item_id`, `intervention_name`, `score`, `explanation`.
  * Format: Top 5 ranked interventions with detailed multi-criteria scoring breakdowns.
* **`learning_paths_<dataset>.json`**:
  * Format: JSON mapping each `student_index` to a 4-week learning path (`theme`, `objective`, `recommended_actions`, `expected_outcome`, `explanation`, `item_ids`) and a designated `risk_band`.
* **`recommender_metrics_<dataset>.json`**:
  * Format: Evaluation metrics JSON (contains `"risk_diagnosis"`, `"ranking"`, `"path_quality"` performance metrics).
* **`recommender_report_<dataset>.md`**:
  * Format: Markdown summary report of metrics and selected case studies.
* **`intervention_catalog.csv`** & **`risk_intervention_mapping.csv`**:
  * Format: Interventions metadata and their target risk mappings.

#### 2. Models
* **Classifier Checkpoints**:
  * Located at `models/saved/final/` as PyTorch weights (e.g., `student-mat_3class_cnn_bilstm_mlp_seed123.pt`, etc.).
  * Hyperparameters: `models/saved/final/<dataset>_3class_best_params.json`.
* **Recommender MLP Checkpoints**:
  * Located at `models/recommendation/` (e.g. `student-mat_mlp.pt`, `xapi_mlp.pt`).
  * Contains the PyTorch state dictionary, feature scaling mean/scale, input/output dims, seed, and dataset details.

#### 3. Prediction CSVs & Performance Metrics
* **Predictions**:
  * Located at `reports/final/predictions/` (e.g. `student-mat_3class_predictions.csv`).
  * Columns: Original features, `True_Label`, `Pred_Label`, `Confidence`, and probability scores `Prob_Class_0`, `Prob_Class_1`, `Prob_Class_2`.
* **Evaluation Metrics**:
  * Located at `reports/final/metrics/` (e.g. `student-mat_3class_locked_test_metrics.json` and `student-mat_3class_fixed_cv.json`).
  * Format: Acc, F1-Macro, Precision-Macro, Recall-Macro, RMSE, R2.

---

## 2. Logic Chain

1. **Rule Engine & Feature Constraints**:
   * *Observation*: Raw dataset columns and `rules.py` definitions show that `student` datasets record prior grades (`G1`, `G2`) and `failures`, but lack interaction activity. Conversely, `xapi` records detailed behavioral clickstream events but lacks past performance grades and study time features.
   * *Inference*: Therefore, risks `R1` (Prior Performance), `R2` (Declining Trend), and `R5` (Study Time) are not natively representable in the `xapi` dataset. Conversely, direct classroom/LMS engagement (`R4`) is not natively representable in the `student` datasets.
   * *Inference*: In the code, the 6 risks for the `student` dataset are completed by adding wellbeing (`Dalc` + `Walc`) and time management (`goout`), while `xapi` maps these to parent support and school satisfaction.

2. **Recommender Execution Flow**:
   * *Observation*: `scripts/run_recommender_pipeline.py` requires ensemble classification outputs (member models trained using 11 seeds) to extract class probabilities (`test_class_probs`).
   * *Inference*: The classifier models in `models/saved/final/` and their respective configs must be fully trained and available before running the recommender pipeline.

3. **Word Document Insertion Strategy**:
   * *Observation*: Section 3.5 ends on line 155, right before the page break on line 156.
   * *Inference*: Subsections 3.5.1 to 3.5.5 must be appended to the Document object in python-docx *before* line 156 to remain under Chapter 3.
   * *Observation*: Section 4.4 loads metrics using `outputs/recommender/recommender_metrics_<dataset>.json` to generate tables.
   * *Inference*: The recommender pipeline `scripts/run_recommender_pipeline.py` must be executed first for all three datasets (`student-mat`, `student-por`, `xapi`) to ensure the JSON files exist in `outputs/recommender/`, otherwise `generate_doc.py` will fail or default.

---

## 3. Caveats

* **Execution Restrictions**: As an Exploration agent with a read-only mandate, no code was executed, modified, or tested. All observations are based purely on static code analysis of files and outputs.
* **Database URL**: Although `src/config.py` contains PostgreSQL configuration, the local environment uses raw CSV files in `data/processed/final/` for pipeline execution.

---

## 4. Conclusion

The RA-HLPR system successfully implements a multi-stage risk-aware recommender pipeline. However, due to raw data differences, several requested risks cannot be mapped to both datasets (e.g., `xapi` lacks historical performance and study time, whereas `student` lacks digital engagement). Modifications to `generate_doc.py` should be inserted at the end of Section 3.5 (before page break) and inside Section 4.4, using the calculated metrics in `outputs/recommender/`.

---

## 5. Verification Method

To independently verify the paths and files described:
1. Confirm file presence in the shell:
   ```powershell
   Test-Path "c:\Huflit\kltn\src\recommender\rules.py"
   Test-Path "c:\Huflit\kltn\scripts\run_recommender_pipeline.py"
   Test-Path "c:\Huflit\kltn\generate_doc.py"
   Test-Path "c:\Huflit\kltn\outputs\recommender\recommender_metrics_student-mat.json"
   ```
2. Verify that the recommender metrics format corresponds to the observation by inspecting:
   `c:\Huflit\kltn\outputs\recommender\recommender_metrics_student-mat.json`
