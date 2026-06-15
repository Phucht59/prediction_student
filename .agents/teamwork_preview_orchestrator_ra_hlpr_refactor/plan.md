# Plan - RA-HLPR Refactoring

## Phase 1: Recommender Code & Pipeline Refactoring
1. **Exploration**:
   - Understand existing data formats (`student-mat`, `student-por`, `xapi`).
   - Find existing features related to academic performance, attendance, engagement, study time.
   - Check what models exist, what the downstream prediction inputs look like.
   - Understand the current evaluation scripts and how document generation (`generate_doc.py`) is structured.
2. **Implementation of downstream components**:
   - `src/recommender/risk_rules.py`: Define rules to generate weak labels for the 6 risks (R1 to R6) based on available features in the dataset.
   - `data/recommender/intervention_catalog.csv`: Create standard catalog with at least 10 items.
   - `src/recommender/hybrid_scorer.py`: Score calculation based on the specified weight formula.
   - `src/recommender/candidate_generator.py`: Generates recommendation candidates based on risk profiles.
   - `src/recommender/path_planner.py`: Generate 4-week personalized learning path (Stabilize, Practice, Reinforce, Evaluate & Adjust).
   - `src/recommender/explanation.py`: Generate explanations for why interventions are recommended.
   - `src/evaluation/recommender_metrics.py` & `src/evaluation/path_quality.py`: Implement ranking and path quality metrics.
   - `scripts/run_recommender_pipeline.py`: Run end-to-end for a given dataset, output prediction files, recommendations, paths, metrics, and markdown reports.
3. **Verification**:
   - Run tests and script validation on `student-mat` and other valid datasets.

## Phase 2: Document Generation & Report Refactoring
1. **Modify `generate_doc.py`**:
   - Update Chapter 3 (3.5.1 to 3.5.5) to describe RA-HLPR architecture, Risk Diagnosis Head, Mapping Rules, Hybrid Scorer, Path Planner, and Explanation.
   - Update Chapter 4 (4.4) to show the evaluation table of the recommender metrics.
   - Add explanation about weak labels and limitations.
   - Run the script to produce `Bao_cao_cuoi_cung.docx` instead of `Bao_cao_tien_do.docx`.
2. **Generate `outputs/recommender/final_recommender_section.md`**:
   - Store report section content in markdown for user reference.

## Verification & Audit
- Run Verification check with Reviewers, Challengers, and Forensic Auditor.
- Ensure that the prediction accuracy/F1 metrics on test data of CNN-BiLSTM + Context MLP are absolutely untouched.
