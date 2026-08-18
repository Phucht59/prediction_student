# Recommendation final summary

## 1. Final architecture

Frozen Hybrid CNN-BiLSTM risk scores feed a Student State grain of student × module × presentation × stage (20/35/50/75; no FINAL). Five independent Explainable Boosting Regressors produce action relevance in [0, 3]. Feasibility v2 is a hard operational layer. Ranking returns top 3 releasable actions with model-faithful local explanations. PostgreSQL persists catalog identity, state snapshots, runs, scores, explanations, and plans.

## 2. Weak labeling / silver labels

Phase 6 froze Gemini 3.5, Gemma 4, Gemini 3.1 (A4 only), and Behavior as weak sources. Phase 7 aggregated them independently: Snorkel for A1/A2/A3/A5; TWO_SOURCE_CONSENSUS for A4 (Snorkel requires ≥3 LFs). Silver rows: 2500 total, 1641 VALID, 548 NO_WEAK_EVIDENCE, 311 REVIEW. NO_WEAK_EVIDENCE is not relevance 0.

## 3. Five EBM models

Target is `expected_relevance`. Features exclude `course_progress` (stage/100) and `risk_band`. Training rows: A1 141, A2 500, A3 500, A4 500, A5 311 REVIEW. OOF MAE/RMSE: 0.036543/0.099083, 0.256669/0.355348, 0.165778/0.232322, 0.367609/0.454298, 0.123467/0.152801. A1 n is small and targets concentrate near 3.

## 4. Feasibility + ranking

A4 Progress Monitoring uses feasibility v2: FEASIBLE / PROGRESS_STATE_OBSERVED. INFEASIBLE actions are not released. UNKNOWN is NEEDS_VERIFICATION. A5 remains REVIEW and becomes REVIEW_REQUIRED when selected. Plan REVIEW if the top released action needs review. No score threshold for NO_ACTION.

## 5. Panel B automated-reference protocol

150 sealed cases, 0 overlap with training identities. Dual Gemini-family labels only. 750 case-actions: 594 DUAL_SOURCE, 156 NO_REFERENCE. ABSTAIN was never mapped to 0. This is AUTOMATED_REFERENCE_EVALUATION, not expert ground truth.

## 6. Final results table

Panel B NDCG@3: EBM 0.927, Random Forest 0.923, Ridge 0.910, Action-Stage Prior 0.824. EBM P@1 0.753, Recall@3 0.849, MRR 0.918, pairwise 0.810, invalid 0, coverage 1.00.

## 7. Bootstrap comparison

2000 case resamples, seed 2026. EBM − Prior +0.103 [0.075, 0.131]. EBM − Ridge +0.016 [0.004, 0.030]. EBM − RF +0.004 [−0.004, 0.012].

EBM achieved the highest NDCG@3 point estimate on Panel B under the automated-reference evaluation. EBM clearly outperformed Action-Stage Prior and Ridge based on paired bootstrap deltas whose 95% intervals were above zero. EBM and Random Forest were statistically indistinguishable under the paired bootstrap interval used here; EBM retained the advantage of glass-box local/global interpretability.

## 8. Interpretability

Global term importances and local EBM contributions are stored. Explanations refer to the raw EBM score, not the clipped operational score.

## 9. Limitations

Panel B references are automated and same-family. A1 has only 41/150 dual numeric references. A5 weak-label conflict remains (REVIEW). Relevance is not evidence that an intervention will improve student outcomes.

## 10. Runtime / database

`RecommendationService` loads the freeze manifest, verifies checksums, scores five EBMs, applies feasibility v2, ranks, and optionally persists one atomic transaction. CLI: `scripts/recommendation/recommend_student.py`. Migration extends the existing catalog/recommendation schemas without dropping user data.

## 11. Reproducibility artifacts

See `artifacts/recommendation/final/FINAL_RECOMMENDATION_FREEZE_MANIFEST.json` and `THESIS_RECOMMENDATION_SOURCE_OF_TRUTH.json`. Validate with `scripts/recommendation/validate_final_freeze.py`.
