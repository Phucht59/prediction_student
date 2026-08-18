# Final System Evidence

## 1. Final prediction architecture

The active prediction path remains one `Hybrid` architecture (`model_id=hybrid`, `display_name=Hybrid`): static and aggregate projectors, temporal CNN and BiLSTM branches, F3 adaptive entropy fusion, one binary logit, and sigmoid `P(Risk)`. UCI, OULAD Early, and OULAD FINAL-100 are separate fitted instances of the same class.

## 2. Binary task contracts

UCI Combined uses `G3 < 10 -> Risk=1`, with G3 excluded from predictors and S0/S1/S2 as views. OULAD uses Fail/Withdrawn -> Risk=1 and Pass/Distinction -> Risk=0. These prediction contracts were previously accepted and were not changed in this gate.

## 3. Reconstructed prediction provenance

Recommendation features contain 179 `OOF_INNER_VALIDATION` rows and 121 `FINAL_OUTER0_HOLDOUT_INFERENCE_NOT_OOF` rows. The prediction identity is reconstructed Hybrid only; no H1 identity is active.

## 4. Recommendation architecture

The recommendation code path uses canonical actions, feasibility, five independent EBMs, risk stratification, and fail-closed safety routing. The current learned artifacts are not eligible for final scientific evaluation because their protected holdout scope was contaminated.

## 5. Holdout isolation proof

The protected set was identified before outcome/relevance access. It contains 121 unique queries, 36 students, and 36 groups. Direct actual-fit-frame overlap is 121 rows, 121 queries, 36 students, and 36 groups. The nonholdout feature partition alone has zero student/group overlap, but the actual recommendation fit frame included the protected rows.

## 6. Recommendation freeze

No scientific freeze was created. `FROZEN_RECOMMENDATION_POLICY.json` and `RECOMMENDATION_FINAL_FREEZE.json` are explicit non-freeze sentinels with status `NOT_FROZEN_HOLDOUT_CONTAMINATION` / `NOT_CREATED_HOLDOUT_CONTAMINATION`.

## 7. Held-out population

The 121-row outcome population was not opened. No rows were dropped, filtered, or evaluated.

## 8. NDCG@3 result

Not computed. The existing 300-query development NDCG is contaminated for this protected split and is not a final held-out result.

## 9. Secondary metrics

Not computed because the fail-closed gate stopped before holdout access.

## 10. Action-level results

Not computed. No action-level claim is supported by this run.

## 11. Safety results

Not evaluated on the protected population. A safety result cannot be claimed without a valid final evaluation population.

## 12. Baseline comparison

Not run. No baseline can be compared to an unopened contaminated holdout.

## 13. Historical evidence distinction

Historical Panel B evidence remains attached to its original recommendation identity. It was not merged into this run and cannot rescue the contaminated protected split.

## 14. Final supported claims

Supported: the prediction system remains accepted as a reconstructed protocol-faithful Hybrid system; the recommendation code path and artifact provenance can be audited; contamination was detected before holdout outcome access.

Not supported: any new held-out NDCG, superiority, safety, action-level, or baseline-comparison claim for the rebuilt recommender.

## 15. Limitations

The current recommendation rebuild used a 300-query feature/training scope that included the 121 rows intended for final holdout evaluation. A new untouched recommendation holdout and clean rebuild are required.

## 16. Research closure statement

Prediction research is closed under the prior prediction acceptance. Recommendation research is **not closed**. No Hybrid retraining, EBM retraining, HPO, threshold tuning, outer rerun, commit, or push was performed in this gate.
