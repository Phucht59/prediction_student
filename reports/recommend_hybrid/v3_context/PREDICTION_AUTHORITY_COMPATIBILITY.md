# Prediction authority compatibility

Survey commit: `356becec7802659c4b9ff20171046871425bfade`.

## High-priority verdict

previous recommendation release **training, threshold selection, EBM features, and Panel A/B cases** use **H1_TABULAR_RESIDUAL_EXPERT** frozen OOF probabilities, not Hybrid CNN-BiLSTM.

Evidence:

- `artifacts/recommend_hybrid/RECOMMEND_HYBRID_CHECKPOINT_MANIFEST.json` `provenance.historical_source_alias = H1_TABULAR_RESIDUAL_EXPERT`, `parameter_count = 160492`, 30 checkpoints, seeds `{42,1201,2026,3407,7319}`, 3 outer folds.
- `src/recommend_hybrid/final/data_builder.py` reads `artifacts/recommend_hybrid/causal/input/landmark_rows.parquet` (absent on `main`; present at `17b519b`) and maps `prediction_risk_probability` → `risk_probability`.
- `hybrid_uncertainty` is **recomputed** as binary entropy of that mean P (`data_builder.py`), not taken from `PredictionResult.uncertainty`.
- `seed_disagreement` is forced to missing (`MISSING_IN_SOURCE_ARTIFACT`).
- `src/recommend_hybrid/prediction_adapter.py` is a **compatibility layer only**. Nothing in the frozen V2 feature table path calls it.

Current prediction authority is Hybrid CNN-BiLSTM (`src/prediction/contracts.py`, `configs/prediction/hybrid_final.json`, `artifacts/prediction/final/FINALIZATION_DECISION.json`).

## Prediction → recommendation input contract

| Field | Phase4 C0 (`PredictionResult`) | V2 rec (`RecommendationFeatures` / `PredictionContext`) | Verdict |
|---|---|---|---|
| identity | `record_id` sha24 | `student_key` = raw `id_student`; `course_key` | **INCOMPATIBLE** without map |
| dataset | `dataset` | implicit OULAD | ADAPT |
| stage | `stage_or_endpoint` `20pct`…`100pct` | `Stage.EARLY_20`…`FINAL_EVALUATION` | SEMANTICALLY_COMPATIBLE_BUT_REVALIDATION_REQUIRED (alias) |
| `risk_probability` | C0 sigmoid / calibrated inner | H1 five-seed OOF mean | **INCOMPATIBLE** numerically |
| `predicted_risk` | binary vs STOP threshold | `predicted_class` | SEMANTICALLY_COMPATIBLE_BUT_REVALIDATION_REQUIRED |
| `threshold` | per-dataset STOP protocol | not stored on `RecommendationFeatures` | ADAPT |
| `uncertainty` | `predict_results` always sets binary Shannon entropy H2(p) of the **single** C0 probability | V2 `data_builder` uses the **same H2 formula** on H1 **mean-of-5-seed** P | SEMANTICALLY_COMPATIBLE_BUT_REVALIDATION_REQUIRED (formula match, distribution shift) |
| seed disagreement | not on `PredictionResult` | required nullable; always NA in V2 table | LEGACY_ONLY / REBUILD |
| checkpoints | two fitted instances `uci`,`oulad`; 3 seeds inner | 30 H1 pt files; 5 seeds × 3 folds × (shared+final) | **INCOMPATIBLE** |
| embeddings | none exposed | `StudentRepresentation` 64-D + 32-D unused by V2 pipeline | LEGACY_ONLY |

## Component verdicts

| Component | Verdict | Why |
|---|---|---|
| `prediction_adapter.py` | DIRECTLY_COMPATIBLE | Maps `PredictionResult` only; unused by V2 artifacts |
| `RecommendationFeatures` schema | SEMANTICALLY_COMPATIBLE_BUT_REVALIDATION_REQUIRED | Field names can be filled from a new builder |
| Feasibility / action catalog | DIRECTLY_COMPATIBLE | No P(risk) except HIGH-only ranking in `candidate_builder` |
| Risk policy thresholds 0.2/0.8/0.6 | INCOMPATIBLE if copied | Selected on H1 P(risk) (`17b519b` `risk_policy/outer_{0,1,2}.json`) |
| Five EBM joblibs | INCOMPATIBLE | Features include `risk_probability` + `hybrid_uncertainty` from H1 |
| Snorkel targets | SEMANTICALLY_COMPATIBLE_BUT_REVALIDATION_REQUIRED | Gemini mostly behavioral; LFs use evidence that may shift |
| Gemini reviews | SEMANTICALLY_COMPATIBLE_BUT_REVALIDATION_REQUIRED | No numeric P in evidence_ids; prompt still had H1 `risk_band` |
| Safety router code | DIRECTLY_COMPATIBLE | Logic reusable; thresholds need revalidation |
| Plans | DIRECTLY_COMPATIBLE | Static templates |
| Simulator on main | LEGACY_ONLY / unwired | Stub; **zero callers** from `recommend()`; protocol wanted frozen Hybrid seed rescoring |
| C0 serving weights | INCOMPATIBLE for production adapter | No Phase4 `.pt` on current surface; H1 rec checkpoints also missing from disk |
| Panel B metrics | LEGACY_ONLY | V2 confirmatory evidence; not a V3 tuning set |
| `StudentRepresentation` | LEGACY_ONLY | H1 embedding contract, unused by explainable V2 |

## What C0 can already supply

- Binary `risk_probability` and `predicted_risk` via `PredictionResult`.
- OULAD cutoff-safe temporal (11) + aggregate (13) + FIT-only static (`src/prediction/data/oulad_features.py`).
- Stage aliases via `canonical_oulad_state`.

## What C0 does not currently supply in V2 shape

- Five-seed OOF ensemble or seed std.
- `due_soon_count`, `regularity_score`, `content_coverage` as defined in `configs/recommend_hybrid/final/feature_contract.yaml`.
- `id_student` as `student_key`.
- Landmark table on `main`.

Machine-readable matrix: `artifacts/recommend_hybrid/v3_context/PHASE4_RECOMMENDATION_COMPATIBILITY.csv`.
