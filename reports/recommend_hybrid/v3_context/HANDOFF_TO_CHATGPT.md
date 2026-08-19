# HANDOFF — Recommendation V3 design (for a new ChatGPT session)

Dense forensic handoff. Do not train, call Gemini, rerun Panel B, or modify Hybrid C0 / V2 production.

## Repo

```text
repo: Phucht59/prediction_student
branch: main
commit: 356becec7802659c4b9ff20171046871425bfade
HEAD == origin/main
scientific V2 source: branch Module_recomend commit 17b519b22e8b69c875d27547d097e6d3b76bc404
Module_recomend tip is NEWER (08f73c72); 17b519b is the frozen scientific release cited by MIGRATION_MANIFEST
```

## Prediction authority (CURRENT, thesis-final)

```text
model_id=hybrid  class=Hybrid  architecture=C0  phase=Phase4
binary risk; outer_test_used_for_phase4_finalization=false
UCI: G3<10 ; states S0/S1/S2
OULAD: Fail/Withdrawn ; states 20pct/35pct/50pct/75pct/100pct
baselines: LR DT RF SVM MLP ; XGB historical only
cutoff: observation_start <= t < cutoff
record_id: sha256("oulad|module|presentation|id_student")[:24]
PredictionResult fields: dataset, record_id, stage_or_endpoint, risk_probability,
predicted_risk, threshold, uncertainty?, model_id="hybrid", metadata
```

Paths: `src/prediction/contracts.py::PredictionResult`, `configs/prediction/hybrid_final.json`, `artifacts/prediction/final/FINALIZATION_DECISION.json`.

## Recommendation current authority (V2)

Released implementation: `src/recommend_hybrid/final/`.

**V2 does NOT consume Phase4 C0.** Actual risk backbone:

```text
historical_source_alias = H1_TABULAR_RESIDUAL_EXPERT
architecture_hash = df5cd885b96e5cea4b840bfc5ca59c08c095f5887df8dd8dcef738edfe8bf70e
parameter_count = 160492
seeds = 42,1201,2026,3407,7319
outer_folds = 3
checkpoint manifest: artifacts/recommend_hybrid/RECOMMEND_HYBRID_CHECKPOINT_MANIFEST.json
```

`src/recommend_hybrid/prediction_adapter.py` only maps `PredictionResult` → dict. V2 `data_builder.py` reads H1 OOF landmark parquet (file **absent on main**; lives on 17b519b).

What landed on main as `src/recommend_hybrid/final/` is **`explainable_v2` renamed**, not `17b519b`'s old conditional-action `final/`. Phase-8 rebuild trees under `artifacts/recommendation/` are a **later overlay**, not V2 Panel B.

C0 `predict_results` **does** fill `uncertainty` as binary Shannon entropy of that single `p` (same formula V2 used on H1 mean-P). No C0 `.pt` is published under `artifacts/hybrid_vnext/phase4/`. H1 `.pt` cited by the rec checkpoint manifest are also **missing** from this checkout (`artifacts/canonical_v3/checkpoints/` absent).

## Exact current pipeline

```text
H1 OOF risk_probability
  → hybrid_uncertainty := binary entropy of mean P  (NOT C0 uncertainty)
  → seed_disagreement := missing (never imputed)
  → stratify_risk (selected on H1: low=0.2 high=0.8 umax=0.6 smax=0.15)
       LOW → INSUFFICIENT_EVIDENCE, empty ranks
       BORDERLINE (incl. high entropy) → HUMAN_REVIEW, empty ranks
       HIGH → hard feasibility (5 actions)
            → FiveEBMRanker (eligible only)
            → safety_router
            → RECOMMEND | HUMAN_REVIEW | INSUFFICIENT_EVIDENCE | NO_FEASIBLE_ACTION
  → RecommendationDecision (top-k only if RECOMMEND or post-rank HUMAN_REVIEW)
```

`plan_builder` and `simulator` are **not called** by `recommend()`. README still lists them. Code: `src/recommend_hybrid/final/pipeline.py::ExplainableRecommendationPipeline.recommend`.

## Five actions

`ASSESSMENT_COMPLETION | RECOVER_ENGAGEMENT | STUDY_REGULARITY | TARGETED_CONTENT_REVIEW | QUIZ_RETRIEVAL_PRACTICE`

Stage map:

| Prediction | Rec stage | Intervene? |
|---|---|---|
| 20pct | EARLY_20 | yes (no TARGETED_CONTENT_REVIEW) |
| 35pct | EARLY_35 | yes |
| 50pct | MIDDLE_50 | yes |
| 75pct | LATE_75 | yes |
| 100pct | FINAL_EVALUATION | **no** (`CandidateAction` forbids it) |

UCI is unused by rec V2.

## Main feature list (EBM)

```text
risk_probability, hybrid_uncertainty, course_progress, inactivity_streak,
active_day_rate, assessments_due, regularity_score, content_coverage,
quiz_activity, missing_assessment_count, due_soon_count, completion_rate,
vle_available, study_material_available, quiz_available, stage
```

`seed_disagreement` excluded (all-NaN). `label_conflict`/`ood_score` routing-only, default 0.0 → gates inactive.

## Label system

Weak supervision Panel A only, 300 cases × 5 actions = 1500 rows; 1499 retained (1 row <2 source families).

Sources: 5 behavioral LF_*_V4 + LF_FEASIBILITY_CONSTRAINT_V4 + REAL_EXTERNAL_GEMINI_REVIEW_V4. Cardinality 0..3. Train-only Snorkel LabelModel, 3 outer folds. Gemini is one family, not three synthetic reviewers.

Frozen: `artifacts/recommend_hybrid/final/weak_labels/`.

`src/recommend_hybrid/final/labeling_functions/core.py` is a **stub**, not the frozen V4 LFs.

## Gemini provenance (DO NOT CALL)

```text
provider: Google Gemini API
prompt_version: external_reviewer_v1
prompt_sha256: f7edfaacd2fad67bf21a175ccc5c0a46abb81b669c08928ab78009c0a24624f3
reviewer_id: gemini_external_reviewer_01
Panel A: 300 cases, 1117 records (flash 77 / flash-lite 1040), abstain=0
Panel B: 150 cases, 557 records (all flash-lite), abstain=0
score: 0/1/2/3
evidence_ids: NEVER include risk_probability
BUT raw prompt JSON includes risk_band + uncertainty_band (from H1) + cutoff + stage + behavioral evidence
protocol yaml says current_model_output_visible=false  → CONFLICT
```

Portability: **CONDITIONALLY_PORTABLE** for all 1674 records. Action semantics unchanged. Not PORTABLE (H1 bands in prompt). Not NON_PORTABLE (reviewer could not cite P(risk) in evidence_ids). Panel B = historical V2 heldout only.

CSV: `artifacts/recommend_hybrid/v3_context/GEMINI_LABEL_PORTABILITY.csv`.

## Panel A / B

- Panel A: development, 300 cases, sampling seed 2026, student-disjoint, training+threshold+EBM+router selection. Frozen.
- Panel B: opened once after development freeze. 150 cases, 557 Gemini. **Must not tune V3.** New confirmatory panel (Panel C) required after C0 rebase. Do not design it now beyond stating the requirement.

## Core metrics (frozen, do not recompute)

Primary: NDCG@3, query-grouped, positive threshold relevance>=1.

Panel B ranker: NDCG@3 `0.9526603067902532`; exact best top1 `0.92`; P@1 `0.9733`; MRR `0.9856`; R@3 `0.8248`; pairwise `0.8354`; invalid-action `0.0`.

Action+stage baseline Panel B: NDCG@3 `0.8275943281032121`.

Bootstrap 2000, seed 2026: mean Δ `+0.12466`, 95% CI `[0.09508, 0.15362]`.

Panel A development (not final): full NDCG@3 `0.97143` vs action+stage `0.88402`. Calibration isotonic rejected (`KEEP_RAW_EBM`, config `a70599afad40`).

## Active baselines that were actually scored

Heldout-reported: **action+stage only**.

Protocol also listed (not all heldout-reported here): rule_severity, global_popularity, logistic_relevance, lambdamart, frozen_neural_action_head. Do not invent extra baselines.

## Known incompatibility with Phase4 C0

```text
LEARNED ARTIFACTS: INCOMPATIBLE without rebuild/retrain
CODE ALGORITHMS (feasibility, 5-EBM class, router, plans): reusable
ADAPTER: prediction_adapter.py exists but unused by V2 data path
IDENTITY: C0 record_id != V2 student_key=id_student
CUTOFF: C0 t<cutoff vs V2 lineage observation_end<cutoff; due_soon_count has no C0 analog
UNCERTAINTY: entropy-of-mean-P vs unspecified C0 uncertainty
SEEDS: V2 5-seed OOF vs C0 3-seed inner; no outer
MAIN LACKS: artifacts/recommend_hybrid/explainable_v2/** and causal/input/**
```

Verdict table: `artifacts/recommend_hybrid/v3_context/PHASE4_RECOMMENDATION_COMPATIBILITY.csv`.

## Portable assets (keep frozen, do not edit)

- Five-action catalog + eligibility policy v4
- Gemini jsonl + hashes + Panel B scores
- Snorkel vote matrix / label parquet (as historical targets)
- EBM joblibs (as V2 baseline, not as C0 production)
- Reports under `reports/recommend_hybrid/final/`
- Scientific source 17b519b (read-only)

## Must-fix gaps (do not implement now)

G01 H1≠C0 risk lineage  
G02 identity join map  
G03 uncertainty is same H2(p) formula but on a different p (H1 5-seed mean vs C0 single logit); still revalidate  
G04 `runtime_authorized` conflict (manifest true vs dataclass false)  
G05 rebuild inputs missing on main (`explainable_v2/`, `causal/`, H1 `.pt`)  
G15 new heldout required; Panel B freeze  
G17 no published C0 serving checkpoint  
G18 plan/simulator unwired  
G20 `stratify_risk` TypeError if `seed_disagreement` is float and cap is `None`  

## Do-not-touch

```text
artifacts/recommend_hybrid/final/heldout/**
artifacts/recommend_hybrid/final/panel_a_reviews/**
artifacts/recommend_hybrid/final/ranker/final_models/**
Panel B metrics JSON/parquet
Gemini records
Hybrid C0 source/metrics
recommendation production code (this KB only added v3_context)
```

Dangerous: any `run_gemini*`, `bootstrap_panel_b`, `rebuild_recommendation_phase8`, `risk_policy_selection.run`. Orphaned pyc under `scripts/recommendation/__pycache__/`.

## Likely next design decisions (do not choose yet)

Compare, using this KB:

1. minimum integration: adapter + C0 risk rebuild + refit EBM + keep actions/feasibility + new Panel C  
2. safe incremental: (1) + real label_conflict + ranked HUMAN_REVIEW + personalized plans + selective Gemini subset  
3. larger redesign: listwise ranker / new actions / UCI rec — higher scientific risk

Do **not** copy H1 thresholds 0.2/0.8 onto C0. Do **not** use Panel B to pick among 1/2/3.

## Exact important paths

```text
src/recommend_hybrid/final/pipeline.py
src/recommend_hybrid/final/ranker.py
src/recommend_hybrid/final/data_builder.py
src/recommend_hybrid/prediction_adapter.py
src/prediction/contracts.py
artifacts/recommend_hybrid/RECOMMEND_HYBRID_CHECKPOINT_MANIFEST.json
artifacts/recommend_hybrid/final/release/FINAL_RELEASE_MANIFEST.json
artifacts/recommend_hybrid/final/heldout/PANEL_B_FINAL_HELDOUT_METRICS.json
artifacts/recommend_hybrid/final/ranker/FIVE_EBM_MANIFEST.json
configs/recommend_hybrid/final/recommendation.yaml
reports/recommend_hybrid/v3_context/MASTER_RECOMMENDATION_KB.md
```
