# MASTER Recommendation V3 forensic knowledge base

Survey-only. Commit `356becec7802659c4b9ff20171046871425bfade` on `main`.  
No training, no Gemini, no Panel B rerun, no C0 or V2 production edits.

Companion files: `HANDOFF_TO_CHATGPT.md`, `ARCHITECTURE_AND_CALL_GRAPH.md`, `DATA_LABEL_LINEAGE.md`, `PREDICTION_AUTHORITY_COMPATIBILITY.md`, `GEMINI_LABEL_PORTABILITY.md`, and `artifacts/recommend_hybrid/v3_context/*`.

---

## 1. Executive state

| Item | State |
|---|---|
| Thesis prediction | Phase4 Hybrid C0, binary, inner 3×3, outer not opened |
| Recommendation on main | V2 explainable five-EBM, frozen Panel B |
| What V2 actually scores on | **H1_TABULAR_RESIDUAL_EXPERT** OOF risk, not C0 |
| Gemini | 1117 Panel A + 557 Panel B, frozen, conditionally portable |
| Panel B | Historical V2 heldout; forbidden as V3 tune set |
| Main completeness | Production code + frozen hashes present; rebuild inputs (`explainable_v2/`, `causal/`) **not** copied |
| Authorization conflict | Release manifest `runtime_authorized=true`; dataclass forbids `True` |

`KB_COMPLETE` requires the files listed in the prompt §31; see `KB_MANIFEST.json`.

---

## 2. Current prediction authority

Verified: Hybrid, C0, Phase4 (`PROJECT.md`, `configs/prediction/hybrid_final.json`, `artifacts/prediction/final/FINALIZATION_DECISION.json`).

```text
UCI: G3<10 ; S0/S1/S2
OULAD: Fail/Withdrawn ; 20/35/50/75/100 as states of one fitted model
availability: CNN and BiLSTM gated by temporal_available; aggregate independent
preprocessing: FIT-only ContextPreprocessor + MaskedStandardScaler
cutoff: observation_start <= event_time < cutoff
```

`PredictionResult` (`src/prediction/contracts.py`): `dataset`, `record_id`, `stage_or_endpoint`, `risk_probability∈[0,1]`, `predicted_risk∈{0,1}`, `threshold∈(0,1)`, `uncertainty?`, `model_id="hybrid"`, `metadata`.  
`recommendation_features()` emits `student_key=record_id` plus optional `hybrid_uncertainty`.

Not H1, not Phase8, not `cnn_bilstm_*`, not XGB.

---

## 3. Recommendation V2 architecture

Public: `src/recommend_hybrid/final/` exported from `src/recommend_hybrid/__init__.py` as `RecommendationPipeline = ExplainableRecommendationPipeline`.

One frozen Hybrid risk (historically H1) → risk bands → hard feasibility → **five independent EBMs** → safety router → plan template → simulator stub.

Statuses: `RECOMMEND`, `INSUFFICIENT_EVIDENCE`, `HUMAN_REVIEW`, `NO_FEASIBLE_ACTION`.

Scientific source: `Module_recomend` @ `17b519b22e8b69c875d27547d097e6d3b76bc404` (`MIGRATION_MANIFEST.json`, README).

---

## 4. Code authority map

| Class | Paths |
|---|---|
| ACTIVE_RUNTIME | `final/pipeline.py`, `risk_policy.py`, `feasibility.py`, `action_eligibility.py`, `ranker.py`, `safety_router.py`, `plan_builder.py`, `final/contracts.py`, package `__init__` |
| ACTIVE_DEV | `data_builder.py`, `candidate_builder.py`, `query_evidence.py`, `weak_labels.py`, `metrics.py`, `calibration.py`, `sampling.py`, `audits.py`, tests under `tests/recommendation/final/` |
| COMPATIBILITY | `prediction_adapter.py`, `legacy_annotation_adapter.py`, `provider_envelope.py` (broken import) |
| FROZEN_EVIDENCE | `artifacts/recommend_hybrid/final/**`, `configs/recommend_hybrid/final/**`, `reports/recommend_hybrid/final/**`, H1 checkpoint manifest |
| HISTORICAL | `imbalance.py`, `artifacts/recommend_hybrid/imbalance/`, reconstruction scripts, `17b519b` `explainable_v2/` |
| LEGACY / DEAD | `StudentRepresentation` 64/32-D; `labeling_functions/core.py` stubs; orphaned `scripts/recommendation/__pycache__/*.pyc` without `.py` |
| UNCERTAIN | none material after survey |

Full CSV: `artifacts/recommend_hybrid/v3_context/RECOMMENDATION_FILE_CLASSIFICATION.csv`.

---

## 5. Data flow

See `ARCHITECTURE_AND_CALL_GRAPH.md`.

Critical: `data_builder.build` **refuses** to run without

- `artifacts/recommend_hybrid/RECOMMEND_HYBRID_CHECKPOINT_MANIFEST.json` (present)
- `artifacts/recommend_hybrid/explainable_v2/run_state/HYBRID_OOF_AUTHORITY_AUDIT.json` (**absent**)
- `artifacts/recommend_hybrid/causal/input/landmark_rows.parquet` (**absent**)

So V2 can **score** a hand-built `RecommendationFeatures` with frozen joblibs, but cannot **rebuild** the scientific table on `main`.

---

## 6. Feature lineage

See `DATA_LABEL_LINEAGE.md` and `configs/recommend_hybrid/final/feature_contract.yaml`.

Prediction features vs rec evidence vs label-derived vs routing-only are distinguished there. Phase4 can reproduce cutoff-safe VLE/assessment tensors; it is **not** shown to reproduce V2 `regularity_score`, `content_coverage`, or `due_soon_count` byte-for-byte.

---

## 7. Actions

Canonical set verified in `final/contracts.py::CanonicalAction` and Gemini enums.

| Action | Purpose | Evidence | Stage bar | Eligibility (v4) | Contra | EBM | Gemini A/B | Plan |
|---|---|---|---|---|---|---|---|---|
| ASSESSMENT_COMPLETION | close due/missing work | missing/due_soon | all intervene | missing>0 or due_soon>0 | NO_OPEN_ASSESSMENT, EXTENSION_PENDING | yes 300/300 | 184 / 97 | submit 24h early |
| RECOVER_ENGAGEMENT | restart VLE | inactivity, active_day_rate | all | VLE on and rate<0.5 | NO_VLE_ACCESS | 299 retained | 230 / 119 | 4 active days/week |
| STUDY_REGULARITY | even cadence | regularity, active_day_rate | all | regularity<0.8 or rate<0.8 | ACUTE_PERSONAL_CIRCUMSTANCE | 300 | 292 / 141 | gap <3 days |
| TARGETED_CONTENT_REVIEW | cover material | content_coverage | not EARLY_20 | coverage<0.8 + materials | NO_STUDY_MATERIAL, ASSESSMENT_OVERLOAD | 300 | 115 / 63 | 80% chapter |
| QUIZ_RETRIEVAL_PRACTICE | retrieval | quiz_available/activity | all | quiz_available | NO_PRACTICE_MATERIAL, ASSESSMENT_OVERLOAD | 300 | 296 / 137 | self-quiz ≥70% |

Legacy names (`STUDY_SCHEDULE`, `LEARNING_CONSOLIDATION`, …) only via `legacy_annotation_adapter.py`; not reachable from the public pipeline.

Measurable targets are template strings, not logged outcomes. Known failure: `TARGETED_CONTENT_REVIEW` never won Panel A top-1 (0/299).

---

## 8. Risk routing

`stratify_risk` (`final/risk_policy.py`):

- `P < low` → LOW
- entropy > `maximum_automatic_uncertainty` **or** seed_disagreement > cap → BORDERLINE
- `P >= high` → HIGH
- else BORDERLINE

Selected on H1, all 3 folds (`17b519b` `risk_policy/outer_*.json`): **low=0.2, high=0.8, umax=0.6, smax=0.15**. Not stored in `configs/recommend_hybrid/final/recommendation.yaml` (only grids).

`query_evidence.py` independently paints bands at **0.3 / 0.6** — conflict.

LOW → no actions. BORDERLINE → HUMAN_REVIEW with **empty** ranks. Only HIGH enters feasibility+EBM.

---

## 9. Feasibility

Deterministic, fail-closed (`feasibility.py` + `action_eligibility.py`). Unknown availability → ineligible. `FINAL_EVALUATION` cannot be a `RecommendationFeatures.stage`.

Independent of P(risk) except that `candidate_builder` only evaluates eligibility when band==HIGH.

---

## 10. Gemini labeling

See `GEMINI_LABEL_PORTABILITY.md`. Do not regenerate.

Prompt version `external_reviewer_v1`. Score 0–3. Zero abstain in frozen files. Reviewer cites only behavioral/availability `evidence_ids`. Prompt still contained H1 `risk_band` / `uncertainty_band`.

---

## 11. Weak supervision

`weak_labels.py`: Snorkel `LabelModel` cardinality 4, ABSTAIN=-1, min confidence + min 2 families. Fold models in `label_model_manifest.json`.

Gemini is the strongest non-feasibility source (weights ~0.78–0.86). Leave-one-source-out: dropping Gemini drops Panel A NDCG by ~0.031, still not “catastrophic” under the 0.05 gate (`PANEL_A_RELEASE_GATES.json`).

Dropping the entire BEHAVIORAL family is unsupported (Snorkel needs ≥3 LFs) — recorded, not a silent pass.

Refit later after evidence rebase. Do not refit now.

---

## 12. Five EBM ranker

`FiveEBMRanker` (`final/ranker.py`). `interpret.glassbox.ExplainableBoostingRegressor`. Config `a70599afad40`. Interactions=3, lr=0.025, max_bins=64, max_rounds=2000, min_samples_leaf=20, outer_bags=8, seed 2026.

Uses `risk_probability` and `hybrid_uncertainty`. Does **not** use `seed_disagreement` or `action_id`. Therefore **cannot** be reused on C0 P(risk) without retraining.

Hashes in `FIVE_EBM_MANIFEST.json` / `FINAL_RELEASE_MANIFEST.json`. Native scale ordinal 0–3; public `clip(pred/3,0,1)`. Calibration rejected on Panel A (`RANKER_SELECTION_BOOTSTRAP.json` `KEEP_RAW_EBM_RANKER`: NDCG CI for isotonic vs raw includes 0 under the locked rule).

Separate action models are protocol-justified (different evidence, different LF support, no action-id shortcut).

---

## 13. Baselines

**Evaluated and frozen on Panel B:** action+stage-only (`panel_a_action_stage_only_baseline` in `PANEL_B_FINAL_HELDOUT_METRICS.json`).

**Evaluated on Panel A gates:** same baseline + context-permutation audit (`PANEL_A_RELEASE_GATES.json`).

**Listed in YAML, not claimed as heldout winners here:** rule_severity, global_popularity, logistic_relevance, lambdamart, frozen_neural_action_head (`recommendation.yaml` challengers). Do not invent scores.

Reusable for V3 comparison as a **frozen V2 reference**, not as a C0-fair baseline until the same candidate pool is rebuilt.

---

## 14. Safety router

`route_ranked_actions` + frozen thresholds (`ROUTER_FREEZE_MANIFEST.json`):

```text
minimum_top1_score=0.5
minimum_top1_margin=0.0
maximum_hybrid_uncertainty=0.6
maximum_seed_disagreement=null   # not applied
maximum_label_conflict=0.4
maximum_ood_score=0.99           # UNAVAILABLE_FOR_PANEL_A_THRESHOLD_TUNING
```

| Signal | Runtime state |
|---|---|
| hybrid_uncertainty | real (entropy of H1 P) |
| top1 score / margin | real from EBM |
| feasibility | real |
| contraindications | real if caller sets them (builder uses empty set) |
| seed_disagreement | NA; not applied |
| label_conflict | default 0.0 → gate inactive |
| ood_score | default 0.0; freeze says unavailable |

Do not “fix” now; V3 should decide whether to populate or retire.

---

## 15. Learning plan

`plan_builder.py::ACTION_PLANS`: static Vietnamese templates. `stage` and risk magnitude do **not** change text. Workload is not summed across top-k. Conflicts cannot occur inside the builder (single top action). `runtime_authorized=False` on the plan dataclass.

---

## 16. Simulator

On main: `SimulationResult` + `validate_empirical_support`. `causal_claim_allowed=False`. No Hybrid forward, no feature perturbation, no C0 hook.

YAML still claims `simulator.enabled: true` and “all frozen hybrid seeds required”. **Implementation on main does not meet that protocol.** Full simulator code/artifacts remain on `17b519b` `explainable_v2` if needed later. No causal claims.

---

## 17. Panel A protocol

Development panel. 300 cases. Grouped stratified sampling, seed 2026, student-disjoint (`sampling.py`). Labels = Gemini + LFs → Snorkel OOF. EBM + router + (rejected) calibration selected here. Release gates passed (`PANEL_A_RELEASE_GATES.json`). `runtime_authorized=false` at freeze. Panel B untouched (`panel_b_touched=false` on all pre-heldout manifests).

---

## 18. Panel B protocol

Heldout. Frozen after development freeze (`EVALUATION_STARTED.json`, `PANEL_B_EVALUATION_PROTOCOL.json`). 150 cases, 557 Gemini, 0 failed calls, evidence coverage 1.0. Metrics computed once; hashes locked in `FINAL_RELEASE_MANIFEST.json`. `post_panel_b_tuning_permitted=false`.

**V3 rule:** historical evidence only. New confirmatory panel required after C0 rebase. Do not design that panel in this phase beyond the requirement.

---

## 19. Metrics

Implemented in `final/metrics.py::evaluate_grouped_ranking`:

| Metric | Def | k / τ | Role |
|---|---|---|---|
| NDCG@3 | sklearn `ndcg_score`, mean over queries with ≥1 positive | k=3 | **primary** |
| Precision@1 | top1 relevance ≥1 | τ=1 | secondary |
| MRR | first relevant ≥1 | τ=1 | secondary |
| Recall@3 | positives in top3 / all positives | k=3, τ=1 | secondary |
| Pairwise accuracy | concordant pairs, ties skipped | — | secondary |
| Invalid-action rate | top1 ineligible | — | gate = 0 |
| Exact best top1 | separate field in freeze JSONs | — | secondary |

Bootstrap: paired case, 2000 iter seed 2026 (Panel B); 5000 iter on some Panel A gates. CI reported in freeze files. Coverage/router rates are **not** inside `RankingMetrics`; they live in router freeze (`recommend_coverage=0.3545` on Panel A operating point).

---

## 20. Claim boundaries

Frozen language (`FINAL_SCIENTIFIC_AUDIT.md`, release manifest):

```text
predictive relevance ranking
plausibility
model-implied risk delta
NOT causal treatment effect
```

Simulator must not emit ATE/CATE. `FORBIDDEN_RANKER_FEATURES` includes `causal_ate`, `causal_cate`, `final_result`, label diagnostics.

---

## 21. Prediction compatibility

See `PREDICTION_AUTHORITY_COMPATIBILITY.md`.

One-line: **code reusable; learned risk-conditioned artifacts not reusable on C0 without rebuild + retrain + new heldout.**

---

## 22. Active inconsistencies

Recorded, not edited:

1. `FINAL_RELEASE_MANIFEST.runtime_authorized=true` vs `RecommendationDecision` raises if True (`final/contracts.py`). Historical freeze JSONs stay `false`. `FINAL_RELEASE_SUMMARY.md` explains the split; the dataclass still cannot represent the released True bit.
2. README / `final/README.md` say “frozen Hybrid CNN-BiLSTM” — scientifically H1 residual expert, not C0.
3. `llm_annotation_protocol.yaml` `current_model_output_visible=false` vs prompt `risk_band`/`uncertainty_band`.
4. `query_evidence` bands 0.3/0.6 vs selected 0.2/0.8.
5. `provider_envelope.py` imports missing `src.recommend_hybrid.explainable_v2.provenance`.
6. YAML simulator enabled vs stub on main.
7. Foundation `PredictionContext` is two-class H1-shaped; C0 `PredictionResult` is the live contract.
8. `StudentRepresentation` 64/32-D still validated if constructed; unused.

Authority if conflict: **frozen hashes + source code behavior** over README adjectives; **C0** over H1 for future work; **Panel B freeze** over any later convenience.

---

## 23. Gaps

See `artifacts/recommend_hybrid/v3_context/GAP_REGISTER.csv`.

MUST_FIX: G01 H1≠C0, G02 identity map, G03 uncertainty, G04 runtime_authorized, G05 missing rebuild inputs, G15 new heldout.

HIGH_VALUE: G06 prompt bands, G07 dead safety signals, G08 empty BORDERLINE ranks, G09 band mismatch, G10 envelope import.

Do not treat G13 (content-review rarity) or UCI expansion as MUST_FIX.

---

## 24. Feasibility map

See `V3_FEASIBILITY_MATRIX.csv`. No architecture chosen.

Design-worthy (YES): C0 rebase, reuse Gemini, selective later labels, refit Snorkel, refit five EBM, keep five EBM as baseline, real label-conflict, ranked HUMAN_REVIEW, personalized plans, new Panel C.

MAYBE: OOD detector, workload controller, temporal consistency, pairwise residual ranker.

---

## 25. What MUST NOT be forgotten in V3

```text
V2 risk != C0 risk
Panel B is dead for tuning
Gemini is conditionally portable, not cleanly blind
seed_disagreement was never real
label_conflict and ood were never real
100pct / FINAL_EVALUATION is non-intervention
do not copy H1 0.2/0.8 onto C0
do not overwrite frozen hashes
do not call Gemini in a survey or “quick check”
StudentRepresentation is not the V2 model
prediction_adapter is not proof that V2 already uses C0
Module_recomend 17b519b is the scientific attic, not a second production
UCI is out of V2 rec scope unless explicitly expanded
simulator language is model-implied risk delta only
```
