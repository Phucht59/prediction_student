# Two-Stage V4 feasibility audit — local task

## Purpose

This task does not train a model and does not change V4 evidence. It measures whether any score/threshold calibration from the already completed V4 OOF predictions could theoretically reach the original end-to-end Precision@1 gate while preserving the registered coverage floor.

The audit is deliberately post-hoc and uses held-out labels. Therefore its oracle result is diagnostic only and can never authorize release.

## Authority

Repository:

```text
C:\hufit\kltn
```

Branch:

```text
codex/constrained-counterfactual-recommender
```

The V4 execution authority is commit:

```text
5550f590e5afa69af54dd2c6d05ea2c9c22aabb6
```

The cache registry hash in `NESTED_OOF_RESULTS.json` is:

```text
96711031fc92876c633898a95b780ab5ea72560640ec6c12f88b412cbb43faa5
```

`cfd71318...` is the historical V3 protocol hash, not the cache registry hash.

Preserve all untracked V2.1 batches. Do not run training, bootstrap, controls, or release again.

## Commands

```powershell
git checkout codex/constrained-counterfactual-recommender
git pull --ff-only origin codex/constrained-counterfactual-recommender
git status --short
git log -1 --oneline
```

Run focused tests:

```powershell
python -m pytest `
  --confcutdir=tests/recommend_hybrid/two_stage_v4 `
  tests/recommend_hybrid/two_stage_v4 `
  -q
```

Run the audit:

```powershell
python scripts/recommend_hybrid/two_stage_v4/feasibility_audit.py
```

Required inputs:

```text
artifacts/recommend_hybrid/two_stage_v4/final_oof/OOF_PREDICTIONS.parquet
artifacts/recommend_hybrid/two_stage_v4/TWO_STAGE_V4_RELEASE.json
configs/recommend_hybrid/two_stage_v4_protocol.yaml
```

Required outputs:

```text
artifacts/recommend_hybrid/two_stage_v4/V4_FEASIBILITY_AUDIT.json
reports/recommend_hybrid/TWO_STAGE_V4_FEASIBILITY_AUDIT.md
```

## Mandatory invariants

```text
models_trained = false
labels_changed = false
release_eligible = false
groups = 29043
positive_groups = 9304
```

The audit must report:

- exact global precision–coverage frontier for direct gate probability;
- exact global frontier for action-derived probability;
- exact global frontier for the selected joint probability;
- exploratory joint/direct × top-action-confidence frontiers;
- per-stage frontiers;
- an optimistic post-hoc oracle over the registered V4 blend, action probability, margin, and stage threshold family;
- a perfect-ranking gate oracle that isolates the recommendability ceiling;
- learner target instability across stages;
- whether the current score family can theoretically reach the original 0.80 / 0.50 gate.

## Interpretation

If:

```text
current_score_family_can_reach_original_gate = false
```

then do not create V4.1 threshold tuning. The current score family has been exhausted. Further work would require a new representation or a scientifically different module boundary.

If the post-hoc oracle exceeds 0.80 at coverage 0.50, report it only as evidence that inner selection/calibration may be insufficient. Do not modify V4 artifacts or release status without a new preregistered protocol.

## Commit

Stage only the two audit outputs:

```powershell
git add `
  artifacts/recommend_hybrid/two_stage_v4/V4_FEASIBILITY_AUDIT.json `
  reports/recommend_hybrid/TWO_STAGE_V4_FEASIBILITY_AUDIT.md

git diff --cached --name-only
git commit -m "research(recommendation): publish v4 feasibility audit"
git push origin codex/constrained-counterfactual-recommender
```

Keep PR #4 Draft. Do not merge.

## Response template

```text
TWO-STAGE V4 FEASIBILITY AUDIT
==============================

Branch:
Remote head:
Working tree:
V2.1 batches preserved:

Models trained:
Labels changed:
Release eligible:

Current end-to-end P@1:
Current coverage:
Current Stage A precision:
Current conditional P@1:
Required Stage A precision for 80%:

Direct score best P@1 at coverage >= 0.50:
Action-derived score best P@1 at coverage >= 0.50:
Joint score best P@1 at coverage >= 0.50:
Joint × action-confidence best P@1 at coverage >= 0.50:

Registered-grid post-hoc oracle P@1 / coverage:
Perfect-ranking gate oracle P@1 / coverage:
Current score family can reach 0.80 / 0.50:

EARLY_20 best P@1 at stage coverage >= 0.50:
EARLY_35 best P@1 at stage coverage >= 0.50:
MIDDLE_50 best P@1 at stage coverage >= 0.50:

Learners with mixed stage targets:
Mixed target rate:

Scientific conclusion:
Runtime authorized: false
Merge allowed: NO
```
