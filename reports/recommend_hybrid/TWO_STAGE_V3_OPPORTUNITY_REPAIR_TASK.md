# Two-Stage V3 opportunity-count serialization repair

## Root cause

The registered V3 action feature `opportunity_count` was already computed by the Hybrid-only silver builder as:

```python
opportunity = int(future_opportunities[family])
```

but was omitted from the serialized candidate-row dictionary. This is an engineering serialization defect, not a new feature proposal.

The correction must use only published assessment and VLE schedules. It must not read Outcome-Grounded V2.1 artifacts, future learner behaviour, outer-test statistics or protected attributes.

## Authority after repair

The following invariants must remain exact:

```text
candidate rows = 82,847
ranking groups = 29,043
positive groups = 9,304
all pre-existing candidate columns unchanged
silver_positive unchanged
current_behavior_signal unchanged
future_behavior_signal unchanged
V2.1 artifacts used = false
```

The only candidate-table change allowed is adding integer `opportunity_count`.

## Local execution

Synchronize:

```powershell
git checkout codex/constrained-counterfactual-recommender
git pull --ff-only origin codex/constrained-counterfactual-recommender
git log -1 --oneline
```

Preserve all existing untracked V2.1 batches.

Run focused tests:

```powershell
python -m pytest `
  --confcutdir=tests/recommend_hybrid/two_stage_v3 `
  tests/recommend_hybrid/two_stage_v3 `
  -q
```

Run the registered serialization repair:

```powershell
python scripts/recommend_hybrid/two_stage_v3/repair_opportunity_count.py
```

Required audit:

```text
artifacts/recommend_hybrid/two_stage_v3/OPPORTUNITY_COUNT_REPAIR.json
```

The audit must report:

```text
status = REPAIRED or ALREADY_REPAIRED
rows = 82847
groups = 29043
positive_groups = 9304
labels_changed = false
existing_columns_changed = false
v2_1_artifacts_used = false
future_behaviour_used = false
minimum_opportunity_count > 0
```

Then continue the original V3 runbook from checkpoint-authority validation:

```powershell
python scripts/recommend_hybrid/validate_checkpoint_authority.py
python scripts/recommend_hybrid/two_stage_v3/build_embedding_cache.py
python scripts/recommend_hybrid/two_stage_v3/train_and_evaluate.py
python scripts/recommend_hybrid/two_stage_v3/bootstrap.py
python scripts/recommend_hybrid/two_stage_v3/verify.py
python scripts/recommend_hybrid/two_stage_v3/release.py
python scripts/recommend_hybrid/two_stage_v3/render_report.py
```

Do not change labels, gates, registered head configurations or feature definitions after execution.

## Staging

Do not use `git add .`.

The repair legitimately modifies only these historical source files:

```text
artifacts/recommend_hybrid/hybrid_only_final/dataset/candidate_rows.parquet
artifacts/recommend_hybrid/hybrid_only_final/dataset/schema.json
artifacts/recommend_hybrid/hybrid_only_final/dataset/CHECKSUMS.json
```

Stage those files together with:

```text
artifacts/recommend_hybrid/two_stage_v3/**
reports/recommend_hybrid/TWO_STAGE_V3_FINAL_RESULTS_VI.md
```

The original candidate Parquet SHA and repaired SHA are retained in `OPPORTUNITY_COUNT_REPAIR.json`, so the failed Hybrid-only evidence remains auditable. The historical Hybrid-only metrics and release status must not be changed.

Keep PR #4 Draft. Merge allowed: NO.
