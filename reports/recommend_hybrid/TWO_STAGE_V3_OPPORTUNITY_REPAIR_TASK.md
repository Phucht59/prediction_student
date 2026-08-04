# Two-Stage V3 opportunity-count serialization repair

## Root cause

The registered V3 action feature `opportunity_count` was already computed by the Hybrid-only silver builder:

```python
opportunity = int(future_opportunities[family])
```

but it was omitted from the serialized candidate-row dictionary. This is an engineering serialization defect, not a new feature, target or protocol change.

The correction uses only published assessment and VLE schedules known at cutoff. It must not use Outcome-Grounded V2.1 artifacts, future learner behaviour, outer-test statistics or protected attributes.

## Frozen invariants

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

The only allowed candidate-table change is adding integer `opportunity_count`.

## Execution authority

Follow the complete updated runbook:

```text
reports/recommend_hybrid/TWO_STAGE_V3_LOCAL_EXECUTION_TASK.md
```

The repair command is:

```powershell
python scripts/recommend_hybrid/two_stage_v3/repair_opportunity_count.py
```

Required audit:

```text
artifacts/recommend_hybrid/two_stage_v3/OPPORTUNITY_COUNT_REPAIR.json
```

Keep PR #4 Draft. Do not use `git add .`. Merge allowed: NO.
