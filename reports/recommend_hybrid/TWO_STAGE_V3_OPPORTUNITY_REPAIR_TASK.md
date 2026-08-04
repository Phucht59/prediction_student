# Two-Stage V3 opportunity-count serialization repair

`opportunity_count` was computed by the Hybrid-only silver builder but omitted from the serialized candidate-row dictionary. This is a serialization correction, not a feature, label or protocol change.

The repair uses only published assessment and VLE schedules known at cutoff. It must preserve exactly:

```text
candidate rows = 82,847
ranking groups = 29,043
positive groups = 9,304
all pre-existing candidate columns unchanged
silver_positive unchanged
V2.1 artifacts used = false
future learner behaviour used = false
```

Execution authority:

```text
reports/recommend_hybrid/TWO_STAGE_V3_LOCAL_EXECUTION_TASK.md
```

Repair command:

```powershell
python scripts/recommend_hybrid/two_stage_v3/repair_opportunity_count.py
```

Required audit:

```text
artifacts/recommend_hybrid/two_stage_v3/OPPORTUNITY_COUNT_REPAIR.json
```

Keep PR #4 Draft. Do not use `git add .`. Merge allowed: NO.
