# Database Index Justification

Post-cutover count: **24 non-primary indexes**, including indexes that enforce
UNIQUE contracts. No speculative index was added.

| Index group | Query/use | Selectivity |
|---|---|---|
| Dataset slug; dataset/version | Resolve one canonical dataset/version | 1 of 3 |
| Record version/source; student key | Resolve a final record or student history | 1 of 16,422 |
| Model dataset/key | Resolve one of ten models per dataset | 1 of 30 |
| Run model/status; dataset version; natural key | Locate final completed runs | 1 of 30 |
| Artifact run/path; dataset | Verify evidence for a run/dataset | 1 of 81 |
| Metric natural dimensions | Exact overall/class/Top-k lookup | 1 of 891 |
| Risk profile run/record and record/run helpers | Resolve one profile | 1 of 15,378 |
| Plan risk/revision | Resolve one plan revision | 1 of 15,378 |
| Action plan/week and natural key | Fetch a plan's weekly actions | about 2–4 of 27,355 |
| Review plan/type | Fetch sparse human review history | currently empty, contract-required |
| Migration version | Detect version collision | 1 of 10 |

## Measured plans

On the populated target:

```text
record lookup:
Index Scan using record_dataset_version_id_source_record_id_key
Execution Time: 0.054 ms

exact metric lookup:
Index Scan using metric_natural_key_idx
Execution Time: 0.028 ms

actions for one plan:
Index Only Scan using action_plan_id_action_code_week_no_priority_key
Execution Time: 0.109 ms
```

The disposable validation initially counted four redundant explicit indexes.
Immutable migration 009 removed them instead of editing an applied migration.
