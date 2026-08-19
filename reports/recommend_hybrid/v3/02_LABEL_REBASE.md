# Block B — label rebase

**STATUS: PASS** (portable reuse; no Gemini call)

Panel B was not read. Panel A frozen jsonl + silver parquet only.

- Feature queries: 66,685
- Exact `query_id` overlap with Panel A: **179 / 300** (expected: C0 OOF is inner outer-fold 0 only)
- Conditionally portable action rows: 895
- Gemini non-null on those rows: 662
- Unmatched rows labeled by behavioral + feasibility LFs only
- Snorkel cardinality 4, min 2 families, train-only fit

`label_conflict` stored for audit, not used at V3 runtime.
