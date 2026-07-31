# Feature schema audit

H0 and H1 both declare 47 temporal channel names, but the score-channel
semantics are different.

- H0 materialization considers a score available only after both submission
  and assessment due date are before the cutoff (`max(date_submitted, date)`).
  It populates cumulative score/count features and includes their summaries in
  a compact 49-feature aggregate branch.
- H1 deliberately excludes score values because raw OULAD lacks an explicit
  score-release timestamp. Its score missing mask remains unavailable. It
  constructs 161 summaries plus four stage-context fields (165 total).

The H0 rule is cutoff-safe under its declared conservative proxy and no
post-cutoff event was found. However, the exact feedback release time is not in
OULAD, so this is an **endpoint feature-authority difference**, not proof that
the H0 and H1 score features are equivalent. H0's score-progress signal must be
explicitly re-authorized before any recovery model uses it.

The complete row-level classification is in
`artifacts/audit/phase8/feature_schema_diff.csv`.
