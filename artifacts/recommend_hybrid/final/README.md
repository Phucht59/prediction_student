# Final recommendation evidence

This directory is the canonical evidence surface for the released recommendation model.
It contains only the evidence needed to reproduce and audit the final Recommendation
release; development experiments remain on the immutable `Module_recomend` lineage
branch instead of cluttering `main`.

Contents:

- `release/` — final release manifest/checksum inventory;
- `heldout/` — one-shot Panel-B final metrics, frozen real reviews, score table,
  preregistration protocol, provider provenance and checksums;
- `panel_a_reviews/` — frozen Panel-A real external review evidence;
- `weak_labels/` — corrected Panel-A weak-label outputs and manifest;
- `ranker/` — frozen five-EBM ranker artifacts;
- `router/` — frozen four-status router policy;
- `development_freeze/` — freeze created before Panel-B access;
- `release_gates/` — Panel-A scientific release-gate evidence;
- `MIGRATION_MANIFEST.json` — maps this clean namespace to the immutable release
  commit that produced it.

Panel B must never be rerun for tuning. Scientific source lineage remains at
`Module_recomend@17b519b22e8b69c875d27547d097e6d3b76bc404`.
