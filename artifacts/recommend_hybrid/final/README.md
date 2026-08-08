# Final recommendation evidence

This directory is the canonical, version-neutral evidence index for the released
recommendation system.

The files and subtrees here are byte-preserving references/copies of evidence
that was frozen before or immediately after the one-shot Panel-B held-out
benchmark. The original versioned paths under
`artifacts/recommend_hybrid/explainable_v2/` remain unchanged to preserve the
historical audit lineage.

Contents:

- `release/` — final release manifest and checksum inventory;
- `heldout/` — Panel-B final held-out metrics, frozen reviews, scores, manifests,
  and provider provenance;
- `panel_a_reviews/` — frozen Panel-A real external review evidence;
- `weak_labels/` — corrected Panel-A weak-label artifacts and manifest;
- `ranker/` — frozen five-EBM ranker artifacts;
- `router/` — frozen four-status safety-router policy;
- `development_freeze/` — development freeze created before Panel B access;
- `release_gates/` — Panel-A scientific release-gate evidence;
- `MIGRATION_MANIFEST.json` — records the production-namespace migration.

Scientific authority remains the hashes recorded in the frozen final release
manifest. No Panel-B tuning or provider rerun is permitted.
