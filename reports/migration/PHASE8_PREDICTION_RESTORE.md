# Phase8 Prediction Restore

## Status

- Restore: **PASS**
- Working directory: `C:\hufit\student`
- Source authority: `codex/backup-hybrid-phase8-2026-08-17` in `C:\hufit\kltn`
- `kltn` modified: **NO**
- Outer rerun / retraining / HPO: **NO / NO / NO**

## Active model contract

`src.prediction` exposes exactly one public architecture: `Hybrid`. UCI Combined and OULAD use that same class with separately fitted instances (`uci`, `oulad_early`, `oulad_final`) and no joint training. The output is one binary logit. UCI uses `G3 < 10`; OULAD uses `Fail` or `Withdrawn`; D3 and F3 are frozen from Phase8.

Principal presentation views are UCI `S2` and OULAD `FINAL-100`. Supporting views are UCI `S0/S1` and OULAD `20/35/50/75`.

## Evidence and equivalence

Frozen outer evidence was copied byte-for-byte under `artifacts/prediction/final/` and final tables under `reports/prediction/final/`. The outer freeze, consumption, recovery, and integrity manifests are preserved. Deterministic source-to-destination model and data fixtures are recorded in `artifacts/migration/MODEL_EQUIVALENCE.json` and `DATA_EQUIVALENCE.json`; no outer labels were used.

Phase8 trained checkpoint files are not present in the authority branch. No checkpoint was fabricated. The active loader accepts only `Hybrid` checkpoints and fails closed for absent/wrong checkpoints; a temporary round-trip fixture verifies serialization and type identity.

## Recommendation impact

Recommendation code remains reusable and its ranking/actions/EBM/weak-label/safety logic was not redesigned. Prediction-derived feature and learned artifacts are classified as **PARTIAL / REQUIRES_REVALIDATION_AFTER_PREDICTION_RESTORE** because their previous `risk_probability` provenance may come from the copied wrong prediction subsystem. The dependency audit is at `artifacts/audit/RECOMMENDATION_PREDICTION_DEPENDENCY_AUDIT.json`.

## Artifacts

- `artifacts/migration/PREDICTION_PHASE8_MIGRATION_MANIFEST.csv`
- `artifacts/migration/ONE_HYBRID_ARCHITECTURE_AUDIT.json`
- `artifacts/migration/BINARY_TARGET_AUDIT.json`
- `artifacts/migration/DATA_EQUIVALENCE.json`
- `artifacts/migration/MODEL_EQUIVALENCE.json`
- `artifacts/migration/MIGRATION_TEST_SUMMARY.json`
- `artifacts/audit/RECOMMENDATION_PREDICTION_DEPENDENCY_AUDIT.json`

Historical copied prediction code and evidence remain under `test_lab/prediction_legacy/` and are not imported by active prediction or recommendation runtime.
