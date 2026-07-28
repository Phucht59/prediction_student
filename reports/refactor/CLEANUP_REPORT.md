# Unified Refactor Cleanup Report

## Outcome

The public UCI prediction authority is now the unified stage-aware system under
`artifacts/final/unified_stage_aware_uci/`. The former separate-stage model
evidence is no longer in the final authority tree.

## Cleanup performed

- Removed local pytest, Ruff, Python and untracked exploratory prediction caches
  by exact path.
- Did not execute broad `git clean`.
- Preserved canonical OULAD, recommendation, database, data and checkpoint
  evidence.
- Relocated 211 legacy evidence/source/config/report files with SHA-256
  preservation to `artifacts/history/legacy_uci_separate_stage_v1/` and
  `reports/history/LEGACY_UCI_SEPARATE_STAGE_MODEL_REPORT.md`.
- Removed the legacy `study early-warning` CLI; `study unified-stage` is the
  only public UCI stage training/evaluation workflow.
- Replaced 38 legacy separate-stage tests with unified stage-view, one-estimator,
  mask, split, checkpoint, protocol and authority tests.

## Protected content

No raw dataset, official frozen prediction, official checkpoint, OULAD
artifact, recommendation policy/data, canonical database, `.env`, DOCX or PDF
was deleted or modified.

## Replacement gate

Cleanup of the old final authority occurred only after:

- unified training completed with 500 dataset/model/fold/seed runs;
- 31,320 ensemble and 156,600 seed-level UCI prediction rows were written;
- 60 stage and 20 UCI overall metric rows validated;
- all 500 checkpoint checksums validated;
- the separate replacement PostgreSQL database passed its stage-aware checks.

Status: `PASS`.
