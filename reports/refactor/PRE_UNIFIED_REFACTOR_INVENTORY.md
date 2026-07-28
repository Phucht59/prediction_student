# Pre-Unified Refactor Inventory

## Source state

- Branch: `codex/unified-stage-aware-system`
- Source commit: `8de5e7bb957afadb213a5e485179435ec014ddda`
- Tracked files: 754
- Untracked files before cleanup: 730
- Untracked content consisted of local exploratory caches under `artifacts/experiments/`.

## Classification

| Classification | Content | Action |
|---|---|---|
| KEEP_CANONICAL | `artifacts/final`, final configs/reports, official checkpoints and predictions | Preserve and checksum |
| KEEP_UNTIL_REPLACEMENT_PASS | Separate-stage UCI evidence and fold cache | Preserve until unified replacement passes |
| LEGACY_ARCHIVE_AFTER_PASS | Superseded separate-stage UCI evidence | Archive with provenance and SHA-256 only after replacement validation |
| DELETE_LOCAL_CACHE | pytest/Ruff/Python caches and untracked exploratory caches | Remove by exact verified path |
| DO_NOT_TOUCH | virtual environments, `test_lab`, backups, raw data, OULAD, recommendation, database evidence, `.env`, DOCX/PDF | Never remove or modify during cleanup |

## Large tracked evidence

The largest tracked files are legitimate evidence rather than junk:

| Path | Bytes | Decision |
|---|---:|---|
| `artifacts/final/recommendation/recommendation_plans.jsonl` | 29,773,256 | KEEP_CANONICAL |
| `artifacts/final/database/persisted_recommendation_plans.jsonl` | 22,325,477 | KEEP_CANONICAL |
| `artifacts/final/predictions/cnn_bilstm_oulad/seed_predictions.parquet` | 16,179,305 | KEEP_CANONICAL |

## Cleanup safety

`git clean -ndX` was used only as a preview. Its output included virtual environments, raw and processed data, backups, `test_lab`, saved models, and ignored final evidence. Therefore broad `git clean` is prohibited. Cleanup is limited to exact cache directories whose resolved paths remain under the repository and whose contents are reproducible.

No canonical artifact, experiment evidence required for replacement comparison, model checkpoint, dataset, backup, database evidence, recommendation evidence, DOCX, or PDF is eligible for deletion.
