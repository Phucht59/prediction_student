# Project file audit

Audit date: 2026-08-02. Branch: `codex/recommendation-scientific-final`.
Starting HEAD: `b2b0972`. The final release is an ancestor. The initial worktree
contained only untracked non-release neural-ranker outputs and scripts; no tracked
authority file had a pending edit. Large local areas were `.venv-oulad-v2` (5.95 GiB),
`.git` (2.96 GiB), `artifacts` (2.94 GiB), `test_lab` (2.46 GiB), and `data` (599 MiB).

| Path | Group | Function | Used by | Action |
| --- | --- | --- | --- | --- |
| `configs/final/` | FINAL_AUTHORITY | Frozen prediction/recommendation authority | validators, runtime | KEEP |
| `src/recommend_hybrid/` | ACTIVE_SOURCE_CODE | Evidence-policy LearningPlan runtime | scripts, tests | KEEP |
| `scripts/recommend_hybrid/` | ACTIVE_SCRIPT | Build/evaluate/validate entry points | release workflow | KEEP |
| `tests/recommend_hybrid/` | ACTIVE_TEST | Recommendation regression suite | pytest | KEEP |
| `artifacts/canonical_v3/` | CANONICAL_ARTIFACT | Frozen prediction artefacts | prediction adapter | KEEP |
| `artifacts/final/recommendation/` | FINAL_RECOMMENDATION | Registry, plans, metrics, checksums | final validator | KEEP |
| `artifacts/recommend_hybrid/scientific_labeling/` | CANONICAL_ARTIFACT | Phase 1/2 scientific-label foundation | Phase validators | KEEP |
| `artifacts/recommend_hybrid/scientific_model/diagnostic_seen_v1/` | NON_RELEASE_DIAGNOSTIC | Ranker failure audit | documentation only | KEEP |
| `archive/non_release_research/neural_ranker_diagnostics/` | NON_RELEASE_DIAGNOSTIC | Archived ranker metadata/scripts | documentation only | ARCHIVE |
| Failed ranker checkpoints and test predictions | CACHE_OR_TEMPORARY | Failed experiment output | none | DELETE |
| `.venv*`, caches, `test_lab/`, `backups/` | CACHE_OR_TEMPORARY / HISTORICAL_ARCHIVE | Local environment or non-release history | none at runtime | IGNORE |
| `src/recommend_hybrid/weak_supervision/` | ACTIVE_SOURCE_CODE | Scientific diagnostic/label research, not final authority | Phase 1/2 validators | KEEP |

No `UNKNOWN_REQUIRES_REVIEW` file was deleted. No canonical checksum, checkpoint,
metric, database schema, action catalog, policy, or stage router was changed.

## Baseline issue found during final verification

`tests/audit/test_phase11_canonical_v3.py::test_preflight_replays_frozen_hashes`
fails before any cleanup change to canonical files: its frozen manifest's
`information_policy` hash differs from the currently committed policy file. This cleanup
does not rewrite either protected side of that comparison. The discrepancy requires a
separate canonical-authority recovery decision and prevents a cleanup PASS.
