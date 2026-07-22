# Repository Audit Before V5

Audit date: 2026-07-18 (Asia/Bangkok). This report is descriptive and does not modify V4 evidence.

## Baseline and worktree

| Check | Result | Evidence / decision |
|---|---|---|
| Frozen base exists | PASS | `ce79aa0b8f7444ac47ae9ae3ba6e72f997c5dd0a` is a commit contained by the V4 branch. |
| V5 branch base | PASS | `codex/project-v5-cnn-bilstm-final` was created at the exact frozen commit. |
| V4 branch rewrite | PASS | No reset, force-push, deletion or V4 artifact write was performed. |
| Initial worktree | PASS | Clean before the V5 branch was created. |
| Raw UCI data | PASS | 395 `student-mat` rows and 649 `student-por` rows are available locally; frozen hashes match. |
| Raw OULAD data | PASS | Required raw tables are available locally; frozen hashes match. |
| Processed OULAD F2 | PASS | Sequence, aggregate, target and split manifest exist and match the V5 protocol hashes. |
| Future benchmark | LOCKED | No future prediction file was read by V5 code or opened for model selection. |
| CUDA | PASS | RTX 2060 6 GB; PyTorch 2.7.1+cu118 reported CUDA available. |
| Docker | UNAVAILABLE | Docker executable was not present in `PATH`; Compose is provided but cannot yet be executed here. |

## Repository scale and duplication

At the start of V5 the repository contained 2,357 tracked files (144,437,693 bytes). The complete local workspace, including ignored raw data, environments, checkpoints and caches, occupied about 8.72 GB; `.git` occupied about 398 MB.

Exact-hash review found extensive historical mirroring. The largest tracked duplicate pair was the V4 `inner_fold_manifest.csv` mirrored under both `artifacts/` and `reports/` (8,558,512 bytes each). Historical `student_mat` run folders also contain hundreds of identical preprocessors and repeated metrics/configuration files. These files are evidence from prior phases, so V5 does not delete them before final evidence lock. V5 writes only to `artifacts/v5` and `reports/v5`, and runtime caches/checkpoints are ignored.

## Code and workflow review

| Area | Finding | V5 action |
|---|---|---|
| Routine CLI | Existing `project.py` validates final V4 release but has no V5 namespace. | Extend with nested `v5` and `db` commands while retaining every existing command. |
| UCI model | Frozen model is sequence-only and uses only G1/G2. | Add a two-branch, multi-task V5 model with an explicit context allowlist. |
| OULAD model | V4 already has strong grouped F2 infrastructure, cache and replay logic. | Reuse the frozen data contract read-only; implement V5 architecture and safe augmentation in a new namespace. |
| Leakage controls | Prior code has train-only preprocessing and grouped OULAD splits. | Preserve and add V5 tests for G3 exclusion, future exclusion, group disjointness and train-only transforms. |
| Long jobs | V4 has job/search caches. | V5 adds protocol/source fingerprints, Optuna SQLite resume and per-outer-fold checksum caches. |
| Paths | Repository runs on Windows; most code uses `pathlib`. | New V5 code uses `pathlib` and POSIX-form artifact identities. |
| Secrets | `.env` exists locally and is ignored. | V5 logs only key presence/redacted settings and requires a dedicated V5/test DSN for mutation. |

## Baseline tests

The initial suite collected 156 tests. First execution produced 148 passes, 6 explicit skips and 2 failures. Both failures were environmental: test subprocesses could not locate `git.exe` because Git was absent from `PATH`. No scientific or implementation assertion failed. V5 validation commands prepend the bundled Git path when run in this Codex environment; the final suite must be rerun with that environment normalized.

## Initial scientific conclusion

V4 remains `PRACTICAL_TIE`, temporal mechanism `FAIL`, and future `NOT_EXECUTED`. Those findings are treated as immutable historical evidence, not targets to overwrite. V5 changes are justified by the new context branch, multi-task objective, controlled augmentation, gated fusion and database lineage—not by a requirement that CNN–BiLSTM win.
