# Phase 1 validation

## Gate result

`PHASE_1_BLOCKED` because OULAD source-of-truth and recommendation lineage conflict. This is an evidence blocker, not a test failure to be hidden.

## Commands and expected scope

| Check | Command | Result |
|---|---|---|
| Git identity | `git status --short; git branch --show-current; git rev-parse HEAD; git remote -v; git lfs status` | PASS before edits; clean `main` at `4f77182…`, correct origin, then Phase 1 branch created |
| Import smoke | `.venv-oulad-v2/Scripts/python.exe -c "import src.models..."` | PASS |
| Main release status | `.venv-oulad-v2/Scripts/python.exe project.py final status` | PASS/READY; training false; expert labels pending |
| Main release validation | `.venv-oulad-v2/Scripts/python.exe project.py final validate` | PRE-EXISTING FAIL: seven final OULAD/database-stage report checksums mismatch; none is a Phase 1 path |
| Standalone final authority | `python scripts/final/validate_final_release.py` | NOT RUN: command writes its own checksum output; audited read-only instead |
| Release/registry tests | `pytest -q tests/release/test_final_artifacts.py tests/release/test_final_release.py tests/unit/test_public_registry.py` | PASS: 22 passed |
| Recommendation DB reconciliation | `pytest -q tests/database/test_recommendation_reconciliation.py` | NOT EXECUTED: 6 skipped because database fixture was unavailable |
| Model/config/checkpoint references | `verify_final_checkpoints()` plus canonical H1 payload/hash audit | PASS: legacy manifest 65/65; canonical H1 15/15 |
| New JSON/CSV | Python JSON parse plus exact CSV header/status enum checks | PASS: required JSON fields and 21 inventory rows |
| Markdown/path hygiene | `git diff --check` and manual local path audit | PASS; no dedicated Markdown link checker exists |
| Ruff | N/A | no Python source file was created or changed |

## Invariance design for Phase 2

1. Hash checkpoint bytes before and after (`SHA-256`) and require exact match.
2. Clone state dict tensors and require identical names, shapes, dtypes and values after adapter construction and inference.
3. Run frozen model in eval/inference mode twice and require deterministic embedding/logits on the same device/dtype.
4. Compare reference versus adapter-path logits, probabilities and predicted class. Require exact equality if the same forward outputs are merely read; otherwise justify tolerance from `torch.finfo(dtype).eps`, device kernels, operation depth and repeated-run measurements.
5. Verify embedding `[batch,64]` for canonical OULAD H1; document separate `[batch,32]` tabular residual embedding and forbid adapter gradients into either frozen branch.

## Gate checklist

- Source of truth singular: **BLOCKED**.
- Baseline commit/config/checkpoint/checksum: **PARTIAL PASS**, with conflict recorded.
- Embedding point: **PASS** for OULAD H1; UCI MAT loader remains incomplete.
- Recommendation inventory and KEEP/REFACTOR/REPLACE/REMOVE/MISSING: **PASS**.
- Expert status: **PASS — PENDING_REAL_EXPERT_LABELS**.
- Artefact/database conflicts: **PASS — recorded, unresolved authority choice remains**.
- Architecture, contracts, expert protocol and five-phase plan: **PASS**.
- Prediction/recommendation production behavior changed: **NO**.

No full training, GPU job, Optuna or full evaluation was executed. The only working-tree paths after validation are the eight Phase 1 artefacts.
