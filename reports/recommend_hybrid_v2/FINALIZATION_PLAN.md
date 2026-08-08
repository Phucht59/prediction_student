# Recommendation V2 Finalization Plan

## Authority audit

- Repository: `C:\hufit\kltn`
- Branch: `Module_recomend`
- Audited HEAD: `a6d19ecffcab3b61adfe8912b890ce0a3fa6b85e`
- Expected checkpoint: `a6d19ecffcab3b61adfe8912b890ce0a3fa6b85e`
- Worktree at audit: tracked files clean; only the preserved untracked read-only preflight directory at `artifacts/recommend_hybrid/explainable_v2/audit/finalization_preflight_v1/`
- Panel B touched: false
- Runtime authorized: false

The frozen Hybrid CNN–BiLSTM risk model, its architecture, and its weights are outside the mutation scope. Panel B remains embargoed until the complete development freeze gate passes.

## Current blockers

1. One eligible Panel-A action row (`414696::GGG::2014B::EARLY_20`, `RECOVER_ENGAGEMENT`) has only one independent active source family. The current label artifact preserves its OOF probability but incorrectly marks it as an ordinary trainable silver label.
2. The five-EBM development runner trains on ordinal expected relevance in `[0,3]`, while the public contract specifies `[0,1]`; the runtime lineage and normalization location must be made explicit and tested.
3. Runtime feasibility requires fields that were unavailable to the frozen query-level annotation policy, creating train/runtime policy drift.
4. Missing `seed_disagreement` is silently represented as zero in candidate construction.
5. The public router still exposes `NO_ACTION` and `MONITOR` rather than the four final statuses.
6. The current Panel-A release report fails the minimum-two-source-family gate.

## Minimal remediation sequence

1. Preserve all 1,500 OOF label rows and probabilities for provenance. Add stable retention metadata with minimum independent family count 2; mark the single unsupported row `retained_for_training=false` and `label_status=INSUFFICIENT_SOURCE_SUPPORT`.
2. Make every supervised ranker consumer fail closed unless it honors `retained_for_training`. Exclude the unsupported row from model fitting and exclude its entire query from query-level model-selection/evaluation comparisons where a complete five-action relevance set is required.
3. Use the frozen query-level V4 eligibility evaluator as the canonical feasibility policy for both development and runtime.
4. Represent unavailable seed disagreement as nullable evidence and apply disagreement thresholds only to real finite values.
5. Preserve native ordinal EBM artifacts where scientifically valid and install exactly one tested public adapter, `clip(native_prediction / 3, 0, 1)`, unless artifact inspection proves the frozen models already use normalized targets.
6. Revalidate the locked EBM selection, rerun raw-versus-cross-fit-isotonic selection, rebuild Panel-A gates, harden the router, and freeze all development contracts and hashes before any Panel-B access.

## Retraining boundaries

- Never retrain or alter the frozen Hybrid CNN–BiLSTM risk model.
- Do not regenerate Gemini Panel-A reviews.
- Do not add a labeling function or source to repair the observed violation.
- Rerun the existing train-only Snorkel OOF aggregation only after retention code and tests pass; its purpose is to regenerate the auditable label artifact with retention metadata, not to tune labels.
- Do not rerun the 432-config / 6,480-fit EBM grid unless exact selected-config validity cannot be established from locked artifacts.
- Recreate the five final action EBMs only when required to enforce corrected supervised-target retention or the selected score contract.

## Locked-grid revalidation decision rule

The locked `search_state.jsonl` contains per-query NDCG contributions for each of all 432 configurations and records a frozen query-order hash. Exact no-refit re-selection is permitted only if that query order can be reconstructed and hash-verified against the corrected Panel-A inputs, and the stored contribution at the excluded query is sufficient to recompute every metric used by the frozen selection rule. If any selection statistic cannot be reconstructed exactly, an audit will identify the missing evidence and the minimum required recomputation will be run. A full grid rerun is authorized only if the locked rule genuinely requires it.

## Panel B embargo and stop conditions

Panel B will not be read, enumerated, reviewed, or evaluated before the development freeze manifest passes all required tests, leakage/privacy audits, and Panel-A release gates. The process stops fail-closed on Panel-B access, secrets, student overlap, post-cutoff leakage, risk-lineage drift, untraceable artifacts, failed release gates, unavailable real provider evidence, or inability to establish selected-config validity without fabrication.

All pre-freeze metrics are labeled `PANEL_A_DEVELOPMENT`. `RUNTIME_AUTHORIZED` remains false until the separate final release conditions pass.
