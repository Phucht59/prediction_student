# Outcome-Grounded V2.1 resumable execution quickstart

Use this after pulling the latest `codex/constrained-counterfactual-recommender` branch.

## Full registered search

The full registered search is already complete. Only rerun the following commands if its authority is invalidated or a trial artifact is missing.

```powershell
foreach ($fold in 0,1,2) {
  python scripts/recommend_hybrid/v2_1/run_resumable_full_registered_search.py --outer-fold $fold --trial-start 0 --trial-stop 6
  python scripts/recommend_hybrid/v2_1/run_resumable_full_registered_search.py --outer-fold $fold --trial-start 6 --trial-stop 12
  python scripts/recommend_hybrid/v2_1/run_resumable_full_registered_search.py --outer-fold $fold --trial-start 12 --trial-stop 18
}
python scripts/recommend_hybrid/v2_1/run_resumable_full_registered_search.py
```

If a trial records `ERROR`, archive and reopen only failed checkpoints, then rerun the same chunk:

```powershell
python scripts/recommend_hybrid/v2_1/retry_failed_full_grid_trials.py --outer-fold all
```

Do not proceed unless `FULL_REGISTERED_SEARCH.json` reports `COMPLETE` and all three `fold_<n>_trials.csv` files contain 18 rows with status `COMPLETE`.

## Final evidence

The two 2,000-replicate bootstrap estimands are already complete. Recalculate only if final OOF authority changes:

```powershell
python scripts/recommend_hybrid/v2_1/corrected_bootstrap.py
```

### Recommended control runner

Use the parallel authority-bound dispatcher. It preserves the exact selected 250-tree LambdaMART configurations while using one model thread per worker. Existing batches are resumed automatically.

The default `round_robin` schedule runs one replicate window for every control before moving to the next window. This prevents a time-limited run from producing NC1-only evidence.

For bounded sessions, run twelve batches at a time. With six controls and batch size five, each invocation advances approximately five replicates per control:

```powershell
python scripts/recommend_hybrid/v2_1/run_parallel_authority_controls.py `
  --replicates 200 `
  --batch-size 5 `
  --workers 2 `
  --schedule round_robin `
  --max-batches 12 `
  --checkpoint-every 1 `
  --control all
```

Repeat the same command until all controls reach 200/200. Existing batches are skipped, and `DISPATCH_PROGRESS.json`, `SUMMARY.csv`, `PROGRESS.json`, and the exact-control marker are refreshed after every completed batch.

For a long uninterrupted session, omit `--max-batches`:

```powershell
python scripts/recommend_hybrid/v2_1/run_parallel_authority_controls.py `
  --replicates 200 `
  --batch-size 5 `
  --workers 2 `
  --schedule round_robin `
  --control all
```

If memory remains stable, three workers may be used. Do not exceed four workers:

```powershell
python scripts/recommend_hybrid/v2_1/run_parallel_authority_controls.py `
  --replicates 200 `
  --batch-size 5 `
  --workers 3 `
  --schedule round_robin `
  --max-batches 18 `
  --control all
```

The frozen protocol still requires 200 completed replicates for each of the six controls. Parallelism, round-robin scheduling and bounded sessions change only execution order; they do not reduce tree count, feature set, folds or replicate count.

To finish one specific control after all controls have broad coverage, use control-major scheduling:

```powershell
python scripts/recommend_hybrid/v2_1/run_parallel_authority_controls.py `
  --replicates 200 `
  --batch-size 5 `
  --workers 2 `
  --schedule control_major `
  --control NC1_LABEL_SHUFFLE_RETRAIN
```

The original sequential authority-bound commands remain valid when process parallelism is unsuitable.

## Exact ablations and release

After all six controls reach 200/200:

```powershell
python scripts/recommend_hybrid/v2_1/run_authority_bound_ablation.py
python scripts/recommend_hybrid/v2_1/run_authority_bound_release.py
```

The authority-bound wrappers archive stale control and ablation outputs whenever the final selected-model SHA changes.

The release gate remains fail-closed. Do not integrate runtime or merge PR #4 unless the gate produces `OUTCOME_GROUNDED_V2_1_OFFLINE_VALIDATED`.
