# Outcome-Grounded V2.1 resumable execution quickstart

Use this after pulling the latest `codex/constrained-counterfactual-recommender` branch.

## Full registered search

Run the 18 frozen trials in six-trial chunks. Every inner fold is checkpointed.

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

Do not proceed until `FULL_REGISTERED_SEARCH.json` reports `COMPLETE` and all three `fold_<n>_trials.csv` files contain 18 rows with status `COMPLETE`.

## Final evidence

Recalculate both 2,000-replicate learner-cluster bootstrap estimands:

```powershell
python scripts/recommend_hybrid/v2_1/corrected_bootstrap.py
```

### Recommended control runner

Use the parallel authority-bound dispatcher. It preserves the exact selected 250-tree LambdaMART configuration while using one model thread per worker. Existing batch files are resumed automatically.

Start conservatively with two workers:

```powershell
python scripts/recommend_hybrid/v2_1/run_parallel_authority_controls.py --replicates 200 --batch-size 5 --workers 2 --control all
```

If memory remains stable, three or four workers may be used. Do not exceed four workers:

```powershell
python scripts/recommend_hybrid/v2_1/run_parallel_authority_controls.py --replicates 200 --batch-size 5 --workers 3 --control all
```

The frozen protocol still requires 200 completed replicates for each of the six controls. Parallel execution changes only CPU scheduling; it does not reduce tree count, feature set, folds, or replicate count.

The original sequential authority-bound commands remain valid when parallel execution is unsuitable:

```powershell
python scripts/recommend_hybrid/v2_1/run_authority_bound_negative_controls.py --replicates 200 --batch-size 5 --control NC1_LABEL_SHUFFLE_RETRAIN
python scripts/recommend_hybrid/v2_1/run_authority_bound_negative_controls.py --replicates 200 --batch-size 5 --control NC2A_TRAIN_STATE_SHUFFLE
python scripts/recommend_hybrid/v2_1/run_authority_bound_negative_controls.py --replicates 200 --batch-size 5 --control NC2B_TEST_STATE_SHUFFLE
python scripts/recommend_hybrid/v2_1/run_authority_bound_negative_controls.py --replicates 200 --batch-size 5 --control NC3_ACTION_IDENTITY_SHUFFLE_RETRAIN
python scripts/recommend_hybrid/v2_1/run_authority_bound_negative_controls.py --replicates 200 --batch-size 5 --control NC4_WRONG_TRAJECTORY_REBUILD
python scripts/recommend_hybrid/v2_1/run_authority_bound_negative_controls.py --replicates 200 --batch-size 5 --control NC5_TIME_REVERSAL_PLACEBO
```

After controls:

```powershell
python scripts/recommend_hybrid/v2_1/run_authority_bound_ablation.py
python scripts/recommend_hybrid/v2_1/run_authority_bound_release.py
```

The authority-bound wrappers archive stale control and ablation outputs whenever the final selected model SHA changes.

The release gate remains fail-closed. Do not integrate runtime or merge PR #4 unless the gate produces `OUTCOME_GROUNDED_V2_1_OFFLINE_VALIDATED`.
