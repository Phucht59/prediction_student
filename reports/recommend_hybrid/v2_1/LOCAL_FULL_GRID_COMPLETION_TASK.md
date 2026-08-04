# Local task: finish Outcome-Grounded V2.1 with durable trial checkpoints

## Authority

- Repository: `C:\hufit\kltn`
- Branch: `codex/constrained-counterfactual-recommender`
- Claim boundary: `OFFLINE_PREDICTIVE_RELEVANCE_NOT_CAUSAL_EFFECT`
- PR #4 must remain Draft and unmerged.

The first full Cartesian search was interrupted because the old wrapper checkpointed only after an entire outer fold. Use the new trial- and inner-fold-resumable runner. It writes durable evidence for every `(outer fold, trial, inner fold)` and promotes results only after all 18 registered configurations complete in all three outer folds.

The two-replicate control outputs already present were generated against the pre-full-grid selected models. They are stale by definition and must not be resumed after the selected model authority changes. The authority-bound wrappers automatically archive them before starting new controls and ablations.

## 1. Synchronize

```powershell
git checkout codex/constrained-counterfactual-recommender
git pull --ff-only origin codex/constrained-counterfactual-recommender
git status --short
git log -1 --oneline
```

The working tree must be clean.

## 2. Focused safeguards

```powershell
python -m pytest tests/recommend_hybrid/outcome_grounded_v2_1/test_corrected_scientific_core.py -q
python -m pytest tests/recommend_hybrid/outcome_grounded_v2_1/test_full_registered_execution.py -q
python -m pytest tests/recommend_hybrid/outcome_grounded_v2_1/test_resumable_authority.py -q
```

## 3. Complete the full registered grid with trial checkpoints

The registered grid contains 18 configurations per outer fold. Run each fold in trial chunks. Completed trial and inner-fold JSON files are reused automatically.

```powershell
python scripts/recommend_hybrid/v2_1/run_resumable_full_registered_search.py --outer-fold 0 --trial-start 0 --trial-stop 6
python scripts/recommend_hybrid/v2_1/run_resumable_full_registered_search.py --outer-fold 0 --trial-start 6 --trial-stop 12
python scripts/recommend_hybrid/v2_1/run_resumable_full_registered_search.py --outer-fold 0 --trial-start 12 --trial-stop 18

python scripts/recommend_hybrid/v2_1/run_resumable_full_registered_search.py --outer-fold 1 --trial-start 0 --trial-stop 6
python scripts/recommend_hybrid/v2_1/run_resumable_full_registered_search.py --outer-fold 1 --trial-start 6 --trial-stop 12
python scripts/recommend_hybrid/v2_1/run_resumable_full_registered_search.py --outer-fold 1 --trial-start 12 --trial-stop 18

python scripts/recommend_hybrid/v2_1/run_resumable_full_registered_search.py --outer-fold 2 --trial-start 0 --trial-stop 6
python scripts/recommend_hybrid/v2_1/run_resumable_full_registered_search.py --outer-fold 2 --trial-start 6 --trial-stop 12
python scripts/recommend_hybrid/v2_1/run_resumable_full_registered_search.py --outer-fold 2 --trial-start 12 --trial-stop 18
```

Then run one final promotion call:

```powershell
python scripts/recommend_hybrid/v2_1/run_resumable_full_registered_search.py
```

Required marker:

```text
artifacts/recommend_hybrid/outcome_grounded_v2_1/FULL_REGISTERED_SEARCH.json
```

It must contain:

```text
status: COMPLETE
execution: TRIAL_AND_INNER_FOLD_RESUMABLE
expected_trials_per_outer_fold: 18
```

Required trial checkpoints:

```text
artifacts/recommend_hybrid/outcome_grounded_v2_1/full_grid_resumable/model_selection/fold_<n>/trials/trial_<nnn>/inner_<n>.json
artifacts/recommend_hybrid/outcome_grounded_v2_1/full_grid_resumable/model_selection/fold_<n>/trials/trial_<nnn>/trial.json
```

Do not delete these checkpoints. Do not manually select a model from outer-test results.

## 4. Recalculate both bootstrap estimands

```powershell
python scripts/recommend_hybrid/v2_1/corrected_bootstrap.py
```

Both outputs must contain 2,000 learner-cluster replicates:

- `final_oof/BOOTSTRAP_GROUP_WEIGHTED.json`
- `final_oof/BOOTSTRAP_LEARNER_WEIGHTED.json`

## 5. Run authority-bound exact negative controls

Use the authority-bound wrapper, not the old exact wrapper. It archives all stale batches whenever the selected model authority changes.

```powershell
python scripts/recommend_hybrid/v2_1/run_authority_bound_negative_controls.py --replicates 200 --batch-size 5 --control NC1_LABEL_SHUFFLE_RETRAIN
python scripts/recommend_hybrid/v2_1/run_authority_bound_negative_controls.py --replicates 200 --batch-size 5 --control NC2A_TRAIN_STATE_SHUFFLE
python scripts/recommend_hybrid/v2_1/run_authority_bound_negative_controls.py --replicates 200 --batch-size 5 --control NC2B_TEST_STATE_SHUFFLE
python scripts/recommend_hybrid/v2_1/run_authority_bound_negative_controls.py --replicates 200 --batch-size 5 --control NC3_ACTION_IDENTITY_SHUFFLE_RETRAIN
python scripts/recommend_hybrid/v2_1/run_authority_bound_negative_controls.py --replicates 200 --batch-size 5 --control NC4_WRONG_TRAJECTORY_REBUILD
python scripts/recommend_hybrid/v2_1/run_authority_bound_negative_controls.py --replicates 200 --batch-size 5 --control NC5_TIME_REVERSAL_PLACEBO
```

Every replicate must use the exact selected family and hyperparameters. No tree-budget reduction is allowed.

A control passes only when all 200 registered replicates exist and final real OOF NDCG@3 exceeds the null distribution's 95th percentile.

The current two-replicate trial values must not be treated as final. They nevertheless show a serious diagnostic risk: several shuffled controls scored above the pre-full-grid model. Preserve this evidence and allow the 200-replicate authority-bound run to decide the gate.

## 6. Run authority-bound exact ablations

```powershell
python scripts/recommend_hybrid/v2_1/run_authority_bound_ablation.py
```

All ten ablations must have actual three-fold OOF predictions and metrics:

- `FULL`
- `NO_RISK_PROFILE`
- `NO_BEHAVIOR_STATE`
- `NO_OPPORTUNITY`
- `NO_DEFICIT`
- `NO_COUNTERFACTUAL_DELTA`
- `NO_ACTION_INTERACTIONS`
- `NO_WORKLOAD`
- `ACTION_PRIOR_ONLY`
- `NO_CONSTRAINTS_OFFLINE_ONLY`

## 7. Evaluate the authority-bound fail-closed release gate

```powershell
python scripts/recommend_hybrid/v2_1/run_authority_bound_release.py
```

This command refuses release if controls or ablations were generated for an older selected model.

Allowed validated status:

```text
OUTCOME_GROUNDED_V2_1_OFFLINE_VALIDATED
RECOMMENDATION_MODULE_COMPLETE
```

Allowed nonvalidated status only after every registered execution completes:

```text
OUTCOME_GROUNDED_V2_1_EVIDENCE_INCONCLUSIVE
RECOMMENDATION_MODULE_NOT_COMPLETE
```

Runtime integration remains forbidden unless the validated status is produced by the gate.

## 8. Full validation

```powershell
python -m pytest tests/recommend_hybrid/outcome_grounded_v2_1 -q
python -m pytest tests/recommend_hybrid -q
python scripts/recommend_hybrid/validate_counterfactual.py
python -m compileall src/recommend_hybrid scripts/recommend_hybrid
ruff check src/recommend_hybrid scripts/recommend_hybrid tests/recommend_hybrid
git diff --check
```

## 9. Commit and push

Do not stage caches, temporary files, incomplete control batches, or interrupted model files unless they are registered resume artifacts intentionally tracked by the repository policy.

Suggested commits:

```text
eval(recommendation): complete resumable registered v2.1 search
audit(recommendation): complete authority-bound controls
audit(recommendation): complete authority-bound ablations
release(recommendation): publish final offline scientific gate
```

Push to the same branch. Keep PR #4 Draft. Do not merge.

## Final response format

```text
OUTCOME-GROUNDED V2.1 SCIENTIFIC COMPLETION
============================================

Branch:
PR:
Latest commit:
Working tree:

Full search status:
Execution mode:
Trials per outer fold:
Selected model and parameters per fold:
Model authority SHA256:
Learners:
Ranking groups:
Candidate rows:

Final OOF NDCG@3:
Precision@1:
MAP@3:
MRR:
Random mean:
Random p95:
Best non-learned baseline:
Improvement:
Group-weighted learner-cluster CI:
Learner-weighted estimate:

Negative controls completed:
Negative controls passed:
Ablations completed:
Full vs no-interactions:
Full vs no-counterfactual:
Full vs action-prior:

Data gate:
Ranking gate:
Personalization gate:
Stability gate:
Negative-control gate:
Ablation gate:
Safety gate:
Reproducibility gate:

Scientific status:
Thesis-scope completion:
Runtime authority:
Claim boundary:
Causal validation:
Expert validation:
Merge allowed: NO
```
