# Local task: complete Outcome-Grounded V2.1 with the full registered search

## Authority

- Repository: `C:\hufit\kltn`
- Branch: `codex/constrained-counterfactual-recommender`
- Claim boundary: `OFFLINE_PREDICTIVE_RELEVANCE_NOT_CAUSAL_EFFECT`
- PR #4 must remain Draft and unmerged.

The corrected OOF result at commit `4ddd20f` is scientifically encouraging but is not final because only the first frozen hyperparameter configuration of each model family was evaluated. The exact controls and initial ablations also reduced selected LambdaMART models from 100 trees to 10 trees. The scripts in this runbook correct those limitations without changing the frozen dataset, labels, action families, primary metric, or release gates.

## 1. Synchronize

```powershell
git checkout codex/constrained-counterfactual-recommender
git pull --ff-only origin codex/constrained-counterfactual-recommender
git status --short
git log -1 --oneline
```

The working tree must be clean before execution.

## 2. Focused safeguards

```powershell
python -m pytest tests/recommend_hybrid/outcome_grounded_v2_1/test_corrected_scientific_core.py -q
python -m pytest tests/recommend_hybrid/outcome_grounded_v2_1/test_full_registered_execution.py -q
```

## 3. Full preregistered model and hyperparameter search

```powershell
python scripts/recommend_hybrid/v2_1/run_full_registered_search.py
```

This command performs the following actions once:

1. archives the first-configuration corrected OOF and model-selection artifacts;
2. evaluates all 18 configurations frozen in the protocol:
   - 3 interaction-logistic configurations;
   - 3 pairwise-logistic configurations;
   - 8 LambdaMART configurations;
   - 4 boosted-tree configurations;
3. performs three learner-grouped inner folds inside each outer fold;
4. selects the model and hyperparameters independently inside each outer-training partition;
5. rebuilds the corrected three-fold OOF predictions and 1,000-run random null.

Do not delete the archive or manually choose a model after seeing outer-fold results.

## 4. Recalculate both bootstrap estimands

```powershell
python scripts/recommend_hybrid/v2_1/corrected_bootstrap.py
```

Required outputs:

- `final_oof/BOOTSTRAP_GROUP_WEIGHTED.json`
- `final_oof/BOOTSTRAP_LEARNER_WEIGHTED.json`

Both must contain 2,000 learner-cluster replicates.

## 5. Run exact retrained negative controls

Run controls separately so each command can be resumed safely:

```powershell
python scripts/recommend_hybrid/v2_1/run_exact_negative_controls.py --replicates 200 --batch-size 5 --control NC1_LABEL_SHUFFLE_RETRAIN
python scripts/recommend_hybrid/v2_1/run_exact_negative_controls.py --replicates 200 --batch-size 5 --control NC2A_TRAIN_STATE_SHUFFLE
python scripts/recommend_hybrid/v2_1/run_exact_negative_controls.py --replicates 200 --batch-size 5 --control NC2B_TEST_STATE_SHUFFLE
python scripts/recommend_hybrid/v2_1/run_exact_negative_controls.py --replicates 200 --batch-size 5 --control NC3_ACTION_IDENTITY_SHUFFLE_RETRAIN
python scripts/recommend_hybrid/v2_1/run_exact_negative_controls.py --replicates 200 --batch-size 5 --control NC4_WRONG_TRAJECTORY_REBUILD
python scripts/recommend_hybrid/v2_1/run_exact_negative_controls.py --replicates 200 --batch-size 5 --control NC5_TIME_REVERSAL_PLACEBO
```

Every replicate must use the exact selected outer-fold family and hyperparameters. Do not reduce `n_estimators`, tree depth, or feature set for null models. Thread count may be reduced only if it does not alter the statistical model.

Expected summary:

```text
artifacts/recommend_hybrid/outcome_grounded_v2_1/negative_controls_retrained/SUMMARY.csv
```

A control passes only when all 200 registered replicates exist and real OOF NDCG@3 exceeds the control distribution's 95th percentile.

## 6. Run all exact ablations

```powershell
python scripts/recommend_hybrid/v2_1/run_exact_ablation.py
```

All ten ablations must have real OOF predictions and metrics:

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

Do not interpret an ablation as successful merely because its name exists in a CSV.

## 7. Evaluate the fail-closed release gate

```powershell
python scripts/recommend_hybrid/v2_1/corrected_release.py
```

The release gate must refuse completion if any required control, ablation, bootstrap, checksum, model-family result, or safety artifact is missing.

Allowed validated status:

```text
OUTCOME_GROUNDED_V2_1_OFFLINE_VALIDATED
RECOMMENDATION_MODULE_COMPLETE
```

Allowed nonvalidated status after full execution:

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

Review generated artifacts before staging. Do not stage caches or temporary files.

Suggested commits:

```text
eval(recommendation): complete full registered v2.1 search
audit(recommendation): complete exact retrained controls
audit(recommendation): complete exact v2.1 ablations
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

Dataset authority:
Full-grid trials per outer fold:
Selected model and parameters per outer fold:
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
