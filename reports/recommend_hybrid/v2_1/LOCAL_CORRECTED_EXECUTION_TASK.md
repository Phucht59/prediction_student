# Local task — complete corrected Outcome-Grounded V2.1

## Authority

- Repository: `C:\hufit\kltn`
- Branch: `codex/constrained-counterfactual-recommender`
- Pull request: #4, keep Draft and unmerged
- Claim boundary: `OFFLINE_PREDICTIVE_RELEVANCE_NOT_CAUSAL_EFFECT`
- Historical V1, V2 and preliminary V2.1 artifacts are immutable.
- Corrected outputs must be written under `artifacts/recommend_hybrid/outcome_grounded_v2_1/final_oof`, `negative_controls_retrained`, and `ablations_executed`.

## 1. Synchronize

```powershell
git checkout codex/constrained-counterfactual-recommender
git pull --ff-only origin codex/constrained-counterfactual-recommender
git status --short
git log -1 --oneline
```

The working tree must be clean before execution.

## 2. Dependency check

```powershell
python -c "import numpy,pandas,scipy,sklearn,pyarrow,yaml,joblib; print('core dependencies PASS')"
python -c "import lightgbm; print('LightGBM', lightgbm.__version__)"
python -c "import xgboost; print('XGBoost', xgboost.__version__)"
```

At least one LambdaMART implementation must be available. The corrected evaluator also uses XGBoost for the pointwise boosted-tree candidate, so install it if unavailable:

```powershell
python -m pip install "xgboost>=2.0" "lightgbm>=4.0"
```

Record exact installed versions in the final report. Do not silently skip a registered model family.

## 3. Focused tests before model execution

```powershell
python -m pytest tests/recommend_hybrid/outcome_grounded_v2_1/test_corrected_scientific_core.py -q
```

All focused tests must pass. Fix code defects only; do not change the dataset, label formula, primary metric, gates, or candidate action families based on results.

## 4. Corrected nested grouped model selection and OOF evaluation

```powershell
python scripts/recommend_hybrid/v2_1/corrected_nested_evaluation.py
```

Mandatory conditions:

- labels fitted separately inside every inner-training partition;
- preprocessing fitted on train only;
- `interaction_logistic`, `pairwise_logistic`, `lambdamart`, and `boosted_tree` all actually evaluated;
- model selection uses three learner-grouped inner folds;
- outer folds 0, 1, 2 each generate a held-out prediction file;
- 1,000 random baseline repetitions complete;
- preliminary top-level OOF artifacts are not overwritten.

Inspect:

```text
artifacts/recommend_hybrid/outcome_grounded_v2_1/model_selection/
artifacts/recommend_hybrid/outcome_grounded_v2_1/final_oof/NESTED_OOF_RESULTS.json
artifacts/recommend_hybrid/outcome_grounded_v2_1/final_oof/BASELINE_COMPARISON.csv
```

If a registered model family errors, install/fix its dependency and rerun. Do not accept a search ledger that only lists the family name.

## 5. Corrected learner-cluster bootstrap

```powershell
python scripts/recommend_hybrid/v2_1/corrected_bootstrap.py
```

Both estimands must be produced:

```text
final_oof/BOOTSTRAP_GROUP_WEIGHTED.json
final_oof/BOOTSTRAP_LEARNER_WEIGHTED.json
```

The group-weighted point estimate must equal the global group-weighted NDCG difference. Do not group all stages of one learner into one ranking list.

## 6. Retrained negative controls

Run a short execution check first:

```powershell
python scripts/recommend_hybrid/v2_1/corrected_negative_controls.py --replicates 2 --batch-size 1 --control all
```

Delete only the two-replicate corrected-control batch files after confirming the implementation works, then run the registered execution:

```powershell
python scripts/recommend_hybrid/v2_1/corrected_negative_controls.py --replicates 200 --batch-size 10 --control all
```

The script is resume-safe. Continue until each control has exactly 200 completed replicates:

- `NC1_LABEL_SHUFFLE_RETRAIN`
- `NC2A_TRAIN_STATE_SHUFFLE`
- `NC2B_TEST_STATE_SHUFFLE`
- `NC3_ACTION_IDENTITY_SHUFFLE_RETRAIN`
- `NC4_WRONG_TRAJECTORY_REBUILD`
- `NC5_TIME_REVERSAL_PLACEBO`

A control passes only when the real corrected OOF NDCG@3 exceeds the control distribution p95. The historical `NEGATIVE_CONTROLS.csv` is only a null-score sanity check and cannot satisfy this gate.

## 7. Executed ablation study

```powershell
python scripts/recommend_hybrid/v2_1/corrected_ablation.py
```

All ten ablations must generate actual fitted predictions and metrics:

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

Do not leave `REGISTERED_REQUIRES_NESTED_RERUN` rows in the corrected summary.

## 8. Scientific release gate

```powershell
python scripts/recommend_hybrid/v2_1/corrected_release.py
```

The release gate must fail closed if any mandatory artifact is missing or partial.

Possible statuses:

```text
OUTCOME_GROUNDED_V2_1_OFFLINE_VALIDATED
OUTCOME_GROUNDED_V2_1_EVIDENCE_INCONCLUSIVE
```

Only the first status permits later runtime integration. Neither status permits causal claims.

## 9. Full validation

```powershell
python -m pytest tests/recommend_hybrid/outcome_grounded_v2_1 -q
python -m pytest tests/recommend_hybrid -q
python scripts/recommend_hybrid/validate_counterfactual.py
python -m compileall src/recommend_hybrid scripts/recommend_hybrid
ruff check src/recommend_hybrid scripts/recommend_hybrid tests/recommend_hybrid
git diff --check
```

Do not modify tests merely to hide a failure.

## 10. Reports

Update the detailed V2.1 reports using corrected artifacts only. Preserve the preliminary numbers as historical preliminary evidence and label them clearly.

The Vietnamese final report must include:

- selected model and hyperparameters for each outer fold;
- all model-family inner-CV results;
- corrected OOF NDCG@1, NDCG@3, Precision@1, MAP@3 and MRR;
- random mean and p95;
- strongest non-learned baseline;
- group-weighted and learner-weighted bootstrap estimates;
- every retrained negative control;
- every executed ablation;
- fold/stage stability;
- temporal result `COMPLETE_INSUFFICIENT_SUPPORT` or a valid supported result;
- safety and protected-feature exclusion;
- scientific gate status and claim boundary.

## 11. Commit and push

Stage only corrected code, outputs, tests, and reports:

```powershell
git status --short
git add scripts/recommend_hybrid/v2_1 tests/recommend_hybrid/outcome_grounded_v2_1 artifacts/recommend_hybrid/outcome_grounded_v2_1 reports/recommend_hybrid/v2_1
git commit -m "eval(recommendation): complete corrected v2.1 scientific evidence"
git push origin codex/constrained-counterfactual-recommender
```

Keep PR #4 Draft. Do not merge.

## Required final response

```text
OUTCOME-GROUNDED V2.1 CORRECTED EXECUTION
=========================================
Branch:
Remote head:
Working tree:
Tests:

Models actually evaluated:
Selected model fold 0:
Selected model fold 1:
Selected model fold 2:

Learners:
Ranking groups:
Candidate rows:
Corrected OOF NDCG@3:
Corrected Precision@1:
Corrected MAP@3:
Random mean:
Random p95:
Best non-learned baseline:
Difference:
Group-weighted 95% CI:
Learner-weighted estimate:

Retrained controls:
Ablations:
Fold stability:
Stage stability:
Temporal status:
Unavailable action selections:
Protected feature use:

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
Causal validation: NOT_PERFORMED
Expert validation: NOT_PERFORMED_NOT_REQUIRED_FOR_OFFLINE_THESIS_SCOPE
Merge allowed: NO
```
