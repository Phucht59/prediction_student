# Integrated Hybrid Two-Stage V3 — Local Execution Task

## Authority

Repository:

```text
C:\hufit\kltn
```

Branch:

```text
codex/constrained-counterfactual-recommender
```

Final architecture:

```text
Frozen residual CNN–BiLSTM prediction backbone
+ integrated recommendability neural head
+ integrated conditional-action neural head
```

This is one hybrid deep-learning recommendation system. Do not train or invoke XGBoost, LightGBM, LambdaMART, Logistic Regression, Random Forest, SVM, HistGradientBoosting or another external recommendation ranker.

## Scientific boundary

```text
OFFLINE_PREDICTIVE_RELEVANCE_NOT_CAUSAL_EFFECT
```

The registered primary gate remains:

```text
Held-out end-to-end Precision@1 >= 0.80
Positive-group coverage >= 0.50
```

Do not change labels, thresholds, registered configs or gates after seeing results.

## Preserve historical untracked files

Existing untracked V2.1 batches must remain untouched. Do not add, delete, move or modify them.

Record the initial status:

```powershell
git status --short
```

## Step 1 — Synchronize

```powershell
git checkout codex/constrained-counterfactual-recommender
git pull --ff-only origin codex/constrained-counterfactual-recommender
git log -1 --oneline
git status --short
```

Confirm that the pulled head contains:

```text
configs/recommend_hybrid/two_stage_v3_protocol.yaml
src/recommend_hybrid/two_stage_v3/model.py
scripts/recommend_hybrid/two_stage_v3/build_embedding_cache.py
scripts/recommend_hybrid/two_stage_v3/train_and_evaluate.py
```

## Step 2 — Focused tests

```powershell
python -m pytest `
  --confcutdir=tests/recommend_hybrid/two_stage_v3 `
  tests/recommend_hybrid/two_stage_v3 `
  -q
```

Expected focused scope before execution:

```text
16 passed
```

Do not continue if tests fail.

## Step 3 — Validate prediction checkpoint authority

```powershell
python scripts/recommend_hybrid/validate_checkpoint_authority.py
```

The exact residual CNN–BiLSTM checkpoint set, preprocessors, architecture hash and parameter count must pass. The frozen prediction backbone parameter count is:

```text
160,492
```

## Step 4 — Build leakage-safe frozen embedding caches

Run:

```powershell
python scripts/recommend_hybrid/two_stage_v3/build_embedding_cache.py
```

This step performs inference only. It must not update prediction-backbone weights.

Required outputs:

```text
artifacts/recommend_hybrid/two_stage_v3/cache/ACTION_CANDIDATES.parquet
artifacts/recommend_hybrid/two_stage_v3/cache/cross_fitted/GROUP_FEATURES.parquet
artifacts/recommend_hybrid/two_stage_v3/cache/outer_0/GROUP_FEATURES.parquet
artifacts/recommend_hybrid/two_stage_v3/cache/outer_1/GROUP_FEATURES.parquet
artifacts/recommend_hybrid/two_stage_v3/cache/outer_2/GROUP_FEATURES.parquet
artifacts/recommend_hybrid/two_stage_v3/cache/CACHE_REGISTRY.json
```

Validate the registry:

```text
status = COMPLETE
backbone_trainable = false
groups = 29,043
positive_groups = 9,304
student_state_dimension = 64
tabular_expert_dimension = 32
```

Leakage rules:

- cross-fitted cache: each group uses its own frozen outer-fold authority;
- outer-k cache: every train and test group is encoded by fold-k checkpoint authority, which was trained without outer fold k;
- future labels are not runtime features;
- protected attributes are not runtime features.

## Step 5 — Nested grouped head training and held-out OOF evaluation

Run:

```powershell
python scripts/recommend_hybrid/two_stage_v3/train_and_evaluate.py
```

Execution authority:

```text
12 registered head configurations
3 inner learner-group folds
3 outer folds
3 final head seeds: 42, 2026, 7319
Frozen prediction backbone: not trainable
Trainable parameters: integrated heads only
```

Required outputs include:

```text
artifacts/recommend_hybrid/two_stage_v3/model_selection/fold_0_trials.csv
artifacts/recommend_hybrid/two_stage_v3/model_selection/fold_1_trials.csv
artifacts/recommend_hybrid/two_stage_v3/model_selection/fold_2_trials.csv
artifacts/recommend_hybrid/two_stage_v3/model_selection/fold_0_selected.json
artifacts/recommend_hybrid/two_stage_v3/model_selection/fold_1_selected.json
artifacts/recommend_hybrid/two_stage_v3/model_selection/fold_2_selected.json
artifacts/recommend_hybrid/two_stage_v3/final_oof/fold_0/metrics.json
artifacts/recommend_hybrid/two_stage_v3/final_oof/fold_1/metrics.json
artifacts/recommend_hybrid/two_stage_v3/final_oof/fold_2/metrics.json
artifacts/recommend_hybrid/two_stage_v3/final_oof/OOF_PREDICTIONS.parquet
artifacts/recommend_hybrid/two_stage_v3/final_oof/NESTED_OOF_RESULTS.json
```

Do not manually edit selected configs, thresholds or checkpoints.

## Step 6 — Learner-cluster bootstrap

```powershell
python scripts/recommend_hybrid/two_stage_v3/bootstrap.py
```

Required:

```text
2,000 replicates
cluster = base_record_id
```

Output:

```text
artifacts/recommend_hybrid/two_stage_v3/final_oof/BOOTSTRAP.json
```

## Step 7 — Exact replay and safety verification

```powershell
python scripts/recommend_hybrid/two_stage_v3/verify.py
```

Verification must report PASS for:

- cache authority complete;
- frozen prediction backbone;
- no external ML ranker;
- no future/protected runtime feature;
- unchanged 29,043-group authority;
- exact group replay;
- numeric replay;
- decision replay;
- all nine final head checkpoints verified.

Output:

```text
artifacts/recommend_hybrid/two_stage_v3/final_oof/VERIFICATION.json
```

## Step 8 — Fail-closed release assessment

Run and preserve its exit code:

```powershell
python scripts/recommend_hybrid/two_stage_v3/release.py
$releaseExit = $LASTEXITCODE
```

The script writes:

```text
artifacts/recommend_hybrid/two_stage_v3/TWO_STAGE_V3_RELEASE.json
```

Interpretation:

### Main OOF gates fail

```text
TWO_STAGE_V3_EVIDENCE_BELOW_GATE
RECOMMENDATION_MODULE_NOT_COMPLETE
runtime_authorized = false
```

Stop scientific expansion. Do not run negative controls. Preserve and publish the true result.

### Main OOF gates pass but controls are not yet run

```text
TWO_STAGE_V3_MAIN_EVALUATION_PASS_CONTROLS_PENDING
RECOMMENDATION_MODULE_SCIENTIFIC_EXECUTION_NOT_COMPLETE
runtime_authorized = false
```

Do not create a runtime package. Report the main PASS and wait for the authority-bound negative-control phase.

The release cannot become runtime-authorized in this task because negative controls and the final runtime package are separate fail-closed phases.

## Step 9 — Render report

Run regardless of release outcome:

```powershell
python scripts/recommend_hybrid/two_stage_v3/render_report.py
```

Output:

```text
reports/recommend_hybrid/TWO_STAGE_V3_FINAL_RESULTS_VI.md
```

## Step 10 — Validation

```powershell
python -m pytest `
  --confcutdir=tests/recommend_hybrid/two_stage_v3 `
  tests/recommend_hybrid/two_stage_v3 `
  -q

python -m pytest tests/recommend_hybrid -q
python scripts/recommend_hybrid/validate_counterfactual.py
python -m compileall src/recommend_hybrid scripts/recommend_hybrid
git diff --check
```

Ruff is already enforced by GitHub Actions for this exact scope. Run it locally only when available:

```powershell
ruff check `
  src/recommend_hybrid/two_stage_v3 `
  scripts/recommend_hybrid/two_stage_v3 `
  tests/recommend_hybrid/two_stage_v3
```

Do not mark the execution failed solely because Ruff is unavailable locally when the current GitHub Integrated V3 workflow is green.

## Step 11 — Commit and push only V3 outputs

Review status carefully:

```powershell
git status --short
```

Do not use `git add .`.

Stage only V3 files:

```powershell
git add `
  artifacts/recommend_hybrid/two_stage_v3 `
  reports/recommend_hybrid/TWO_STAGE_V3_FINAL_RESULTS_VI.md
```

Confirm historical untracked V2.1 batches are not staged:

```powershell
git diff --cached --name-only
```

Commit and push:

```powershell
git commit -m "research(recommendation): publish integrated two-stage v3 evidence"
git push origin codex/constrained-counterfactual-recommender
```

Keep PR #4 Draft. Do not merge.

## Required Codex response

```text
INTEGRATED HYBRID TWO-STAGE V3 EXECUTION
========================================

Branch:
Remote head:
Working tree:
Untracked V2.1 preserved:
PR status:

Architecture:
Frozen prediction backbone parameters:
External ML ranker:
Trainable component:
Protocol hash:
Cache registry hash:

Groups:
Learners:
Positive groups:
Candidate rows:

Fold 0 end-to-end Precision@1:
Fold 0 positive-group coverage:
Fold 0 conditional Precision@1:
Fold 1 end-to-end Precision@1:
Fold 1 positive-group coverage:
Fold 1 conditional Precision@1:
Fold 2 end-to-end Precision@1:
Fold 2 positive-group coverage:
Fold 2 conditional Precision@1:

OOF Stage A precision:
OOF Stage A recall:
OOF Stage A ROC-AUC:
OOF Stage A Average Precision:
OOF Stage A Brier score:
OOF Stage B conditional Precision@1:
OOF Stage B Precision@1 on all positive groups:
OOF NDCG@3:
OOF MRR:
OOF end-to-end Precision@1:
OOF positive-group coverage:
OOF abstention rate:
Action diversity:
Top-action concentration:

End-to-end Precision bootstrap 95% CI:
Coverage bootstrap 95% CI:
Conditional Precision bootstrap 95% CI:

EARLY_20 end-to-end Precision@1 / coverage / conditional Precision@1:
EARLY_35 end-to-end Precision@1 / coverage / conditional Precision@1:
MIDDLE_50 end-to-end Precision@1 / coverage / conditional Precision@1:

ASSESSMENT_COMPLETION issued / precision / conditional precision:
STUDY_REGULARITY issued / precision / conditional precision:
VLE_ENGAGEMENT issued / precision / conditional precision:
QUIZ_OR_RETRIEVAL_PRACTICE issued / precision / conditional precision:
CONTENT_REVIEW issued / precision / conditional precision:

Cache authority:
Backbone frozen:
Future/protected features absent:
External ML ranker absent:
Exact numeric replay:
Exact decision replay:

Main precision gate:
Coverage gate:
Conditional precision gate:
Fold stability gate:
Stage stability gate:
Diversity gate:
Safety gate:
Reproducibility gate:

Scientific status:
Thesis-scope completion:
Negative controls required next:
Runtime authorized:
Claim boundary:
Causal validation: NOT_PERFORMED
Expert validation: NOT_PERFORMED_NOT_REQUIRED
Merge allowed: NO

Focused tests:
Recommendation tests:
Counterfactual validation:
Compileall:
Ruff:

Artifacts:
- ...

Reports:
- ...

Remaining limitations:
- ...
```
