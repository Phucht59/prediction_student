# Action-Aware Integrated Two-Stage V4 — Local Execution Task

## Authority

Repository:

```text
C:\hufit\kltn
```

Branch:

```text
codex/constrained-counterfactual-recommender
```

V3 evidence is immutable historical evidence:

```text
End-to-end Precision@1 = 0.6431
Positive-group coverage = 0.5213
Stage A precision = 0.6826
Conditional Precision@1 = 0.9421
Status = TWO_STAGE_V3_EVIDENCE_BELOW_GATE
```

V4 corrects one registered architectural defect: V3 candidate binary loss was applied only to positive groups, leaving action logits unsupervised on negative groups. V4 applies candidate binary supervision to every valid candidate and combines direct recommendability with masked noisy-OR action recommendability.

No label, release gate, frozen embedding, candidate row, learner, stage or action is changed.

## Architecture

```text
Frozen residual CNN–BiLSTM backbone — 160,492 parameters
+ direct recommendability neural head
+ all-group candidate action neural head
+ masked noisy-OR action recommendability
+ preregistered direct/action joint gate
+ stage-specific selective thresholds
```

No XGBoost, LightGBM, LambdaMART, Logistic Regression, Random Forest, SVM, HistGradientBoosting or external recommendation ranker is permitted.

## Release gates

```text
Held-out end-to-end Precision@1 >= 0.80
Learner-bootstrap lower 95% Precision@1 >= 0.78
Positive-group coverage >= 0.50
Conditional Precision@1 >= 0.80
Each outer fold Precision@1 >= 0.75
Supported stage Precision@1 >= 0.70
Action diversity >= 3
Top-action concentration <= 0.75
```

Do not modify these gates after execution.

## Preserve historical files

Existing untracked V2.1 batches must remain untouched. Do not add, delete, move or modify them. Do not use `git add .`.

## Step 1 — Synchronize

```powershell
git checkout codex/constrained-counterfactual-recommender
git pull --ff-only origin codex/constrained-counterfactual-recommender
git log -1 --oneline
git status --short
```

Confirm these files exist:

```text
configs/recommend_hybrid/two_stage_v4_protocol.yaml
src/recommend_hybrid/two_stage_v4/model.py
src/recommend_hybrid/two_stage_v4/metrics.py
src/recommend_hybrid/two_stage_v4/selection.py
scripts/recommend_hybrid/two_stage_v4/train_and_evaluate.py
```

## Step 2 — Focused tests

```powershell
python -m pytest `
  --confcutdir=tests/recommend_hybrid/two_stage_v4 `
  tests/recommend_hybrid/two_stage_v4 `
  -q
```

Do not continue if tests fail.

## Step 3 — Validate inherited frozen authorities

```powershell
python scripts/recommend_hybrid/validate_checkpoint_authority.py
```

Confirm the existing V3 cache remains complete:

```powershell
python -c "import json, pathlib; p=pathlib.Path('artifacts/recommend_hybrid/two_stage_v3/cache/CACHE_REGISTRY.json'); d=json.loads(p.read_text()); assert d['status']=='COMPLETE'; assert d['backbone_trainable'] is False; assert d['groups']==29043; assert d['positive_groups']==9304; print(d['protocol_sha256'])"
```

Do not rebuild the CNN–BiLSTM backbone. Do not change the repaired candidate parquet.

## Step 4 — Nested V4 head training and held-out OOF evaluation

```powershell
python scripts/recommend_hybrid/two_stage_v4/train_and_evaluate.py
```

Execution authority:

```text
12 preregistered V4 head configurations
3 inner learner-group folds
3 outer folds
3 final seeds: 42, 2026, 7319
frozen prediction backbone: not trainable
candidate binary population: ALL_VALID_CANDIDATES
```

Required outputs:

```text
artifacts/recommend_hybrid/two_stage_v4/model_selection/fold_0_trials.csv
artifacts/recommend_hybrid/two_stage_v4/model_selection/fold_1_trials.csv
artifacts/recommend_hybrid/two_stage_v4/model_selection/fold_2_trials.csv
artifacts/recommend_hybrid/two_stage_v4/model_selection/fold_0_selected.json
artifacts/recommend_hybrid/two_stage_v4/model_selection/fold_1_selected.json
artifacts/recommend_hybrid/two_stage_v4/model_selection/fold_2_selected.json
artifacts/recommend_hybrid/two_stage_v4/final_oof/fold_0/metrics.json
artifacts/recommend_hybrid/two_stage_v4/final_oof/fold_1/metrics.json
artifacts/recommend_hybrid/two_stage_v4/final_oof/fold_2/metrics.json
artifacts/recommend_hybrid/two_stage_v4/final_oof/OOF_PREDICTIONS.parquet
artifacts/recommend_hybrid/two_stage_v4/final_oof/NESTED_OOF_RESULTS.json
```

Do not manually edit selected configurations, blend weights, stage thresholds or action thresholds.

## Step 5 — Learner-cluster bootstrap

```powershell
python scripts/recommend_hybrid/two_stage_v4/bootstrap.py
```

Required:

```text
replicates = 2,000
cluster = base_record_id
```

## Step 6 — Exact replay and safety verification

```powershell
python scripts/recommend_hybrid/two_stage_v4/verify.py
```

Verification must pass:

```text
cache_complete
prediction_backbone_frozen
candidate_binary_all_groups
external_ml_ranker_absent
future_and_protected_features_absent
group_authority_unchanged
exact_group_replay
numeric_replay
decision_replay
all_head_checkpoints_verified
```

## Step 7 — Fail-closed release

```powershell
python scripts/recommend_hybrid/two_stage_v4/release.py
$releaseExit = $LASTEXITCODE
```

If main gates fail:

```text
TWO_STAGE_V4_EVIDENCE_BELOW_GATE
RECOMMENDATION_MODULE_NOT_COMPLETE
runtime_authorized = false
```

Stop. Do not run negative controls and do not modify labels, features, registered configs, thresholds or gates.

If all main gates pass:

```text
TWO_STAGE_V4_MAIN_EVALUATION_PASS_CONTROLS_PENDING
RECOMMENDATION_MODULE_SCIENTIFIC_EXECUTION_NOT_COMPLETE
runtime_authorized = false
```

Stop before runtime packaging. Authority-bound negative controls are the next separate phase.

## Step 8 — Render report

Run regardless of release result:

```powershell
python scripts/recommend_hybrid/two_stage_v4/render_report.py
```

Output:

```text
reports/recommend_hybrid/TWO_STAGE_V4_FINAL_RESULTS_VI.md
```

## Step 9 — Validation

```powershell
python -m pytest `
  --confcutdir=tests/recommend_hybrid/two_stage_v4 `
  tests/recommend_hybrid/two_stage_v4 `
  -q

python -m pytest tests/recommend_hybrid -q
python scripts/recommend_hybrid/validate_counterfactual.py
python -m compileall src/recommend_hybrid scripts/recommend_hybrid
git diff --check
```

Run Ruff only when available locally. GitHub Actions is the authority for V4 Ruff validation:

```powershell
ruff check `
  src/recommend_hybrid/two_stage_v4 `
  scripts/recommend_hybrid/two_stage_v4 `
  tests/recommend_hybrid/two_stage_v4
```

## Step 10 — Stage only V4 evidence

```powershell
git status --short

git add `
  artifacts/recommend_hybrid/two_stage_v4 `
  reports/recommend_hybrid/TWO_STAGE_V4_FINAL_RESULTS_VI.md

git diff --cached --name-only
```

Confirm no V2.1 batch is staged.

```powershell
git commit -m "research(recommendation): publish action-aware two-stage v4 evidence"
git push origin codex/constrained-counterfactual-recommender
```

Keep PR #4 Draft. Do not merge.

## Required Codex response

```text
ACTION-AWARE INTEGRATED TWO-STAGE V4 EXECUTION
==============================================

Branch:
Remote head:
Working tree:
Untracked V2.1 preserved:
PR status:

Architecture:
Frozen prediction backbone parameters:
External ML ranker:
Candidate binary population:
Protocol hash:
Inherited cache registry hash:

Groups:
Learners:
Positive groups:
Candidate rows:

Fold 0 end-to-end Precision@1 / coverage / conditional Precision@1:
Fold 1 end-to-end Precision@1 / coverage / conditional Precision@1:
Fold 2 end-to-end Precision@1 / coverage / conditional Precision@1:

OOF direct gate ROC-AUC / AP / Brier:
OOF action-derived gate ROC-AUC / AP / Brier:
OOF joint gate ROC-AUC / AP / Brier:
OOF Stage A precision:
OOF Stage A recall:
OOF Stage B conditional Precision@1:
OOF ranking-only Precision@1:
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

EARLY_20 end-to-end P@1 / coverage / conditional P@1:
EARLY_35 end-to-end P@1 / coverage / conditional P@1:
MIDDLE_50 end-to-end P@1 / coverage / conditional P@1:

ASSESSMENT_COMPLETION issued / precision / conditional precision:
STUDY_REGULARITY issued / precision / conditional precision:
VLE_ENGAGEMENT issued / precision / conditional precision:
QUIZ_OR_RETRIEVAL_PRACTICE issued / precision / conditional precision:
CONTENT_REVIEW issued / precision / conditional precision:

Cache authority:
Backbone frozen:
All-group candidate supervision:
Future/protected features absent:
External ML ranker absent:
Exact numeric replay:
Exact decision replay:

Main precision gate:
Bootstrap precision gate:
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
