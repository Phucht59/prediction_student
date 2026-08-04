# Hybrid-only final local execution task

## Authority

Repository:

```text
C:\hufit\kltn
```

Branch:

```text
codex/constrained-counterfactual-recommender
```

The final recommendation architecture contains exactly one learned model:

```text
frozen residual CNN–BiLSTM
```

Do not run or resume the V2.1 XGBoost/LambdaMART controls or ablations. Those artifacts are historical experiments and are not final-runtime authority.

## Meaning of the 80% requirement

The frozen primary gate is:

```text
Top-1 Precision on issued recommendations >= 0.80
```

The comparison target is the action-specific direct future-behavior silver label in held-out OULAD trajectories.

The gate also requires:

```text
Actionable coverage >= 0.50
Bootstrap Precision@1 lower 95% bound >= 0.78
Each outer fold Precision@1 >= 0.75
Supported stage Precision@1 >= 0.70
Action diversity >= 3
Top-action concentration <= 0.75
```

Do not call this causal effectiveness, guaranteed grade improvement, or expert validation.

## Prohibited changes

Do not:

- train XGBoost, LightGBM, LambdaMART, Logistic Regression, a pairwise ranker, or another learned recommendation model;
- change the silver-label definition after seeing results;
- lower the 0.80 precision gate or the 0.50 coverage gate;
- use future behavior as a scoring input;
- add protected attributes to scoring;
- fit thresholds on an outer test fold;
- remove difficult stages, actions, courses, or learners based on their results;
- merge PR #4.

## Step 1 — Synchronize

```powershell
git checkout codex/constrained-counterfactual-recommender
git pull --ff-only origin codex/constrained-counterfactual-recommender
git status --short
git log -1 --oneline
```

The working tree must be clean before execution.

## Step 2 — Focused code tests

```powershell
python -m pytest `
  --confcutdir=tests/recommend_hybrid/hybrid_only_final `
  tests/recommend_hybrid/hybrid_only_final `
  -q
```

Do not continue if focused tests fail.

## Step 3 — Build direct future silver cohort

```powershell
python scripts/recommend_hybrid/hybrid_only_final/build_silver_dataset.py
python scripts/recommend_hybrid/hybrid_only_final/normalize_evidence.py
```

Required outputs:

```text
artifacts/recommend_hybrid/hybrid_only_final/dataset/candidate_rows.parquet
artifacts/recommend_hybrid/hybrid_only_final/dataset/learner_stage_groups.parquet
artifacts/recommend_hybrid/hybrid_only_final/dataset/cohort_flow.json
artifacts/recommend_hybrid/hybrid_only_final/dataset/schema.json
artifacts/recommend_hybrid/hybrid_only_final/dataset/CHECKSUMS.json
```

Confirm:

- `additional_learned_model_used = false`;
- future columns are evaluation-only;
- at least two scientific actions exist in every rankable group;
- evidence and need are normalized onto frozen runtime semantics.

## Step 4 — Nested deterministic tuning

```powershell
python scripts/recommend_hybrid/hybrid_only_final/tune_and_evaluate_fast.py
```

This searches only arithmetic weights and abstention thresholds. It must not install or invoke XGBoost, LightGBM, scikit-learn models, or any auxiliary ranker.

The fast evaluator must filter unavailable, high-uncertainty, low-evidence and insufficient-risk-reduction candidates before ranking, matching runtime behavior.

## Step 5 — Refine selection from inner trials only

```powershell
python scripts/recommend_hybrid/hybrid_only_final/refine_selection.py
```

This step does not search new configurations and does not inspect outer-test results for selection. It only reads completed inner-validation trial tables:

- if any config meets Precision@1 >= 0.80 and coverage >= 0.50, choose the one with the best coverage and stability;
- otherwise choose the highest inner Precision@1 among configurations that still meet coverage >= 0.50.

Do not edit the selected JSON files manually.

Required outputs after Steps 4–5:

```text
artifacts/recommend_hybrid/hybrid_only_final/model_selection/fold_0_trials.csv
artifacts/recommend_hybrid/hybrid_only_final/model_selection/fold_1_trials.csv
artifacts/recommend_hybrid/hybrid_only_final/model_selection/fold_2_trials.csv
artifacts/recommend_hybrid/hybrid_only_final/model_selection/fold_0_selected.json
artifacts/recommend_hybrid/hybrid_only_final/model_selection/fold_1_selected.json
artifacts/recommend_hybrid/hybrid_only_final/model_selection/fold_2_selected.json
artifacts/recommend_hybrid/hybrid_only_final/evaluation/OOF_PREDICTIONS.parquet
artifacts/recommend_hybrid/hybrid_only_final/evaluation/OOF_RESULTS.json
artifacts/recommend_hybrid/hybrid_only_final/evaluation/FOLD_METRICS.csv
artifacts/recommend_hybrid/hybrid_only_final/evaluation/BASELINE_METRICS.csv
artifacts/recommend_hybrid/hybrid_only_final/HYBRID_ONLY_SELECTED_CONFIG.json
```

## Step 6 — Learner-cluster bootstrap

```powershell
python scripts/recommend_hybrid/hybrid_only_final/bootstrap.py
```

Required:

```text
2,000 learner-cluster replicates
```

Output:

```text
artifacts/recommend_hybrid/hybrid_only_final/evaluation/BOOTSTRAP.json
```

## Step 7 — Deterministic and safety verification

```powershell
python scripts/recommend_hybrid/hybrid_only_final/verify_fast.py
```

The verification must report PASS for:

- deterministic replay;
- no future feature in scoring;
- no protected feature in scoring;
- no forbidden learned recommendation model;
- zero unknown runtime actions;
- zero availability/prerequisite violations.

## Step 8 — Fail-closed release

Run:

```powershell
python scripts/recommend_hybrid/hybrid_only_final/release.py
$releaseExit = $LASTEXITCODE
```

### All gates pass

```text
HYBRID_ONLY_OFFLINE_SILVER_VALIDATED
RECOMMENDATION_MODULE_COMPLETE
runtime_authorized = true
```

Only in this case may the script generate:

```text
configs/recommend_hybrid/hybrid_only_selected.yaml
```

### Full execution completes but one or more gates fail

```text
HYBRID_ONLY_SILVER_EVIDENCE_BELOW_GATE
RECOMMENDATION_MODULE_NOT_COMPLETE
runtime_authorized = false
```

Do not change the protocol or rerun with relaxed gates. Preserve the actual result.

## Step 9 — Render final report

Run regardless of release outcome:

```powershell
python scripts/recommend_hybrid/hybrid_only_final/render_report.py
```

Output:

```text
reports/recommend_hybrid/HYBRID_ONLY_FINAL_RESULTS_VI.md
```

## Step 10 — Full validation

```powershell
python -m pytest tests/recommend_hybrid/hybrid_only_final -q
python -m pytest tests/recommend_hybrid -q
python scripts/recommend_hybrid/validate_counterfactual.py
python -m compileall src/recommend_hybrid scripts/recommend_hybrid
ruff check src/recommend_hybrid scripts/recommend_hybrid tests/recommend_hybrid
git diff --check
```

If the release passed, instantiate the release-gated runtime loader and `RecommendHybridOnlyFinalPipeline` in a smoke test. Construction must fail closed when the release config is removed or its checksum is changed.

## Step 11 — Commit and push

Commit all valid generated artifacts and reports, including failed gates. Do not omit unfavorable folds or actions.

```powershell
git add configs/recommend_hybrid artifacts/recommend_hybrid/hybrid_only_final reports/recommend_hybrid/HYBRID_ONLY_FINAL_RESULTS_VI.md
git commit -m "release(recommendation): publish hybrid-only held-out evidence"
git push origin codex/constrained-counterfactual-recommender
```

Keep PR #4 Draft. Do not merge.

## Required Codex response

```text
HYBRID-ONLY FINAL RECOMMENDATION COMPLETION
===========================================

Branch:
Remote head:
Working tree:
PR status:

Learned model authority:
Additional learned ranker:
Protocol hash:
Dataset hash:

Learners:
Ranking groups:
Candidate rows:
Groups with positive silver action:

Outer fold 0 Precision@1:
Outer fold 0 coverage:
Outer fold 1 Precision@1:
Outer fold 1 coverage:
Outer fold 2 Precision@1:
Outer fold 2 coverage:

OOF Precision@1:
OOF actionable coverage:
Precision bootstrap 95% CI:
Coverage bootstrap 95% CI:
Action diversity:
Top-action concentration:

Risk-reduction-only baseline:
Evidence-only baseline:
Lowest-workload baseline:

Temporal leakage:
Protected-feature use:
Constraint violations:
Deterministic replay:

Precision gate:
Coverage gate:
Fold stability gate:
Stage stability gate:
Safety gate:
Reproducibility gate:

Scientific status:
Thesis-scope completion:
Runtime authorized:
Claim boundary:
Causal validation: NOT_PERFORMED
Expert validation: NOT_PERFORMED_NOT_REQUIRED
Merge allowed: NO

Artifacts:
- ...

Reports:
- ...

Remaining limitations:
- ...
```
