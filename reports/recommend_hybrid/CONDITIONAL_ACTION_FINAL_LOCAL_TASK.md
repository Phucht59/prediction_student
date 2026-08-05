# Final conditional action-ranking local task

## Purpose

This task does not train a model. It evaluates the already completed V4 held-out OOF action logits under a scientifically narrower and explicit module boundary:

```text
rank actions after external eligibility has already been decided
```

The task may complete the conditional recommendation submodule. It may not validate or authorize the end-to-end recommendation issuance system.

## Authority

Repository:

```text
C:\hufit\kltn
```

Branch:

```text
codex/constrained-counterfactual-recommender
```

Read:

```text
configs/recommend_hybrid/conditional_action_final_protocol.yaml
reports/recommend_hybrid/FINAL_CONDITIONAL_ACTION_RECOMMENDER_SCOPE.md
```

## Prohibited

Do not:

- train or retrain any model;
- modify V4 OOF predictions;
- modify silver labels;
- change the positive evaluation population;
- report conditional Precision@1 as end-to-end accuracy;
- authorize runtime;
- stage V2.1 untracked batches;
- merge PR #4.

## Commands

Synchronize:

```powershell
git checkout codex/constrained-counterfactual-recommender
git pull --ff-only origin codex/constrained-counterfactual-recommender
git log -1 --oneline
git status --short
```

Preserve the untracked V2.1 batches. Do not use `git add .`.

Run focused tests:

```powershell
python -m pytest `
  --confcutdir=tests/recommend_hybrid/conditional_action_final `
  tests/recommend_hybrid/conditional_action_final `
  -q
```

Run final evaluation:

```powershell
python scripts/recommend_hybrid/conditional_final/run_fast.py
$conditionalExit = $LASTEXITCODE
```

Required outputs:

```text
artifacts/recommend_hybrid/conditional_action_final/CONDITIONAL_ACTION_FINAL_EVIDENCE.json
reports/recommend_hybrid/CONDITIONAL_ACTION_FINAL_RESULTS_VI.md
```

## Required evidence

The script must report:

- ranking-only Precision@1 on all 9,304 positive held-out groups;
- learner-cluster bootstrap with 2,000 replicates;
- NDCG@3 and MRR;
- every outer fold;
- every stage;
- action-selection diversity;
- risk-reduction-only baseline;
- evidence-only baseline;
- lowest-workload baseline;
- 5,000 random-ranking controls;
- 5,000 stage/fold-stratified label-vector permutations;
- 5,000 action-identity permutations;
- preserved V4 end-to-end context.

## Conditional release gates

```text
Ranking-only Precision@1 >= 0.90
Bootstrap lower 95% Precision@1 >= 0.90
NDCG@3 >= 0.95
MRR >= 0.95
Every outer fold Precision@1 >= 0.90
Every supported stage Precision@1 >= 0.85
Action-selection diversity >= 4
All permutation p-values <= 0.001
Improvement over best deterministic baseline >= 0.20
Temporal leakage = 0
Protected-feature use = 0
Exact replay = PASS
```

## Interpretation

If all conditional gates pass:

```text
CONDITIONAL_ACTION_RANKING_OFFLINE_VALIDATED
CONDITIONAL_RECOMMENDATION_MODULE_COMPLETE
END_TO_END_RECOMMENDATION_SYSTEM_NOT_VALIDATED
runtime_authorized = false
```

If any gate fails:

```text
CONDITIONAL_ACTION_RANKING_EVIDENCE_BELOW_GATE
CONDITIONAL_RECOMMENDATION_MODULE_NOT_COMPLETE
END_TO_END_RECOMMENDATION_SYSTEM_NOT_VALIDATED
runtime_authorized = false
```

The script exits non-zero when conditional gates fail. Preserve the generated evidence and report regardless.

## Validation

```powershell
python -m pytest `
  --confcutdir=tests/recommend_hybrid/conditional_action_final `
  tests/recommend_hybrid/conditional_action_final `
  -q

python -m compileall `
  scripts/recommend_hybrid/conditional_final

git diff --check
```

Ruff only when available locally:

```powershell
ruff check `
  scripts/recommend_hybrid/conditional_final `
  tests/recommend_hybrid/conditional_action_final
```

GitHub Actions is authoritative for Ruff.

## Commit

Stage only:

```powershell
git add `
  artifacts/recommend_hybrid/conditional_action_final/CONDITIONAL_ACTION_FINAL_EVIDENCE.json `
  reports/recommend_hybrid/CONDITIONAL_ACTION_FINAL_RESULTS_VI.md

git diff --cached --name-only
```

Commit and push:

```powershell
git commit -m "research(recommendation): publish final conditional action evidence"
git push origin codex/constrained-counterfactual-recommender
```

Keep PR #4 Draft. Do not merge.

## Response template

```text
FINAL CONDITIONAL ACTION-RANKING EVIDENCE
=========================================

Branch:
Remote head:
Working tree:
V2.1 batches preserved:

Models trained:
Labels changed:
Module boundary:
End-to-end issuance in scope:

Positive evaluation groups:
Learners:
Ranking-only Precision@1:
Bootstrap Precision@1 95% CI:
NDCG@3:
MRR:
Action-selection diversity:
Top-action concentration:

Fold 0 Precision@1:
Fold 1 Precision@1:
Fold 2 Precision@1:

EARLY_20 Precision@1:
EARLY_35 Precision@1:
MIDDLE_50 Precision@1:

Risk-reduction-only Precision@1:
Evidence-only Precision@1:
Lowest-workload Precision@1:
Best baseline improvement:

Random ranking p-value:
Label permutation p-value:
Action identity permutation p-value:

Conditional gates pass:
Conditional scientific status:
Conditional thesis-scope completion:
End-to-end system status:
V4 end-to-end Precision@1:
V4 coverage:
Runtime authorized: false
Claim boundary:
Merge allowed: NO
```
