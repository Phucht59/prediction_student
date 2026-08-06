# Local Codex Prompt — Recommendation V2 scientific optimization

Continue only on branch:

```text
codex/recommendation-v2-scientific-optimization
```

## Goal

Run and validate Recommendation V2 in this order:

1. Audit the action taxonomy.
2. Run constrained behaviour simulations through the frozen Hybrid ensemble.
3. Evaluate full-population intervention eligibility.
4. Select ranking utility weights on validation only.
5. Report final test metrics and baselines.
6. Validate artefacts, tests, Ruff, compile and Git LFS.
7. Commit and push compact evidence only.

Do not modify the frozen Hybrid checkpoints, scientific silver labels, stage cutoffs, outer folds or official seeds.

## Environment

Use:

```powershell
cd C:\hufit\kltn
$PY = "C:\Users\tranh\AppData\Local\Programs\Python\Python310\python.exe"
```

Confirm:

```powershell
git checkout codex/recommendation-v2-scientific-optimization
git pull --ff-only origin codex/recommendation-v2-scientific-optimization
nvidia-smi
& $PY -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Required device:

```text
NVIDIA GeForce RTX 2060 6 GB
PyTorch CUDA available: true
```

## Preflight

```powershell
& $PY -m pytest tests/recommend_hybrid/test_recommendation_v2.py -q
& $PY -m compileall src/recommend_hybrid/v2 scripts/recommend_hybrid/v2
& $PY -m ruff check src/recommend_hybrid/v2 scripts/recommend_hybrid/v2 tests/recommend_hybrid/test_recommendation_v2.py
```

Fix only implementation errors. Do not change scientific gates to make results look better.

## Full run

```powershell
& $PY scripts/recommend_hybrid/v2/run_recommendation_v2.py `
  --device cuda `
  --batch-size 512
```

If CUDA OOM occurs, reduce only batch size:

```text
512 → 256 → 128 → 64
```

Do not reduce:

- 4 stages;
- 5 actions;
- 3 simulation strengths;
- 3 outer folds;
- 5 frozen Hybrid seeds;
- validation/test separation.

## Required scientific interpretation

The three tasks must remain separate:

```text
SHOULD_INTERVENE
WHICH_BEHAVIOURAL_ACTION
MODEL_BASED_RISK_SENSITIVITY
```

- Eligibility is evaluated on all cutoff-eligible learner-stage groups using held-out final risk outcome only as an offline target.
- Ranking metrics measure consistency with scientific silver labels.
- Simulated risk reduction is frozen-model response, not a causal effect.
- Governance routes and human escalation are outside the learned five-action ranker.
- `ASSESSMENT_TIMELINESS` remains a research candidate unless its audit and expert review pass.
- Do not claim a metric is 100% real-world recommendation accuracy.
- Runtime authorization remains false.

## Required artefacts

```text
artifacts/recommend_hybrid/v2/taxonomy_audit.json
artifacts/recommend_hybrid/v2/simulation_summary.json
artifacts/recommend_hybrid/v2/FULL_POPULATION_EVIDENCE.json
reports/recommend_hybrid/v2/ACTION_TAXONOMY_AUDIT.md
reports/recommend_hybrid/v2/HYBRID_INTERVENTION_SENSITIVITY.md
reports/recommend_hybrid/v2/FULL_POPULATION_RECOMMENDATION_RESULTS.md
reports/recommend_hybrid/v2/RECOMMENDATION_V2_VALIDATION.json
```

Do not commit learner-level:

```text
artifacts/recommend_hybrid/v2/simulation_rows.parquet
```

## Final checks

```powershell
& $PY scripts/recommend_hybrid/v2/validate_recommendation_v2.py
& $PY -m pytest tests/recommend_hybrid/test_recommendation_v2.py -q
& $PY -m compileall src/recommend_hybrid/v2 scripts/recommend_hybrid/v2
& $PY -m ruff check src/recommend_hybrid/v2 scripts/recommend_hybrid/v2 tests/recommend_hybrid/test_recommendation_v2.py
git diff --check
git lfs fsck
git status --short
```

## Commit and push

Commit only code, compact JSON evidence and Markdown reports. Preserve unrelated untracked files.

```powershell
git add <files in scope>
git commit -m "complete recommendation v2 scientific optimization"
git push origin codex/recommendation-v2-scientific-optimization
```

Do not merge PR #8. Keep it draft until independent review.

## Final response format

```text
RECOMMENDATION V2 COMPLETION
============================

Branch:
Commit:
Push:
CUDA:
GPU:

Taxonomy audit:
Five learned actions:
Governance routes outside ranker:
Assessment timeliness candidate:

Eligibility population:
Validation groups:
Test groups:
Selected policy:
Test AUROC:
Test PR-AUC:
Test Precision:
Test Recall:
Test F1:
Test Balanced Accuracy:
Test Brier:
Test ECE:
Intervention rate:
False-issue rate:
Missed-support rate:
Defer rate:

Ranking issued groups:
Precision@1:
Recall@1:
Recall@3:
NDCG@3:
MRR:
Pairwise accuracy:
Positive-group coverage:
Best baseline:
Improvement over best baseline:

Simulation rows:
Constraint violations:
Mean risk delta by action/stage:
Positive reduction fraction:
Monotonic strength fraction:
Causal claim: NO

Validation:
Tests:
Ruff:
Compile:
Git LFS:
Runtime authorized: NO
Known limitations:
```
