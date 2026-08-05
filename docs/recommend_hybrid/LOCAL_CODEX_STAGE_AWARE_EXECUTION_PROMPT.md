# Local Codex Execution Prompt — Four-Stage Recommendation and Causal Evidence

Copy the prompt below into the local Codex session. Codex is the execution agent only. It must not redesign the experiment.

---

You are executing a preregistered local experiment for repository `Phucht59/prediction_student`.

## Authority and branch

1. Work only on branch `codex/stage-aware-causal-recommendation`.
2. Do not read, merge, cherry-pick, or copy code from any other non-main branch.
3. The branch was created from main commit `f6da032602a7f90ee8826f216b92bd2eb536d59c`.
4. Fetch the remote branch and confirm the working tree is clean before execution.
5. Do not modify the frozen canonical Hybrid checkpoints, their manifests, architecture, or existing official final artifacts.

Run:

```powershell
git fetch origin
git checkout codex/stage-aware-causal-recommendation
git pull --ff-only origin codex/stage-aware-causal-recommendation
git lfs pull
git status --short
git rev-parse HEAD
```

Stop if the branch is wrong, the pull is not fast-forward, required LFS objects are missing, or the repository has unrelated local changes.

## Environment checks

Use the existing project environment. Install only the locked dependencies if required:

```powershell
python --version
python -m pip install -r requirements-lock.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The intended machine has an NVIDIA GTX 2060 6 GB. CUDA is preferred. Do not change model dimensions or scientific protocol because of hardware.

Confirm these local inputs exist:

```text
data/raw/studentVle.csv
data/raw/vle.csv
data/raw/assessments.csv
data/raw/studentAssessment.csv
artifacts/recommend_hybrid/RECOMMEND_HYBRID_CHECKPOINT_MANIFEST.json
all checkpoint files referenced by that manifest
artifacts/final/h1_final/predictions.parquet
```

## Static validation before the long run

Run exactly:

```powershell
python -m compileall src/recommend_hybrid/final src/recommend_hybrid/causal scripts/recommend_hybrid/final scripts/recommend_hybrid/causal scripts/recommend_hybrid/run_complete_stage_aware_recommendation.py
python -m pytest tests/recommend_hybrid/test_stage_aware_causal.py tests/recommend_hybrid/test_frozen_hybrid_imbalance.py tests/recommend_hybrid/test_four_stage_action_head.py -q
```

If a test fails, diagnose the actual implementation error. You may make the smallest code fix required, but you must not alter:

- stage order `EARLY_20`, `EARLY_35`, `MIDDLE_50`, `LATE_75`;
- action order or action identities;
- treatment definitions or minimum improvements;
- train-only treatment fitting;
- student-grouped splitting;
- cross-fitted AIPW method;
- identifiability gates;
- causal issuance gates;
- 3 outer folds × 5 seeds for the action head;
- 1,000 student-cluster bootstrap iterations;
- imbalance modes `none`, `class_weight`, `smote`, `adasyn`;
- claim boundary or runtime authorization.

Do not weaken or delete a test merely to make it pass.

## One complete execution

Run the full workflow once:

```powershell
python scripts/recommend_hybrid/run_complete_stage_aware_recommendation.py `
  --rebuild-landmark `
  --rebuild-labels `
  --chunksize 750000 `
  --embedding-batch-size 128 `
  --training-batch-size 512 `
  --epochs 50 `
  --patience 8 `
  --device cuda `
  --bootstrap 1000
```

This command must perform all of the following:

1. Rebuild the canonical OULAD landmark source using disk-backed SQLite aggregation.
2. Generate frozen OOF Hybrid embeddings at 20%, 35%, 50%, and 75%.
3. Rebuild scientific candidates, train-only label models, and silver labels.
4. Train the integrated conditional action head using 3 grouped outer folds × 5 seeds.
5. Calibrate four stage thresholds using validation rows only.
6. Produce held-out OOF ranking evidence including `LATE_75`.
7. Run frozen-embedding imbalance evidence for `none`, `class_weight`, `SMOTE`, and `ADASYN` overall and by stage.
8. Build and evaluate 20 target trials: 4 stages × 5 actions.
9. Run cross-fitted AIPW, overlap/balance/ESS gates, and 1,000 student-cluster bootstrap iterations.
10. Generate validation and thesis-ready reports.

## OOM policy

If CUDA runs out of memory:

1. Retry with `--embedding-batch-size 64`.
2. If action-head training still OOMs, retry with `--training-batch-size 256`.
3. If necessary, use `--device cpu` only for the action head.

Do not reduce seeds, folds, epochs cap, bootstrap iterations, feature set, stage coverage, or action coverage. Record every retry in the final execution report.

## Scientific failure policy

A below-gate or non-identifiable result is a valid scientific result.

- Do not tune on test results.
- Do not change a threshold after viewing held-out metrics.
- Do not lower SMD, overlap, ESS, sample-count, ranking, or diversity gates.
- Do not remove difficult stages or actions.
- Do not fabricate ATE, CATE, confidence intervals, or ranking metrics.
- Keep `CAUSAL_EVIDENCE_NOT_IDENTIFIABLE` when a trial fails its gate.
- Keep runtime authorization false.

## Required outputs

Verify these compact release artifacts exist:

```text
artifacts/recommend_hybrid/final_stage_aware_v2/FOUR_STAGE_ACTION_HEAD_EVIDENCE.json
artifacts/recommend_hybrid/final_stage_aware_v2/manifest.json
artifacts/recommend_hybrid/causal/imbalance/metrics.json
artifacts/recommend_hybrid/causal/imbalance/metrics_early_20.json
artifacts/recommend_hybrid/causal/imbalance/metrics_early_35.json
artifacts/recommend_hybrid/causal/imbalance/metrics_middle_50.json
artifacts/recommend_hybrid/causal/imbalance/metrics_late_75.json
artifacts/recommend_hybrid/causal/target_trials/stage_action_effects.json
artifacts/recommend_hybrid/causal/target_trials/manifest.json
artifacts/recommend_hybrid/causal/WORKFLOW_MANIFEST.json
artifacts/recommend_hybrid/STAGE_AWARE_COMPLETE_MANIFEST.json
reports/recommend_hybrid/FOUR_STAGE_ACTION_HEAD_RESULTS.md
reports/recommend_hybrid/causal/STAGE_AWARE_CAUSAL_VALIDATION.json
reports/recommend_hybrid/causal/STAGE_AWARE_CAUSAL_RESULTS.md
reports/recommend_hybrid/STAGE_AWARE_COMPLETE_VALIDATION.json
```

Large inputs, checkpoints, OOF prediction Parquet files, NPZ archives, and individual-effect CSV files are intentionally ignored by Git. Keep them locally unless existing Git LFS policy explicitly tracks them.

## Final validation

Run:

```powershell
python scripts/recommend_hybrid/validate_complete_stage_aware_release.py
python -m pytest tests/recommend_hybrid -q
git status --short
git lfs fsck
```

Inspect the reports. Do not summarize only the best stage or best balancing method. Report all four stages and all four imbalance modes.

## Commit and push

Commit only:

- code fixes genuinely required for execution;
- compact JSON manifests and evidence;
- compact Markdown reports;
- small CSV summaries already allowed by repository policy.

Do not force-add ignored large artifacts.

Use an intentional commit message such as:

```powershell
git add src scripts tests configs docs reports/recommend_hybrid artifacts/recommend_hybrid
# Review staged files before committing.
git diff --cached --stat
git commit -m "release: complete four-stage causal recommendation evidence"
git push origin codex/stage-aware-causal-recommendation
```

## Final response format

Return exactly these sections:

1. Branch and final commit SHA.
2. Static tests and full tests: PASS/FAIL with commands.
3. Four-stage ranker overall metrics.
4. Per-stage metrics for 20%, 35%, 50%, and 75%.
5. Imbalance comparison for all four methods.
6. Number of target trials estimated, non-identifiable, unavailable, and failed.
7. ATE and 95% CI for every identifiable stage-action trial.
8. Balance, overlap, trim fraction, and ESS failures.
9. Exact claim boundary.
10. Artifact paths committed and large artifacts retained locally.
11. Git LFS and working-tree status.
12. Any implementation fixes made locally, with file names and reasons.

Do not claim that recommendation display improves grades. The strongest permitted statement is observational effect estimation under the preregistered assumptions for identifiable overlap populations.

---
