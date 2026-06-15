# Handoff Report — RA-HLPR Architectural Integrity Audit

## 1. Observation

During my investigation of the repository at `c:\Huflit\kltn`, I directly observed the following:

1. **Uncommitted Diffs in Working Directory for Locked Test Metrics**:
   Running `git diff reports/final/metrics/` shows that the locked test metrics have been modified in the working tree:
   ```json
   diff --git a/reports/final/metrics/student-por_3class_locked_test_metrics.json b/reports/final/metrics/student-por_3class_locked_test_metrics.json
   index 4bb55be..b01430a 100644
   --- a/reports/final/metrics/student-por_3class_locked_test_metrics.json
   +++ b/reports/final/metrics/student-por_3class_locked_test_metrics.json
   @@ -1,8 +1,8 @@
    {
   -    "Accuracy": 0.8461538461538461,
   -    "F1-Macro": 0.8156483004028224,
   -    "Precision-Macro": 0.7966721767321467,
   -    "Recall-Macro": 0.8394383394383395,
   -    "RMSE": 0.3922322702763681,
   -    "R2": 0.5625841184387618
   +    "Accuracy": 0.8,
   +    "F1-Macro": 0.7764887400098308,
   +    "Precision-Macro": 0.7452568279140287,
   +    "Recall-Macro": 0.8371794871794872,
   +    "RMSE": 0.4472135954999579,
   +    "R2": 0.43135935397039027
    }
   ```
   and `reports/final/metrics/xapi_3class_locked_test_metrics.json` similarly shows a degradation in metrics (F1-Macro dropped from `0.785015` to `0.765255`).

2. **Modification of Predictor Ensemble Checkpoints**:
   Checking the modification times of the ensemble checkpoints in `models/saved/final/` using PowerShell:
   ```powershell
   Get-ChildItem -Path models/saved/final -Filter *.pt | Select-Object Name, LastWriteTime
   ```
   revealed that they were overwritten on `6/15/2026 12:22 AM`, which is within the current work session:
   - `xapi_3class_cnn_bilstm_mlp_seed1337.pt` (LastWriteTime: 6/15/2026 12:22:31 AM)
   - `student-por_3class_cnn_bilstm_mlp_seed1337.pt` (LastWriteTime: 6/15/2026 12:21:47 AM)

3. **Remediation of FocalLoss Bypass**:
   Checking the diff of `src/models/models.py` against `HEAD:src/models.py` shows that the dynamic class registration bypass for `FocalLoss` (via `globals()["Focal" + "Loss"] = Focal_Loss`) has been completely removed:
   ```diff
   -class Focal_Loss(nn.Module):
   -    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
   -        super().__init__()
   -        self.weight = weight
   -        self.gamma = gamma
   -        self.reduction = reduction
   -
   -    def forward(self, inputs, targets):
   -        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
   -        pt = torch.exp(-ce_loss)
   -        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
   -        if self.reduction == 'mean':
   -            return focal_loss.mean()
   -        elif self.reduction == 'sum':
   -            return focal_loss.sum()
   -        return focal_loss
   ```
   The files `src/train_pipeline.py` and `scripts/run_pipeline.py` have also been modified to remove all imports and usages of `FocalLoss`.

4. **Pytest Unit Test Execution**:
   Running `C:\Users\THPhu\anaconda3\envs\kltn\python.exe -m pytest tests/test_recommender.py tests/test_thesis_pipeline.py -v` succeeded completely:
   ```
   tests/test_recommender.py::test_rules_generate_weak_labels_student PASSED
   tests/test_recommender.py::test_rules_generate_weak_labels_xapi PASSED
   tests/test_recommender.py::test_risk_head_and_training PASSED
   tests/test_recommender.py::test_knowledge_base_loading PASSED
   tests/test_recommender.py::test_hybrid_scorer_and_path_planner PASSED
   tests/test_recommender.py::test_evaluation_metrics PASSED
   tests/test_thesis_pipeline.py::test_model_is_cnn_bilstm_mlp_and_outputs_three_class_probabilities PASSED
   tests/test_thesis_pipeline.py::test_xapi_model_supports_independent_branch_dropouts PASSED
   tests/test_thesis_pipeline.py::test_xapi_optuna_space_matches_high_trial_configuration PASSED
   tests/test_thesis_pipeline.py::test_resampling_neighbor_count_is_configurable PASSED
   tests/test_thesis_pipeline.py::test_forbidden_architectures_and_losses_are_removed PASSED
   tests/test_thesis_pipeline.py::test_weighted_cross_entropy_supports_imbalanced_classes PASSED
   tests/test_thesis_pipeline.py::test_feature_selector_keeps_required_sequence_columns PASSED
   tests/test_thesis_pipeline.py::test_learning_path_engine_returns_staged_roadmap_not_variable_tweaks PASSED
   tests/test_thesis_pipeline.py::test_learning_path_report_has_one_row_per_student PASSED
   tests/test_thesis_pipeline.py::test_postgres_schema_stores_features_confidence_and_learning_paths PASSED
   ============================= 16 passed in 8.91s ==============================
   ```

5. **Recommendation Pipeline Integrity**:
   No hardcoded test results, facade implementations, or bypassed checks were found in `src/recommendation.py`, `src/recommender/`, or `src/evaluation/recommender_eval.py`. The recommendation logic genuinely queries the PyTorch MLP model (`RecommendationMLP`), computes scoring weights, and plans the 4-week paths dynamically.

---

## 2. Logic Chain

1. The user request requires checking that the "original CNN-BiLSTM performance predictor ensemble checkpoints and locked test metrics are completely untouched and unmodified."
2. Observation 1 shows that `reports/final/metrics/student-por_3class_locked_test_metrics.json` and `reports/final/metrics/xapi_3class_locked_test_metrics.json` have been modified in the local working directory compared to their committed `HEAD` versions.
3. Observation 2 shows that the ensemble checkpoints in `models/saved/final/` were overwritten/modified on disk (LastWriteTime matching the current session).
4. Observation 3 indicates that these modifications occurred because the `FocalLoss` class (which was dynamically registered in a bypassed manner to circumvent architecture constraints) was successfully remediated and removed. Removing the forbidden loss forced a retraining of the performance predictor, which degraded model performance and modified the locked metrics and checkpoints.
5. While the remediation of the `FocalLoss` bypass was correct and necessary, the check requiring that the original ensemble checkpoints and locked test metrics remain completely untouched and unmodified has failed.
6. According to the system instructions, "If ANY check fails, your verdict is INTEGRITY VIOLATION and you MUST reject the work product."
7. Therefore, the verdict is **INTEGRITY VIOLATION**.

---

## 3. Caveats

- The modification to the checkpoints and metrics was a direct side effect of removing `FocalLoss` to comply with the forbidden architectures constraint. Leaving the original metrics untouched would create a discrepancy where the committed metrics are stale and cannot be reproduced by the current `CrossEntropyLoss`-based codebase.
- The recommendation system (RA-HLPR) itself is clean, performs genuine scoring/planning, contains no hardcoding or facade implementations, and runs successfully.

---

## 4. Conclusion

**Verdict**: **INTEGRITY VIOLATION**

While the codebase has successfully expunged the dynamic registration bypass for `FocalLoss`, this architectural change degraded performance and resulted in the modification/overwriting of both the original CNN-BiLSTM predictor ensemble checkpoints and the locked test metrics. Because these files were modified, the project fails the strict requirement to leave them completely untouched.

---

## 5. Verification Method

To independently verify these findings:
1. Run `git status` to see the modified locked test metrics files:
   ```bash
   git status reports/final/metrics/
   ```
2. Run `git diff` to view the degraded metric values:
   ```bash
   git diff reports/final/metrics/
   ```
3. Run the unit tests to confirm the 16 core tests pass cleanly:
   ```bash
   C:\Users\THPhu\anaconda3\envs\kltn\python.exe -m pytest tests/test_recommender.py tests/test_thesis_pipeline.py -v
   ```

---

## Forensic Audit Report

**Work Product**: Downstream Risk-Aware Hybrid Learning Path Recommender (RA-HLPR) system
**Profile**: General Project
**Verdict**: INTEGRITY VIOLATION

### Phase Results
- **Bypass and Facade Detection**: PASS — No dynamic class registration bypass, hardcoding, or dummy implementations remain in the codebase.
- **Predictor and Metric Verification**: FAIL — The original CNN-BiLSTM performance predictor ensemble checkpoints and locked test metrics were modified/overwritten.
- **Unit Test Execution**: PASS — The test suite compiles and runs cleanly with 16 passed tests.
- **Codebase Integrity Audit**: PASS — No facades or fabricated test results are present in the recommendation engine.
