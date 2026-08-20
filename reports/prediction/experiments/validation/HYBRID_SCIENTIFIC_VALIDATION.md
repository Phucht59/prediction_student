# Hybrid CNN–BiLSTM — scientific validation

Production architecture, weights, cutoffs, and recommendation were **not** changed. Outer-test labels were **not** used for HPO, model choice, or thresholds.

This report separates (A) evidence already locked in the repo from (B) analyses computed in this evaluation pass. Numbers below that are labeled **locked Phase 4** are the official inner 3×3. Numbers labeled **this run** are VALID scores from frozen C0 numerics (UCI 3×3 retrain; OULAD 3 folds, seed 42 checkpoints). Small drift versus locked means is expected; it does not replace the authority tables.

---

## 0. Evidence map

| Requirement | Already proven in repo? | This pass |
| --- | --- | --- |
| Nested protocol; outer not for HPO/threshold | **YES** — `FINALIZATION_DECISION.json`, `TRAINING_CONFIG.json` `outer_test_used=false` | Confirmed |
| Inner 3×3 Hybrid PR-AUC / Acc / F1 / Recall | **YES** — `uci_final.csv`, `oulad_final.csv` | Cited as authority |
| Hybrid vs LR and RF (minimum) | **YES** — same CSVs, also DT/SVM/MLP | Recomputed on paired VALID scores for tests |
| ROC-AUC, specificity, confusion, Brier | **PARTIAL** — Hybrid ECE in locked tables; no CM/Brier/ROC-AUC | **Computed** |
| Bootstrap CI, McNemar, DeLong, effect size | **NO** | **Computed** |
| Calibration plot, H₂(p); no isotonic/temperature | **PARTIAL** — ECE only | **Computed** (84 plots; no scaling) |
| Ablation Tabular / CNN / BiLSTM | **NO** for C0 (historical H0 only) | **Computed** as branch-only scoring of the trained Hybrid |
| SHAP / feature importance | **NO** | **Computed** KernelSHAP on RF of packed UCI S2 vectors |
| Fusion branch contribution | **PARTIAL** — availability unit tests | **Computed** gate masses + branch PR-AUC |
| FP/FN case studies | **NO** | **Computed** confusion counts on VALID |
| Subgroup / fairness | **NO** | **Computed** UCI sex/school/subject; OULAD gender/disability/IMD/module/age |
| Hyperparameter sensitivity / Optuna re-plot | Optuna DBs **not in this repo** | **NOT AVAILABLE** |
| Overfit by fold/seed/stage | **YES** — `OVERFIT_AUDIT.json` | Cited |
| Leakage (G3, cutoff, FIT-only, outer firewall) | **YES** — `LEAKAGE_AUDIT.json` | Cited |
| Cross-dataset + information growth | **YES** — final prediction report | Cited |
| External dataset besides UCI/OULAD | **NO** | **NOT AVAILABLE** (no third dataset; none invented) |
| Phase-4 C0 outer-test confirmation | **NO** — `outer_test_final/` is historical Phase 8, `not_current_prediction_authority=true`, includes XGB | **NOT APPLICABLE** as C0 evidence |

Integrity of this pass: `MODEL_CHANGED=false`, `HPO_PERFORMED=false`, `OUTER_OPENED=false`, `changed_files=[]`.

---

## 1. Methodology

- **Model:** one Hybrid CNN–BiLSTM (`architecture_id=C0`), UCI and OULAD share topology; inputs differ.
- **Splits:** inner FIT / STOP / VALID. Threshold from STOP only (F1, then recall, then `|t−0.5|`). Outer excluded from training, preprocessing, sampler, and threshold.
- **UCI this run:** 3 folds × 3 seeds, frozen `TRAINING_CONFIG.json` numerics, isolated trainer.
- **OULAD this run:** existing C0 checkpoints `c0_inner_fold{0,1,2}_seed42.pt`. Seeds 1201 and 2026 are **NOT AVAILABLE** (never materialized on disk).
- **Baselines:** Logistic Regression and Random Forest on the **same packed Hybrid tensors**, `class_weight=balanced`. STOP threshold chosen independently per model. No post-hoc probability calibration.
- **Forbidden here:** isotonic regression, temperature scaling, Optuna, architecture edits, promoting a sampler or baseline to production.

---

## 2. Outer-test results

**NOT APPLICABLE for current Hybrid C0.**

`artifacts/prediction/final/outer_test_final/` is a Phase 8 historical freeze (`HISTORICAL.json`: `not_current_prediction_authority=true`). It scores a different Hybrid surface (separate early vs FINAL-100, XGB still present). Using it as C0 confirmation would be a protocol error.

Locked Phase 4 status remains: `outer_test_used_for_phase4_finalization=false`. Final authority was user-authorized on **robust inner 3×3**, not nested outer confirmation.

Stability across **inner** folds/seeds is in §10 (overfit audit) and this-run VALID stds.

---

## 3. Baseline comparison

### 3.1 Locked Phase 4 inner 3×3 (authority) — PR-AUC

**UCI**

| Model | S0 | S1 | S2 | macro |
| --- | ---: | ---: | ---: | ---: |
| Hybrid | 0.4547 | **0.8214** | **0.9101** | 0.7288 |
| LR | 0.4754 | 0.7794 | 0.8812 | 0.7120 |
| RF | **0.4995** | 0.7895 | 0.9072 | **0.7320** |

**OULAD**

| Model | 20 | 35 | 50 | 75 | 100 | 5-stage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid | 0.7624 | **0.8058** | **0.8483** | **0.8885** | **0.9204** | **0.8451** |
| LR | **0.7632** | 0.7986 | 0.8399 | 0.8828 | 0.9114 | 0.8392 |
| RF | 0.7522 | 0.7940 | 0.8402 | 0.8847 | 0.9154 | 0.8373 |

Source: `reports/prediction/final/uci_final.csv`, `oulad_final.csv`. DT/SVM/MLP are in those files; RF and LR are the required minimum.

### 3.2 This-run VALID means (full suite)

UCI = 9 jobs. OULAD = 3 jobs (seed 42).

| Dataset | Level | Model | PR-AUC | ROC-AUC | F1 | Prec | Rec | Spec | Acc | Brier | ECE | H₂ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| UCI | S0 | Hybrid | 0.462 | 0.752 | 0.453 | 0.337 | 0.746 | 0.576 | 0.612 | 0.203 | 0.231 | 0.870 |
| UCI | S0 | LR | 0.475 | 0.757 | 0.470 | 0.349 | 0.741 | 0.619 | 0.645 | 0.186 | 0.179 | 0.662 |
| UCI | S0 | RF | **0.496** | **0.795** | **0.502** | 0.366 | 0.822 | 0.603 | 0.649 | **0.141** | **0.086** | 0.745 |
| UCI | S1 | Hybrid | **0.821** | 0.942 | 0.696 | 0.676 | 0.751 | 0.892 | 0.861 | 0.099 | 0.102 | **0.397** |
| UCI | S1 | LR | 0.725 | 0.906 | 0.672 | 0.599 | 0.769 | 0.859 | 0.840 | 0.113 | 0.107 | 0.516 |
| UCI | S1 | RF | 0.799 | **0.943** | **0.724** | 0.682 | 0.790 | 0.896 | 0.873 | **0.082** | **0.075** | 0.475 |
| UCI | S2 | Hybrid | 0.910 | **0.970** | **0.778** | 0.743 | 0.835 | 0.917 | **0.899** | 0.071 | 0.097 | **0.371** |
| UCI | S2 | LR | 0.816 | 0.938 | 0.721 | 0.773 | 0.686 | **0.943** | 0.888 | 0.091 | 0.092 | 0.467 |
| UCI | S2 | RF | **0.911** | 0.969 | 0.772 | 0.713 | 0.852 | 0.904 | 0.893 | **0.061** | **0.069** | 0.378 |
| OULAD | 20% | Hybrid | 0.756 | 0.784 | 0.676 | 0.621 | 0.748 | 0.657 | 0.695 | 0.205 | 0.134 | **0.545** |
| OULAD | 20% | LR | **0.765** | **0.794** | **0.686** | 0.609 | 0.786 | 0.625 | 0.694 | 0.183 | 0.055 | 0.771 |
| OULAD | 20% | RF | 0.747 | 0.773 | 0.666 | 0.584 | 0.778 | 0.585 | 0.668 | 0.187 | **0.028** | 0.826 |
| OULAD | 35% | Hybrid | **0.808** | **0.832** | **0.697** | 0.689 | 0.712 | 0.780 | 0.752 | 0.163 | 0.077 | **0.529** |
| OULAD | 50% | Hybrid | **0.848** | **0.873** | **0.731** | 0.731 | 0.734 | 0.837 | 0.797 | 0.138 | 0.075 | **0.417** |
| OULAD | 75% | Hybrid | **0.889** | **0.913** | **0.782** | 0.815 | 0.753 | 0.912 | 0.858 | 0.100 | 0.039 | **0.346** |
| OULAD | 100% | Hybrid | **0.921** | **0.939** | **0.838** | 0.894 | 0.789 | 0.956 | 0.903 | **0.074** | **0.025** | **0.310** |

This run agrees with the locked pattern: RF wins UCI S0; Hybrid is competitive or better from UCI S1 and from OULAD 35% onward. ECE is worst at UCI S0 (0.23) and OULAD 20% (0.13), and falls as information increases.

---

## 4. Statistical significance

Paired VALID scores. Hybrid − comparator. Bootstrap 400 resamples, 95% CI. DeLong on ROC-AUC. McNemar on STOP-thresholded labels. Cohen's g = McNemar effect size. Means over jobs. **Positive ΔPR-AUC = Hybrid better.**

| Dataset | Level | vs | ΔPR-AUC | 95% CI | p_boot PR | DeLong p | McNemar p | Cohen's g |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UCI | S0 | RF | −0.032 | [−0.117, +0.051] | 0.39 | 0.11 | 0.029 | −0.07 |
| UCI | S1 | RF | +0.018 | [−0.030, +0.068] | 0.42 | 0.59 | 0.58 | −0.05 |
| UCI | S2 | RF | −0.002 | [−0.028, +0.024] | 0.58 | 0.57 | 0.37 | +0.07 |
| UCI | S2 | LR | +0.092 | [+0.030, +0.162] | 0.018 | 0.020 | 0.65 | +0.06 |
| OULAD | 20% | LR | −0.009 | [−0.017, −0.001] | 0.047 | 0.017 | 0.18 | −0.00 |
| OULAD | 35% | RF | +0.018 | [+0.011, +0.025] | 0.000 | 0.000 | 0.000 | +0.08 |
| OULAD | 50% | RF | +0.011 | [+0.005, +0.017] | 0.000 | 0.003 | 0.23 | +0.05 |
| OULAD | 100% | RF | +0.011 | [+0.005, +0.016] | 0.027 | 0.026 | 0.17 | +0.05 |

Interpretation without cherry-pick:

- UCI S0: RF is better on the point estimate; the PR-AUC CI includes 0 (small n). McNemar p=0.029 says hard labels differ; effect size is small (g≈−0.07).
- UCI S1/S2 vs RF: no significant PR-AUC gap (CI crosses 0).
- UCI S2 vs LR: Hybrid ranking is better (CI excludes 0).
- OULAD 20% vs LR: LR slightly better on PR-AUC (CI just excludes 0); McNemar not significant.
- OULAD 35%+ vs RF: Hybrid PR-AUC advantage is small (~0.01) but the bootstrap CI is above 0 on 35/50/100.

p-values are not used to change the model. Effect sizes on OULAD are modest.

---

## 5. Probability and calibration

No isotonic or temperature scaling.

- Reliability diagrams: `artifacts/experiments/validation/plots/calib_*.png` (84 figures including confusion).
- UCI S0: high ECE (0.23) and high H₂ (0.87) — probabilities are uncertain and miscalibrated when no grades exist.
- UCI S2 / OULAD 75–100: ECE 0.02–0.10, H₂ 0.31–0.37 — sharper and better calibrated as information grows.
- RF often has lower ECE than Hybrid at the same stage; Hybrid has lower H₂ (more peaked) on later OULAD stages.

This does **not** justify adding calibration to production in this experiment.

---

## 6. Ablation and fusion contribution

These are **branch-only scores of the trained Hybrid**, not separately retrained networks. Retrained CNN-only / BiLSTM-only OULAD 3×3 is **NOT AVAILABLE** (compute; would also change the training distribution).

**PR-AUC of one representation**

| Dataset | Level | Tabular-only | CNN-only | BiLSTM-only | Full Hybrid (this run) |
| --- | --- | ---: | ---: | ---: | ---: |
| UCI | S0 | 0.462 | 0.462 | 0.462 | 0.462 |
| UCI | S1 | 0.526 | 0.772 | 0.771 | 0.821 |
| UCI | S2 | 0.509 | 0.904 | 0.904 | 0.910 |
| OULAD | 20% | 0.674 | 0.628 | 0.665 | 0.756 |
| OULAD | 50% | 0.750 | 0.696 | 0.788 | 0.848 |
| OULAD | 100% | 0.724 | 0.766 | 0.893 | 0.921 |

**Gate masses (mean softmax weight)**

| Dataset | Level | Tabular | CNN | BiLSTM |
| --- | --- | ---: | ---: | ---: |
| UCI | S0 | 1.00 | 0.00 | 0.00 |
| UCI | S1 | 0.025 | 0.146 | 0.829 |
| UCI | S2 | 0.024 | 0.153 | 0.824 |
| OULAD | 20% | 0.257 | 0.259 | 0.484 |
| OULAD | 100% | 0.187 | 0.297 | 0.516 |

Findings: S0 is purely tabular (availability contract). After grades appear, BiLSTM dominates UCI fusion. OULAD keeps a three-way mix; full Hybrid beats any single branch at every OULAD cutoff in this run. CNN-only is the weakest OULAD branch at 20–75%.

---

## 7. Error analysis (FP/FN)

VALID Hybrid confusion **means** (counts per fold):

| Dataset | Level | TP | FP | FN | TN | n |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| UCI | S0 | 44 | 92 | 15 | 125 | 277 |
| UCI | S1 | 44 | 23 | 15 | 194 | 277 |
| UCI | S2 | 49 | 18 | 10 | 200 | 277 |
| OULAD | 20% | 1884 | 1172 | 639 | 2234 | 5929 |
| OULAD | 50% | 1508 | 558 | 550 | 2849 | 5465 |
| OULAD | 100% | 1257 | 150 | 336 | 3257 | 5000 |

S0: many FP (specificity 0.58) — the model over-calls risk without G1/G2. Later stages cut FP. OULAD 20% still has a large FP mass; 100% FN remain (~336) while FP drop. Per-row scores: `artifacts/experiments/validation/scores_uci.parquet`, `scores_oulad.parquet`. Narrative “case studies” of named students are not added here because record IDs are hashes, not identifiable dossiers.

---

## 8. Subgroup and fairness

Gaps = max − min across groups on VALID Hybrid (mean over jobs). Attributes present in the data only.

| Dataset | Attribute | PR-AUC gap | TPR gap | FPR gap |
| --- | --- | ---: | ---: | ---: |
| UCI | sex | 0.17 | 0.41 | 0.33 |
| UCI | school | 0.19 | 0.46 | 0.35 |
| UCI | subject | 0.23 | 0.42 | 0.34 |
| OULAD | gender | 0.06 | 0.12 | 0.10 |
| OULAD | disability | 0.05 | 0.17 | 0.21 |
| OULAD | imd_band | 0.23 | 0.38 | 0.30 |
| OULAD | code_module | 0.30 | 0.60 | 0.35 |
| OULAD | age_band | 0.18 | 0.31 | 0.16 |

Example, OULAD 20%: female PR-AUC 0.72 vs male 0.79; disability=Y has higher TPR (0.86) and higher FPR (0.52) than N. Module and IMD gaps are the largest OULAD slices — they mix **base-rate and course difficulty** with model behavior; they are reported, not “fixed” by a new architecture.

UCI n per VALID fold is ~277, so sex/school/subject TPR gaps are noisy.

---

## 9. Explainability

- **Fusion:** §6 gate masses.
- **Permutation / DeepSHAP on Hybrid:** not run (KernelSHAP on flattened Hybrid tensors would mix OHE statics with interpolated weeks).
- **KernelSHAP on RF**, same packed UCI S2 vectors, fold 0, 80 evaluate / 40 background: top magnitudes are packed indices `f58, f63, f57, f62` (late flattened temporal/aggregate region). Full table: `artifacts/experiments/validation/shap_rf_uci_s2.csv`. This explains the **RF on Hybrid features**, not the neural gate.

---

## 10. Sensitivity and robustness

**Optuna:** **NOT AVAILABLE.** Study databases lived under `C:\hufit\kltn` and are not on this machine. Frozen numerics remain `artifacts/prediction/final/TRAINING_CONFIG.json`. No re-HPO.

**Locked overfit audit** (Phase 4 official 3×3, train−VALID PR-AUC gap):

| State | PR-AUC | gap | class |
| --- | ---: | ---: | --- |
| UCI S0 | 0.4547 | 0.125 | HIGH |
| UCI S1 | 0.8214 | 0.035 | MODERATE |
| UCI S2 | 0.9101 | 0.020 | MODERATE |
| OULAD 20–100 | 0.762–0.920 | 0.034→0.016 | LOW |

UCI S0 is the robustness weak point (overfit + RF superiority + worst calibration). OULAD gaps shrink as cutoff increases.

---

## 11. Cross-dataset

Same C0 topology. UCI is a short grade sequence (T=2); OULAD is week-level VLE (11 channels). Raw PR-AUC is **not** comparable across datasets. The honest contrast is margin versus the strongest active baseline and the information-growth slope:

- UCI: 0.45 → 0.82 → 0.91 as G1 then G2 appear.
- OULAD: 0.76 → 0.81 → 0.85 → 0.89 → 0.92 as cutoff 20→100.

---

## 12. Leakage / overfit audit (locked)

`artifacts/prediction/final/LEAKAGE_AUDIT.json`:

- G3 never a predictor; S0 has no G1/G2.
- OULAD `observation_start ≤ t < cutoff`; `final_result` / `date_unregistration` forbidden as predictors.
- FIT-only preprocessing; FIT/STOP/VALID disjoint; `outer_test_used=false`.
- Known confounder (not leakage in the feature sense): OULAD 100% history length is associated with Withdrawn (`length_roc_auc_withdrawn≈0.993`). 100% must not be read as a pure academic-risk score.

This evaluation used the same cutoff-safe builders and did not resample VALID.

---

## 13. External validation

**NOT AVAILABLE.** No licensed third student dataset is in the workspace. No synthetic “external” set was created.

---

## 14. Conclusions

1. **Do not change Hybrid authority.** Inner locked 3×3 plus this-run paired tests do not show a consistent, large Hybrid loss to LR/RF after S0. OULAD 35%+ Hybrid vs RF ΔPR-AUC ≈ +0.01 with CI above 0. UCI S0 remains the documented limitation (RF better; HIGH overfit; poor calibration).
2. **Do not add SMOTE/ADASYN** (previous experiment) and **do not add isotonic/temperature** in this pass.
3. Missing nested **C0 outer confirmation** is a protocol gap, not a license to mine Phase 8 outer numbers.
4. Fairness gaps exist (module, IMD, UCI school/subject) and should be reported as limitations, not “solved” by a new architecture without a pre-specified fairness objective.
5. Ablation supports keeping the three-way fusion: full Hybrid beats single-branch scores on OULAD; UCI S0 correctly ignores temporal branches.

Artifacts: `artifacts/experiments/validation/` (metrics, scores, stat_tests, ablation, subgroup, plots). Code: `experiments/validation/` (isolated from `src/prediction`).
