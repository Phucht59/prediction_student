# Imbalance handling and fair baseline (final)

Binary risk, UCI prevalence ≈ 0.22. Outer test not used.

## Hybrid does handle imbalance

The locked Hybrid is **cost-sensitive**, not “unweighted”:

- FIT only: `pos_weight = n_neg / n_pos` (same formula every fold).
- XGB baseline: `scale_pos_weight = n_neg / n_pos`.
- LR/DT/RF/SVM: `class_weight=balanced`.
- CatBoost: `auto_class_weights=Balanced`.

That is the same family of methods as “class weights / cost-sensitive learning” in the proposal, alongside SMOTE/ADASYN.

## SMOTE / ADASYN (proposal requirement)

Tested **FIT-only** on Hybrid tensors (scaled static + aggregate + sequence). STOP/VALID untouched.

**Result: not selected.** Interpolating CNN/LSTM sequences does not create real G1/G2 or VLE weeks. UCI screen: SMOTE lowered F1 and collapsed S1 recall (e.g. 0.49). Focal+SMOTE also increased fold variance. Negative result is kept as evidence, not hidden.

Selected Hybrid recipe: **class-weighted BCE + STOP F1 threshold**. Ranking metrics (AP, ROC-AUC) do not use a threshold.

Low Precision/Recall at **S0 / 20%** is mainly **missing information** (no grades / short VLE), not “no imbalance method”. At S2 / 75–100% F1 is ~0.80–0.84.

## Fair baseline (fixed)

Old Panel A gave trees **last/mean/max/std/slope** of the sequence. Hybrid had to learn those from the tensor. That is extra engineering, not extra raw data — still unfair for an architecture claim.

**Current comparator:** one-weight baselines on **the same Hybrid tensors** (static + aggregate + masked temporal + progress). No engineered summaries. One estimator per family, all stages. Full tables: `PARITY_ONE_WEIGHT.md`.

Same-run Hybrid vs that ceiling (AP, 3×3):

- UCI S1 Hybrid 0.811 vs LR 0.769; S2 0.913 vs LR 0.896. S0 RF 0.480 vs Hybrid 0.456.
- OULAD 20% XGB 0.766 vs Hybrid 0.747. 35–100% |Δ AP| vs XGB ≤ 0.003.

Do not claim OULAD architecture win against this ceiling. Serving model remains Hybrid CNN–BiLSTM.
