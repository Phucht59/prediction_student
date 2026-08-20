# Imbalance experiment: SMOTE / ADASYN on frozen Hybrid CNN–BiLSTM

Isolated experiment. Production Hybrid CNN–BiLSTM and Recommendation V were not changed.

## 1. Objective

Under the exact frozen Hybrid configuration, does train-only SMOTE or ADASYN improve VALID metrics versus CONTROL?

## 2. Why imbalance handling is required

The thesis requires minority-class synthetic sampling (SMOTE, ADASYN) as an investigated method. That requirement is satisfied by a controlled comparison, not by replacing the locked Hybrid.

## 3. Experimental design

Three conditions. Everything except the FIT sampler is identical: architecture C0, Phase-4 numerics, folds 0–2, seeds 42/1201/2026, FIT-only preprocessing, STOP early-stop on macro PR-AUC, STOP-chosen threshold, untouched VALID.

Independent variable: `SAMPLER ∈ {control, smote, adasyn}`.

## 4. CONTROL

Hybrid trained on the original FIT distribution. No resampling.

## 5. SMOTE

`imblearn.SMOTE` on student-level flattened Hybrid tensors of FIT rows only, per information level, then unpacked back to static / aggregate / temporal / progress.

## 6. ADASYN

`imblearn.ADASYN` with the same packing and the same train-only rule.

Temporal sequences are not synthesized timestep-by-timestep. UCI masks are restored from the information level. OULAD masks are recovered from non-zero temporal support. This is defensible for UCI (`T≤2`). For OULAD it interpolates week vectors and is a limitation, not a new sequence-SMOTE algorithm.

## 7. Data leakage prevention

- Split first (inner FIT/STOP/VALID). Outer test is excluded and was not used to choose a sampler.
- Preprocessor fit on FIT only.
- Sampler fit on FIT tensors only.
- STOP and VALID never resampled; tensor fingerprints checked after each job.
- OULAD events remain `observation_start <= t < cutoff`.
- Split source: `recovered_from_official_oof_valid` (the kltn Phase-1 parquet bundle is absent on this machine; `inner_fold` is the official reconstructed OOF VALID assignment, one fold per student; FIT/STOP use `StratifiedGroupKFold(n_splits=5, random_state=42)`).

Machine-readable audit: `artifacts/experiments/imbalance/LEAKAGE_AUDIT.json`.

## 8. Dataset / information levels

- UCI: S0, S1, S2
- OULAD: 20pct, 35pct, 50pct, 75pct, 100pct as views of one Hybrid, not separate models

## 9. Training configuration

Copied from `artifacts/prediction/final/TRAINING_CONFIG.json` (lr, dropout, weight decay, batch size, `pos_weight_multiplier`, entropy floor). AdamW, grad clip 1.0, max 24 epochs, patience 8, early-stop on STOP macro PR-AUC. `pos_weight` is computed from original FIT labels and kept after resampling. No HPO.

## 10. Results

Means ± std over 3 folds × 3 seeds (9 jobs). Primary metric is PR-AUC.

### UCI

| level | sampler | PR-AUC | F1 | Precision | Recall | Accuracy | Macro-F1 | Bal. acc. | Minority recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 | CONTROL | 0.4621 | 0.4531 | 0.3369 | 0.7459 | 0.6123 | 0.5712 | 0.6608 | 0.7459 |
| S0 | SMOTE | 0.4487 | 0.4513 | 0.3178 | 0.8100 | 0.5695 | 0.5436 | 0.6573 | 0.8100 |
| S0 | ADASYN | 0.4420 | 0.4498 | 0.3308 | 0.7380 | 0.6033 | 0.5612 | 0.6521 | 0.7380 |
| S1 | CONTROL | 0.8209 | 0.6959 | 0.6764 | 0.7510 | 0.8614 | 0.8026 | 0.8216 | 0.7510 |
| S1 | SMOTE | 0.8098 | 0.6873 | 0.6561 | 0.7565 | 0.8550 | 0.7960 | 0.8196 | 0.7565 |
| S1 | ADASYN | 0.8074 | 0.6993 | 0.6479 | 0.7941 | 0.8550 | 0.8015 | 0.8332 | 0.7941 |
| S2 | CONTROL | 0.9096 | 0.7782 | 0.7429 | 0.8349 | 0.8993 | 0.8564 | 0.8760 | 0.8349 |
| S2 | SMOTE | 0.9077 | 0.7703 | 0.7462 | 0.8248 | 0.8978 | 0.8521 | 0.8715 | 0.8248 |
| S2 | ADASYN | 0.9041 | 0.7895 | 0.7424 | 0.8537 | 0.9035 | 0.8634 | 0.8856 | 0.8537 |

### OULAD

| level | sampler | PR-AUC | F1 | Precision | Recall | Accuracy | Macro-F1 | Bal. acc. | Minority recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20pct | CONTROL | 0.7564 | 0.6769 | 0.6189 | 0.7484 | 0.6963 | 0.6949 | 0.7030 | 0.7484 |
| 20pct | SMOTE | 0.7414 | 0.6690 | 0.5984 | 0.7614 | 0.6795 | 0.6789 | 0.6906 | 0.7614 |
| 20pct | ADASYN | 0.7314 | 0.6661 | 0.5842 | 0.7754 | 0.6694 | 0.6692 | 0.6832 | 0.7754 |
| 35pct | CONTROL | 0.8061 | 0.7028 | 0.6723 | 0.7384 | 0.7498 | 0.7431 | 0.7478 | 0.7384 |
| 35pct | SMOTE | 0.7897 | 0.6870 | 0.6382 | 0.7514 | 0.7256 | 0.7207 | 0.7302 | 0.7514 |
| 35pct | ADASYN | 0.7817 | 0.6862 | 0.6460 | 0.7349 | 0.7306 | 0.7246 | 0.7311 | 0.7349 |
| 50pct | CONTROL | 0.8463 | 0.7303 | 0.7317 | 0.7316 | 0.7967 | 0.7835 | 0.7842 | 0.7316 |
| 50pct | SMOTE | 0.8366 | 0.7215 | 0.6989 | 0.7488 | 0.7825 | 0.7713 | 0.7760 | 0.7488 |
| 50pct | ADASYN | 0.8305 | 0.7194 | 0.7003 | 0.7410 | 0.7825 | 0.7708 | 0.7743 | 0.7410 |
| 75pct | CONTROL | 0.8888 | 0.7804 | 0.8159 | 0.7491 | 0.8577 | 0.8376 | 0.8313 | 0.7491 |
| 75pct | SMOTE | 0.8809 | 0.7688 | 0.8364 | 0.7144 | 0.8552 | 0.8316 | 0.8208 | 0.7144 |
| 75pct | ADASYN | 0.8747 | 0.7626 | 0.8190 | 0.7185 | 0.8493 | 0.8260 | 0.8176 | 0.7185 |
| 100pct | CONTROL | 0.9216 | 0.8370 | 0.9195 | 0.7695 | 0.9047 | 0.8848 | 0.8686 | 0.7695 |
| 100pct | SMOTE | 0.9166 | 0.8293 | 0.9151 | 0.7593 | 0.9005 | 0.8795 | 0.8630 | 0.7593 |
| 100pct | ADASYN | 0.9122 | 0.8216 | 0.9093 | 0.7508 | 0.8963 | 0.8742 | 0.8576 | 0.7508 |

Raw cells: `artifacts/experiments/imbalance/results_raw.csv` (216 rows = 2 datasets × levels × 3 samplers × 3 folds × 3 seeds).

## 11. Delta vs CONTROL

Positive = sampler better than CONTROL.

| dataset | level | SMOTE Δ PR-AUC | ADASYN Δ PR-AUC | SMOTE Δ Macro-F1 | ADASYN Δ Macro-F1 | SMOTE Δ min. recall | ADASYN Δ min. recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UCI | S0 | -0.0134 | -0.0202 | -0.0276 | -0.0100 | +0.0642 | -0.0079 |
| UCI | S1 | -0.0111 | -0.0135 | -0.0066 | -0.0012 | +0.0055 | +0.0431 |
| UCI | S2 | -0.0018 | -0.0055 | -0.0043 | +0.0070 | -0.0102 | +0.0188 |
| OULAD | 20pct | -0.0150 | -0.0250 | -0.0160 | -0.0257 | +0.0130 | +0.0270 |
| OULAD | 35pct | -0.0164 | -0.0244 | -0.0224 | -0.0185 | +0.0130 | -0.0035 |
| OULAD | 50pct | -0.0097 | -0.0157 | -0.0122 | -0.0127 | +0.0172 | +0.0094 |
| OULAD | 75pct | -0.0079 | -0.0142 | -0.0060 | -0.0115 | -0.0348 | -0.0307 |
| OULAD | 100pct | -0.0050 | -0.0094 | -0.0053 | -0.0106 | -0.0101 | -0.0187 |

Mean PR-AUC delta across the eight reported levels: SMOTE **-0.0100**, ADASYN **-0.0160**.

## 12. Interpretation

Neither SMOTE nor ADASYN improved the primary metric (PR-AUC) at any UCI or OULAD information level.

Minority-class recall sometimes rose (UCI S0 SMOTE; UCI S1/S2 ADASYN; OULAD 20/35/50 SMOTE; OULAD 20/50 ADASYN). Those recall gains came with lower PR-AUC and, in most cells, lower macro-F1. That is consistent with oversampling moving the STOP-chosen threshold toward more positive predictions without improving ranking quality.

This is a valid negative result for flattened student-level SMOTE/ADASYN on the frozen Hybrid. It does not mean class imbalance is irrelevant; it means these two samplers, applied this way, did not beat CONTROL.

## 13. Limitations

- Flattened SMOTE on OULAD week tensors interpolates sequences; it is not temporally generative.
- CONTROL is re-trained in this isolated trainer with frozen numerics. Every information level is trained each epoch; production Phase 4 used stage-balanced sampling. Small drift versus the published 3×3 mean is possible.
- `pos_weight` from the frozen protocol is kept even after resampling.
- Inner-fold identity was recovered from official reconstructed OOF VALID assignments because `C:\hufit\kltn` Phase-1 parquet files are not on this machine. Outer-test IDs from `outer_test_final/predictions.parquet` have zero overlap with the recovered inner set.

## 14. Evidence for SMOTE/ADASYN usefulness

The experiment does **not** provide evidence that SMOTE or ADASYN should enter the production Hybrid. The thesis imbalance-handling requirement is met by having run and reported the controlled comparison. No sampler is selected as a new production model.

## 15. Production Hybrid was NOT changed

- `PREDICTION_AUTHORITY` = Hybrid CNN–BiLSTM (frozen)
- `MODEL_CHANGED` = false
- `FINAL_WEIGHTS_CHANGED` = false
- `HPO_PERFORMED` = false
- `OUTER_OPENED` = false
- `RECOMMENDATION_CHANGED` = false
- `changed_files` = []

Integrity snapshot: `artifacts/experiments/imbalance/INTEGRITY_AUDIT.json`.
