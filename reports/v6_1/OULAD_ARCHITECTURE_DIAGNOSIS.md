# OULAD CNN–BiLSTM architecture diagnosis (V6.1)

## Evidence protection

All new training was isolated under `artifacts/v6_1_oulad_architecture_diagnosis`.
Frozen V5/V5.1/V6 checkpoints, OOF predictions and canonical results were not
overwritten. Architecture selection used only outer-training fold 0 inner CV;
Future OULAD remained locked.

## Architecture diagnosis

| Model | Params | Macro-F1 | At-risk F1 | PR-AUC | Brier |
|---|---:|---:|---:|---:|---:|
| Aggregate + static only | 22,849 | 0.8247 ± 0.0015 | 0.7809 | 0.8860 | 0.1185 |
| CNN temporal | 22,897 | 0.8208 ± 0.0022 | 0.7756 | 0.8845 | 0.1197 |
| BiLSTM temporal | 80,689 | 0.8248 ± 0.0007 | 0.7825 | 0.8855 | 0.1190 |
| Serial CNN-BiLSTM temporal | 88,945 | 0.8245 ± 0.0009 | 0.7807 | 0.8847 | 0.1202 |
| Full serial hybrid | 99,443 | 0.8282 ± 0.0009 | 0.7854 | 0.8894 | 0.1175 |
| Parameter-matched CNN | 76,561 | 0.8225 ± 0.0016 | 0.7784 | 0.8837 | 0.1202 |
| Serial + CNN skip | 108,947 | 0.8269 ± 0.0013 | 0.7846 | 0.8890 | 0.1180 |
| Parallel CNN || BiLSTM | 108,947 | 0.8277 ± 0.0011 | 0.7861 | 0.8874 | 0.1185 |

## Hypothesis verdicts

- H1_capacity_imbalance: **PARTIAL** — cnn_matched_delta_vs_cnn_small=+0.0017, cnn_matched_delta_vs_bilstm=-0.0024
- H2_serial_bottleneck: **NOT_SUPPORTED** — best_skip_or_parallel_delta_vs_serial_full=-0.0005
- H3_dilation_mismatch: **SUPPORTED** — best_d1_or_multidilation_delta_vs_d2=+0.0011
- H4_aggregate_redundancy: **PARTIAL** — aggregate_static_delta_vs_full_serial=-0.0035
- H5_data_limitation: **PARTIAL** — 

Registered scenario: **B_CAPACITY_BIAS_PARTIALLY_CONFIRMED**.

## Selected architecture

None — no new architecture passed the preregistered development gate.

Config hash: `707e42bab25f4cb7bae79049ca93c12f85c875b86e5e61385d53ab8892b1dd0c`.

## Final outer evaluation

Development gate did not pass, so the preregistered rule prohibited opening a new outer evaluation. Frozen V5.1 and XGBoost evidence was left unchanged.

## Temporal-order evidence

Candidate role: `best_architectural_diagnostic_no_outer_candidate`; threshold was frozen from original
inner OOF and reused for every destruction condition.

- original: Macro-F1 0.8287, delta +0.0000
- reversed: Macro-F1 0.8243, delta -0.0044
- shuffled: Macro-F1 0.8221, delta -0.0066
- bag_of_weeks: Macro-F1 0.8250, delta -0.0036

## Recommendation semantic correction

Circular pseudo-observed logic existed and was removed. `activity_level`,
`inactivity_streak`, `assessment_progress`, and `grade_trend` now require real
pre-cutoff sequence measurements. Missing observed state causes abstention rather
than probability-to-behavior fabrication.

Withdrawal reliability is
**EXPLORATORY_DISABLED_FOR_RECOMMENDATION** because observed withdrawal recall
was at most 0.00273224043715847. The horizon
may remain exploratory, but it cannot assert an engagement mechanism or trigger a
mechanism-specific recommendation.

## Validation

- Passed: 77
- Skipped: 1
- Failed: 0
- Frozen evidence modified: no
- Outer test used for selection: no
- Future OULAD accessed: no

## Scientific conclusion

The evidence is classified as **B_CAPACITY_BIAS_PARTIALLY_CONFIRMED**. Capacity imbalance and
dilation did disadvantage the small CNN modestly: parameter matching added
+0.0017
Macro-F1 and dilation one added
+0.0011.
However, the matched CNN still trailed the BiLSTM by
-0.0024; direct
skip and parallel paths did not beat the full serial control, while aggregate +
static alone was already close to the temporal models. Therefore the serial
design was not shown to suppress a useful CNN expert. The dominant explanation
is limited incremental local signal plus redundancy with compact features, with
a smaller contribution from capacity and dilation choices.

This conclusion follows the preregistered aggregate inner-CV rules; no seed or
fold was selected after the fact, and the negative outer-evaluation gate result
was retained.
