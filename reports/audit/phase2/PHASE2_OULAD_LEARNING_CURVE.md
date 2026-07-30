# Phase 2 — OULAD Learning Curve

## Design

`DIAGNOSTIC_ONLY`: CNN-BiLSTM, preregistered outer fold 0, seed 42, both
protocol inner folds, one uninterrupted 30-epoch trajectory per inner fold.
One shared checkpoint objective is aggregated across 20/35/50/75% stages.

## Aggregate results

| Signal | Best epoch | Best value |
| --- | ---: | ---: |
| Macro-F1 @ 0.5 | 3 | 0.7650 |
| NLL | 3 | 0.4689 |
| PR-AUC | 7 | 0.8303 |
| Inner-threshold Macro-F1 | 7 | 0.7750 |
| Brier | 3 | 0.1545 |
| ECE | 3 | 0.0607 |

At epoch 4, mean NLL was 0.4732 and fixed
Macro-F1 was 0.7595. Best post-4 NLL
improvement was -0.0004; best post-4 fixed Macro-F1 improvement
was +0.0007. Therefore the four-epoch finding is
**INCONCLUSIVE**.

Epochs within 1% of best NLL are 3, 4, 9. Overfitting is material and sustained from epoch 17.
There is no evidence about epochs beyond 30, so no extrapolation is made.
The evidence supports a Phase 3 training cap of 15 epochs with inner early
stopping; it does not support spending search budget beyond 30.

## Stage convergence by NLL

| Stage | Best epoch | Best NLL |
| --- | ---: | ---: |
| E1_EARLY_20PCT | 3 | 0.5876 |
| E2_EARLY_35PCT | 3 | 0.5216 |
| L1_LATE_75PCT | 9 | 0.3346 |
| M1_MIDDLE_FROZEN | 3 | 0.4311 |

Stage-specific optima are descriptive only. The protocol still selects one
shared estimator across stages. Per-fold threshold-optimized Macro-F1 uses the
same validation fold for threshold and epoch diagnosis and is therefore
optimistic; it is not presented as an unbiased final metric.

## Required questions

1. **Does epoch 4 underfit?** Inconclusive: fold-0 NLL selects 3, fold-1
   selects 9, while the aggregate NLL optimum is 3.
2. **Do metrics improve after epoch 4?** PR-AUC and threshold-optimized
   Macro-F1 improve slightly through epoch 7; aggregate NLL and F1@0.5 do not.
3. **Best region?** Epochs 3-9, depending on the preregistered objective.
4. **Best NLL?** Aggregate epoch 3;
   fold-specific epochs [3, 9].
5. **Best Macro-F1?** F1@0.5 at epoch
   3; inner-threshold Macro-F1 at epoch
   7.
6. **Best calibration?** Aggregate Brier and ECE both select epoch 3.
7. **Do stages converge alike?** No. Early/middle stages select epoch 3 by
   NLL, while 75% selects epoch 9.
8. **Does 75% overfit earlier or later than 20%?** Later by the observed NLL
   optimum (9 versus 3); this is descriptive inner evidence.
9. **Is >4 epochs required?** Not consistently. Later epochs help ranking
   signals, but not aggregate NLL/F1@0.5 and not both folds under NLL.
10. **Is >30 epochs required?** No evidence; the trajectory ends at 30 and
    sustained degradation begins much earlier.
