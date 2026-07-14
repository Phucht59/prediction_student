# Neural Sanity Ablation V2.2 conclusion

## Scientific validity

The strict validator accepted the run: 150/150 expected jobs, 9,480/9,480 prediction rows, no missing/unexpected/duplicate jobs or rows, exact outer-validation coverage, valid probability/argmax contract, recomputed scalar metrics/confusion matrices/per-class F1, valid checksums, and verified legacy-79 isolation.

S0 is an exact control reproduction of `benchmark-v2-full-20260713c`: all 1,580 matched record-level predictions agree and its fold-aggregated Macro-F1 is exactly `0.7983838344`. Therefore the S1–S5 diagnostic contrasts are interpretable in this source environment.

## Frozen estimator results

| Variant | Macro-F1 mean ± outer-fold SD | QWK | Ordinal MAE |
|---|---:|---:|---:|
| S0 control | 0.798384 ± 0.052624 | 0.801801 | 0.187440 |
| S1 drop_last=False | 0.806425 ± 0.024769 | 0.807576 | 0.180397 |
| S2 kernel=1 only | 0.749397 ± 0.057234 | 0.755302 | 0.219782 |
| S3 budget 40/8/3 | 0.833014 ± 0.041471 | 0.832254 | 0.159583 |
| S4 drop_last=False + kernel=1 | 0.797012 ± 0.037473 | 0.800831 | 0.188661 |
| S5 full sanity | 0.818127 ± 0.024364 | 0.824652 | 0.170813 |

## Main effects

- **Drop-last (S1 − S0):** `+0.008041`; wins/losses `3/2`. This improves variance and ordinal metrics modestly but does not meet the predeclared +0.010 practical-effect threshold.
- **Kernel-one alone (S2 − S0):** `−0.048986`; wins/ties/losses `0/2/3`, with materially higher seed variability and two High-class collapses. Kernel=1 is not beneficial as an isolated replacement of each fold's selected kernel.
- **Budget 40/8/3 (S3 − S0):** `+0.034630`; wins/losses `5/0`. This is the only clearly consistent practical improvement: QWK rises, ordinal MAE falls, mean within-fold seed SD falls from `0.05662` to `0.04595`, and mean High-class F1 rises from `0.76087` to `0.82013`.
- **Drop-last + kernel-one (S4 − S0):** `−0.001372`; wins/losses `2/3`. No practical evidence of benefit.
- **Full sanity (S5 − S0):** `+0.019744`; wins/losses `4/1`, but seed SD rises to `0.06248`. It is an encouraging diagnostic result, not a stable replacement for S3.
- **Budget conditional on S4 (S5 − S4):** `+0.021115`; wins/losses `4/1`. The longer budget helps even after forcing kernel=1/drop_last=False, but cannot recover the isolated kernel-one penalty completely.

## Budget and representation diagnostics

The 20-epoch budget is binding for S0: 12/25 jobs reached the cap and the median selected epoch was 19. Under S3, only 5/25 jobs reached cap. Thus the short training budget is a supported source of underperformance/variance. Kernel-2 affects folds 0–2 in S0/S1/S3 and has CNN output length 3 from a length-2 input; S2/S4/S5 enforce kernel-1 and preserve sequence length 2. The results do not support attributing the weak CNN score chiefly to the kernel-2 extra timestep, because forcing kernel-1 in S2 reduces performance and stability.

## Boundary and High-class findings

No variant created two-step boundary errors at G2 values 9, 10, 14, or 15. S3 reduces High-class errors at G2=14 from 20 to 17 and at G2=15 from 44 to 35 relative to S0. S5 improves some boundary counts but remains worse than S3 at G2=15. Kernel-one alone is unstable: 4 jobs have High F1 below 0.5 and 2 collapse to High F1=0, versus 2/0 for S0.

## Fixed references

S3 remains below the immutable V2 baselines: G2 rule by `0.064728`, HGB G1+G2 by `0.056716`, Small MLP by `0.054844`, and BiLSTM-only by `0.003449` Macro-F1. These are fixed-reference contrasts, not new paired baseline experiments. S3 is still more than 0.03 below HGB and Small MLP; CNN-BiLSTM remains a research comparator rather than a primary model.

## Decision

The evidence supports keeping the CNN-BiLSTM only as a **research comparator** with a non-binding training budget for any future controlled work. It does not support kernel-one as a universal correction, declaring S3/S5 a final model, or continuing sequence-model development as the primary path. The completed diagnostic is sufficient to propose a separately reviewed ordinal tabular MLP V3, but this phase does not implement it.
