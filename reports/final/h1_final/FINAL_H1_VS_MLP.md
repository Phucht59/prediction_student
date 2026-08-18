# Final H1 versus MLP

Primary answer: **NO — PRACTICAL TIE**.

- H1 Macro-F1: **0.777138**
- MLP Macro-F1: **0.777599**
- H1 − MLP: **-0.000461**
- Fold direction: H1 higher in **1/3**
- Seed/fold direction: H1 higher in **10/15**
- Paired grouped bootstrap population delta: **-0.000430**
- 95% CI: **[-0.002778, 0.001858]**

| Outer fold | MLP | H0 | H1 | H1 − MLP | H1 − H0 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.780035 | 0.773816 | 0.778984 | -0.001051 | 0.005168 |
| 1 | 0.773327 | 0.772580 | 0.772170 | -0.001157 | -0.000410 |
| 2 | 0.779436 | 0.778242 | 0.780260 | 0.000825 | 0.002018 |

The fold-averaged primary metric and pooled-observation bootstrap point estimate
differ slightly because they use different aggregation weights; both place the
difference close to zero. The interval crosses zero. H1's inner-development
advantage of +0.006080 over MLP did not generalize and reversed to −0.000461.
No model change follows this result.
