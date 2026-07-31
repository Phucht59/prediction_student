# H0 vs H1 endpoint forensic comparison

## What is directly comparable

- Exact record identity: PASS (15,378)
- Exact target identity: PASS
- Exact cutoff-day identity: PASS
- Exact outer-fold identity: PASS
- Five-seed probability ensembles: PASS
- Per-fold inner-only thresholds: PASS

Therefore 0.828084 and 0.798400 are directly comparable as final predictions
on the same endpoint population. They are **not** a controlled
architecture-only comparison because the full input and training recipes differ.

## Architecture

H0 has 100,938 parameters, kernels [2,3],
24 convolution channels and dilation 2. H1 has
160,492 parameters, kernels [2,3,5],
32 convolution channels, dilation 1 and a tabular residual expert. More
parameters did not compensate for the changed endpoint signal and recipe.

## Prediction-quality evidence

| Metric | H0 | H1 | H1-H0 |
|---|---:|---:|---:|
| Macro-F1 | 0.828084 | 0.798400 | -0.029684 |
| PR-AUC | 0.893355 | 0.863039 | -0.030316 |
| ROC-AUC | 0.908156 | 0.876142 | -0.032014 |
| NLL | 0.358778 | 0.406292 | +0.047514 |
| Brier | 0.113355 | 0.130702 | +0.017346 |

H1 creates +267 additional false positives and
+168 additional false negatives. It loses in both
specificity and risk recall.

## Early-warning versus endpoint

The frozen Phase 6 H1 result at the same M1/F2 50% cutoff was Macro-F1
0.793953 and PR-AUC
0.861498. Phase 7 endpoint H1
is close to those values (0.798400,
0.863039); it did not suffer a new 0.03 collapse when the endpoint
runner started. Instead, it faithfully carried the score-free early-warning
representation into the endpoint. Historical H0's advantage comes from a
different endpoint-specific feature, preprocessing, pretraining and training
recipe.
