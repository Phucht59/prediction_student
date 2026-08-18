# Phase 5 — MLP Gap Analysis

Under the fair Phase 5 INNER protocol, MLP is below H0 rather than above it. The stability gap M0-H0 is `-0.004001`. Therefore closed-gap fraction is not applicable.

| Stage | MLP only | H0 only | Disagreement | Correlation |
| --- | --- | --- | --- | --- |
| E1_EARLY_20PCT | 0.061752 | 0.067367 | 0.129118 | 0.895918 |
| E2_EARLY_35PCT | 0.043364 | 0.047537 | 0.090901 | 0.940867 |
| L1_LATE_75PCT | 0.020595 | 0.021896 | 0.042491 | 0.974311 |
| M1_MIDDLE_FROZEN | 0.031732 | 0.036448 | 0.068181 | 0.961409 |

The models are highly correlated but make non-identical errors, particularly at 20%, supporting a bounded complementarity test without implying causation.
