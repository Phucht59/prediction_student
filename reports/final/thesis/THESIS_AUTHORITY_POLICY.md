# Final thesis authority policy

## UCI

- Student-Mat primary endpoint: CNN-BiLSTM, Macro-F1 **0.901460**.
- Student-Por primary endpoint: CNN-BiLSTM, Macro-F1 **0.862259**.
- S0/S1/S2 are secondary stage-aware analyses and never replace the headlines.

## OULAD

- `0.828084` means **legacy endpoint Macro-F1** under the conservative score-
  availability proxy. It is partially scientifically valid and is not called
  target leakage or invalid.
- `0.798400` means **strict no-unverified-score endpoint Macro-F1** for H1.
- H1 early-warning results at 20%, 35%, 50% and 75% are frozen secondary
  evidence. Their mean is not a final endpoint.

The H1 architecture is particularly useful as a stage-aware early-warning
model because it combines temporal behavioral sequences with aggregate/static
student information. This is predictive evidence, not a causal intervention
claim.
