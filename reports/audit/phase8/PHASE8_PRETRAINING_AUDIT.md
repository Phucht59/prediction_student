# Pretraining audit

H0 did execute `P1_MASKED_AND_NEXT_WEEK`; H1 did not request or execute
pretraining.

H0 provenance is concrete:

- 15 final runs each reference a distinct pretraining checkpoint hash.
- Replay maximum absolute difference is zero.
- Pretraining is fit separately on each outer-training partition.
- Five epochs, with masked valid weeks and ten registered masked/next-week
  tasks.
- Outer/future data access is false.

The registered inner gate measured:

- Macro-F1 gain: +0.001911
- PR-AUC gain: +0.002283

Pretraining is therefore a confirmed difference and plausible secondary
contributor, but its controlled inner gain is far smaller than the 0.029684
final gap. It is not sufficient as a single-cause explanation.
