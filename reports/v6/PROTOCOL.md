# V6.0 integrated system protocol

V6.0 is an integration release, not a V5.5 architecture search. The repository
base is `24cca2b7f0904504e6f1c937af04589938e1a73f`; the scientific prediction
base is V5.1 at `308370cf6c6f16e65cc0f0aaa3f38393ae141e16`; and the recommendation
base is V5.2 at `b9087ceb1600582ad1351b134a2f4c4d9af77d89`.

The target, F2 cutoff, grouped split, fixed seeds, primary Macro-F1 metric and
Future OULAD lock are unchanged. Candidate selection is restricted to
outer-training data. Outer-test predictions are never used to choose an
architecture, epoch, threshold, loss weight or extension.

The registered ladder is A (locked V5.1), B (minimal masked/next-week
pretraining), C (withdrawal-survival and final-outcome heads), D (risk-ranking
objective), and E (small graph context only after a positive graph audit). Each
stage adds one idea and must pass its pre-registered gate before the next stage
is eligible.

V5.2 recommendation logic remains the technical base. V6 adds a versioned risk
profile and decision-policy adapter; it does not invent expert labels or claim
causal student-outcome improvement. Missing real expert labels remain explicitly
`PENDING_EXPERT_LABELS`.

