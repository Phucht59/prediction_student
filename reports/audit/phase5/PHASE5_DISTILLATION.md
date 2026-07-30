# Phase 5 — Distillation

Distillation was triggered because H1 was within 0.003 of MLP (and already above it). Teacher targets were cross-fitted within training data. Lambda 0.10 won the screening grid, but stability H2-H1 = `-0.000158`. This is below +0.002 and provides no main gain, so H2 is rejected in favor of simpler H1.
