# Phase 9 hybrid contribution ablation

All ablations use inner development evidence only and evaluate the same
deterministically trained model recipe.

- Full H1-R Macro-F1: 0.796611
- Residual disabled: 0.789914 (delta full-minus-disabled +0.006697)
- Temporal disabled: 0.784728 (delta full-minus-disabled +0.011884)

These results describe branch contribution; they were not used to create a
new architecture candidate.
