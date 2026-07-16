# Thesis Claims After Fairness Closure

## Allowed

- V3-D0-ENS has the highest fair point-estimate Macro-F1 (0.831126), but its advantage over the strongest fair comparator A0F-ENS is 0.002454, below the registered 0.005 margin.
- The corrected temporal-family verdict is **PRACTICAL_TIE**; it is unchanged from the old V3 headline verdict.
- D0-ENS shows a positive exploratory paired-bootstrap lead over H3CF-ENS, A1-ENS, MLD and MLF for Macro-F1, while uncertainty versus A0F-ENS and P0-ENS includes zero.
- Probability ensembles use exactly seeds 42, 2026 and 3407 with fold-specific thresholds reconstructed from pooled inner-OOF predictions.
- PostgreSQL technical lineage, reproduction, append-only constraints and least-privileged permission checks passed.

## Prohibited

- Do not claim overall or operational superiority over the strongest fair comparator.
- Do not call mean-of-seed metrics an ensemble, use a single favorable seed, or call the future-presentation benchmark untouched/external validation.
- Do not claim scientific confirmation, causal effectiveness, production validation, or recommendation effectiveness.

## Superseded

- Any V3 bootstrap row comparing a probability ensemble to a single-seed or mean-of-metrics comparator is historical mixed-contract evidence and cannot support the final verdict.
