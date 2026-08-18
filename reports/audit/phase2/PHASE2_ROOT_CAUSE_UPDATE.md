# Phase 2 — Root-Cause Update

| Root cause | Phase 1 | Phase 2 evidence | Updated status |
| --- | --- | --- | --- |
| Fixed four-epoch refit | High-priority design issue | Post-4 NLL gain -0.0004; fold gains [{'inner_fold': 0, 'best_nll_epoch': 3, 'best_nll': 0.4584104767555373, 'nll_improvement_vs_epoch4': 0.009011141841222814}, {'inner_fold': 1, 'best_nll_epoch': 9, 'best_nll': 0.4695277103477782, 'nll_improvement_vs_epoch4': 0.009363530158281896}] | INCONCLUSIVE |
| Checkpoint/threshold objective mismatch | Confirmed design issue | Policies separated; NLL recommended before threshold fitting | CONFIRMED AND REPAIRED IN CONTROL PLANE |
| Scalar two-gate fusion | Architectural bottleneck hypothesis | Not manipulated in Phase 2 | INCONCLUSIVE |
| Greater CNN depth | Historical small gains/no replacement | Not manipulated; training control had priority | NOT JUSTIFIED |
| Strong tabular aggregates | Confirmed limitation | Unchanged and leakage-safe | CONFIRMED LIMITATION |

The diagnostic does not establish that architecture is the dominant cause.
Corrected training control should be used before any architecture expansion.
