# Phase 7 Final Endpoint vs Early Warning

The two tracks answer different questions and remain permanently separate.

## Main endpoint result

H1 is trained specifically for `F2_MIDDLE_OFFICIAL_SINGLE_CUTOFF` and obtains
Macro-F1 0.798400.

## Frozen early-warning evidence

| Stage | Observation | H1 Macro-F1 | PR-AUC |
|---|---:|---:|---:|
| E1 | 20% | 0.713635 | 0.772028 |
| E2 | 35% | 0.750632 | 0.816099 |
| M1 | 50% | 0.793953 | 0.861498 |
| L1 | 75% | 0.850333 | 0.906090 |

These early-warning values were reused from Phase 6 without recomputation.
They use a shared stage-aware checkpoint per fold/seed. The Phase 7 endpoint
uses separate task-specific checkpoints while retaining the same H1
architecture.
