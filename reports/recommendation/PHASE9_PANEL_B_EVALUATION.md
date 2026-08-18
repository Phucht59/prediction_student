# Phase 9 Panel B evaluation

`PHASE9_DATA = DONE`

This is an AUTOMATED_REFERENCE_EVALUATION against Gemini 3.5 + Gemini 3.1 weak references.
It is not expert ground truth and not a causal claim about student outcomes.
Gemini 3.5 and Gemini 3.1 are the same model family.

## Reference agreement

- Overall exact agreement: `596/750` (0.794667).

| Action | Exact | Linear kappa | Quadratic kappa | DUAL | SINGLE | NO_REFERENCE |
|---|---:|---:|---:|---:|---:|---:|
| assessment_recovery | 1.000000 | NA | NA | 41 | 0 | 109 |
| re_engagement | 0.760000 | 0.816800 | 0.913001 | 150 | 0 | 0 |
| study_planning | 0.720000 | 0.682936 | 0.781937 | 150 | 0 | 0 |
| progress_monitoring | 0.700000 | 0.515017 | 0.622103 | 150 | 0 | 0 |
| retrieval_practice | 0.793333 | 0.475612 | 0.547413 | 103 | 0 | 47 |

## Frozen model comparison

| Model | NDCG@3 | 95% CI | P@1 | Recall@3 | MRR | Pairwise | Invalid | Coverage |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| EBM | 0.926767 | [0.912227, 0.940571] | 0.753333 | 0.849103 | 0.917949 | 0.809544 | 0.000000 | 1.000000 |
| ACTION_STAGE_PRIOR | 0.823777 | [0.800863, 0.846320] | 0.653333 | 0.793974 | 0.862821 | 0.658998 | 0.000000 | 1.000000 |
| RIDGE | 0.910247 | [0.888836, 0.929399] | 0.746667 | 0.829872 | 0.903846 | 0.774614 | 0.000000 | 1.000000 |
| RANDOM_FOREST | 0.923084 | [0.908189, 0.937584] | 0.753333 | 0.847179 | 0.914103 | 0.801928 | 0.000000 | 1.000000 |

## Paired NDCG@3 deltas vs EBM

| Contrast | Mean delta | 95% CI |
|---|---:|---|
| EBM - ACTION_STAGE_PRIOR | 0.103018 | [0.075002, 0.130990] |
| EBM - RIDGE | 0.016409 | [0.004038, 0.030343] |
| EBM - RANDOM_FOREST | 0.003846 | [-0.003877, 0.012428] |

## A5 REVIEW

- A5 top-1 rate: `0.380000`.
- A5 top-3 rate: `0.593333`.
- REVIEW plan rate: `0.380000`.
A5 remains REVIEW and was not suppressed.

No model was tuned on Panel B.
