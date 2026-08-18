# Phase 7 weak supervision

Phase 7 aggregates frozen Phase 6 weak labels into probabilistic silver labels.
Gemini, Gemma, and Behavior remain weak sources, not ground truth.
No API call, no Panel B, no FINAL stage, no EBM training, and no Optuna search were used.

## Authority

- Phase 6 source manifest version: `recommendation.phase6_source_manifest.v1`.
- Weak-supervision config version: `recommendation.weak_supervision.v5`.
- Label-model version: `recommendation.weak_supervision.v5`.
- Cardinality: `4`.
- Project seeds: `[42, 1201, 2026]`.
- Panel-B overlap: `0`.

## Matrices

| Action | Shape | Effective LFs |
|---|---|---|
| assessment_recovery | 500×3 | `LF_GEMINI35, LF_GEMMA4, LF_BEHAVIOR` |
| re_engagement | 500×3 | `LF_GEMINI35, LF_GEMMA4, LF_BEHAVIOR` |
| study_planning | 500×3 | `LF_GEMINI35, LF_GEMMA4, LF_BEHAVIOR` |
| progress_monitoring | 500×2 | `LF_GEMINI35, LF_GEMINI31` |
| retrieval_practice | 500×3 | `LF_GEMINI35, LF_GEMMA4, LF_BEHAVIOR` |

## Pre-Snorkel source diagnostics

| Action | Source | Coverage | ABSTAIN rate | Distribution |
|---|---|---:|---:|---|
| assessment_recovery | LF_GEMINI35 | 0.282000 | 0.718000 | `{'0': 0, '1': 0, '2': 17, '3': 124, 'ABSTAIN': 359}` |
| assessment_recovery | LF_GEMMA4 | 0.282000 | 0.718000 | `{'0': 0, '1': 0, '2': 19, '3': 122, 'ABSTAIN': 359}` |
| assessment_recovery | LF_BEHAVIOR | 0.282000 | 0.718000 | `{'0': 0, '1': 0, '2': 20, '3': 121, 'ABSTAIN': 359}` |
| re_engagement | LF_GEMINI35 | 1.000000 | 0.000000 | `{'0': 192, '1': 98, '2': 122, '3': 88, 'ABSTAIN': 0}` |
| re_engagement | LF_GEMMA4 | 1.000000 | 0.000000 | `{'0': 179, '1': 53, '2': 120, '3': 148, 'ABSTAIN': 0}` |
| re_engagement | LF_BEHAVIOR | 1.000000 | 0.000000 | `{'0': 168, '1': 109, '2': 117, '3': 106, 'ABSTAIN': 0}` |
| study_planning | LF_GEMINI35 | 1.000000 | 0.000000 | `{'0': 53, '1': 190, '2': 203, '3': 54, 'ABSTAIN': 0}` |
| study_planning | LF_GEMMA4 | 1.000000 | 0.000000 | `{'0': 25, '1': 282, '2': 193, '3': 0, 'ABSTAIN': 0}` |
| study_planning | LF_BEHAVIOR | 0.492000 | 0.508000 | `{'0': 68, '1': 71, '2': 56, '3': 51, 'ABSTAIN': 254}` |
| progress_monitoring | LF_GEMINI35 | 1.000000 | 0.000000 | `{'0': 54, '1': 178, '2': 212, '3': 56, 'ABSTAIN': 0}` |
| progress_monitoring | LF_GEMINI31 | 1.000000 | 0.000000 | `{'0': 82, '1': 214, '2': 92, '3': 112, 'ABSTAIN': 0}` |
| retrieval_practice | LF_GEMINI35 | 0.622000 | 0.378000 | `{'0': 14, '1': 117, '2': 176, '3': 4, 'ABSTAIN': 189}` |
| retrieval_practice | LF_GEMMA4 | 0.622000 | 0.378000 | `{'0': 24, '1': 176, '2': 111, '3': 0, 'ABSTAIN': 189}` |
| retrieval_practice | LF_BEHAVIOR | 0.622000 | 0.378000 | `{'0': 78, '1': 78, '2': 77, '3': 78, 'ABSTAIN': 189}` |

## Pairwise overlap and agreement

Agreement is between weak-label sources, not expert annotators. Gemini 3.5 and Gemini 3.1 are the same model family.

| Action | Pair | Overlap | Exact | Conflict | Linear kappa | Quadratic kappa |
|---|---|---:|---:|---:|---:|---:|
| assessment_recovery | LF_GEMINI35_vs_LF_GEMMA4 | 141 | 0.956000 | 0.156028 | 0.299774 | 0.299774 |
| assessment_recovery | LF_GEMINI35_vs_LF_BEHAVIOR | 141 | 0.958000 | 0.148936 | 0.347366 | 0.347366 |
| assessment_recovery | LF_GEMMA4_vs_LF_BEHAVIOR | 141 | 0.974000 | 0.092199 | 0.613210 | 0.613210 |
| re_engagement | LF_GEMINI35_vs_LF_GEMMA4 | 500 | 0.588000 | 0.412000 | 0.631515 | 0.769998 |
| re_engagement | LF_GEMINI35_vs_LF_BEHAVIOR | 500 | 0.586000 | 0.414000 | 0.616322 | 0.755880 |
| re_engagement | LF_GEMMA4_vs_LF_BEHAVIOR | 500 | 0.608000 | 0.392000 | 0.641740 | 0.772001 |
| study_planning | LF_GEMINI35_vs_LF_GEMMA4 | 500 | 0.564000 | 0.436000 | 0.418445 | 0.566416 |
| study_planning | LF_GEMINI35_vs_LF_BEHAVIOR | 246 | 0.142000 | 0.711382 | 0.156271 | 0.301978 |
| study_planning | LF_GEMMA4_vs_LF_BEHAVIOR | 246 | 0.130000 | 0.735772 | 0.029415 | 0.100889 |
| progress_monitoring | LF_GEMINI35_vs_LF_GEMINI31 | 500 | 0.562000 | 0.438000 | 0.556862 | 0.713714 |
| retrieval_practice | LF_GEMINI35_vs_LF_GEMMA4 | 311 | 0.638000 | 0.581994 | 0.016558 | 0.044278 |
| retrieval_practice | LF_GEMINI35_vs_LF_BEHAVIOR | 311 | 0.536000 | 0.745981 | 0.085946 | 0.148043 |
| retrieval_practice | LF_GEMMA4_vs_LF_BEHAVIOR | 311 | 0.556000 | 0.713826 | -0.014480 | -0.098432 |

## Aggregators

| Action | Aggregator | Seed policy | Seeds used | Same-seed max\|Δp\| | Cross-seed max\|Δp\| | Estimated LF reliability parameters |
|---|---|---|---|---:|---:|---|
| assessment_recovery | `SNORKEL` | `average_three_seeds` | `[42, 1201, 2026]` | 0.000000 | 0.152439 | `LF_BEHAVIOR=0.878958, LF_GEMINI35=0.853530, LF_GEMMA4=0.876898` |
| re_engagement | `SNORKEL` | `average_three_seeds` | `[42, 1201, 2026]` | 0.000000 | 0.181502 | `LF_BEHAVIOR=0.682006, LF_GEMINI35=0.663400, LF_GEMMA4=0.683448` |
| study_planning | `SNORKEL` | `average_three_seeds` | `[42, 1201, 2026]` | 0.000000 | 0.320283 | `LF_BEHAVIOR=0.379587, LF_GEMINI35=0.551305, LF_GEMMA4=0.509014` |
| progress_monitoring | `TWO_SOURCE_CONSENSUS` | `not_applicable_two_source_consensus` | `[]` | 0.000000 | 0.000000 | `LF_GEMINI31=NA, LF_GEMINI35=NA` |
| retrieval_practice | `SNORKEL` | `average_three_seeds` | `[42, 1201, 2026]` | 0.000000 | 0.130764 | `LF_BEHAVIOR=0.485682, LF_GEMINI35=0.502821, LF_GEMMA4=0.500200` |

LabelModel fitting is meaningfully stochastic on at least one action; those actions average the three project seeds.

## Silver labels

- Total rows: `2500`.
- VALID: `1641`.
- NO_WEAK_EVIDENCE: `548`.
- REVIEW: `311`.

All-abstain case-actions keep `silver_status=NO_WEAK_EVIDENCE` and do not receive class-0 probabilities.
Feasibility remains a separate field; INFEASIBLE is never converted into relevance class 0.

## Per-action quality

| Action | LFs | Usable | All-abstain | Aggregator | Mean E[R] | Mean conf. | Median conf. | Mean entropy | vs Majority | Status |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| assessment_recovery | 3 | 141 | 359 | `SNORKEL` | 2.958876 | 0.968901 | 0.997022 | 0.089770 | 0.929078 | `PASS` |
| re_engagement | 3 | 500 | 0 | `SNORKEL` | 1.392011 | 0.839231 | 0.913010 | 0.386920 | 0.951435 | `PASS` |
| study_planning | 3 | 500 | 0 | `SNORKEL` | 1.418218 | 0.761669 | 0.808140 | 0.652368 | 0.849711 | `PASS` |
| progress_monitoring | 2 | 500 | 0 | `TWO_SOURCE_CONSENSUS` | 1.504000 | 0.781000 | 1.000000 | 0.303598 | 1.000000 | `PASS_WITH_WARNING` |
| retrieval_practice | 3 | 311 | 189 | `SNORKEL` | 1.486364 | 0.647645 | 0.623623 | 0.845397 | 0.967593 | `REVIEW` |

### Class distributions on aggregated rows

| Action | Aggregator hard labels | Majority hard labels |
|---|---|---|
| assessment_recovery | `0=0, 1=0, 2=6, 3=135, ABSTAIN=0` | `0=0, 1=0, 2=16, 3=125, ABSTAIN=0` |
| re_engagement | `0=184, 1=68, 2=132, 3=116, ABSTAIN=0` | `0=173, 1=61, 2=103, 3=116, ABSTAIN=0` |
| study_planning | `0=17, 1=255, 2=221, 3=7, ABSTAIN=0` | `0=29, 1=187, 2=130, 3=0, ABSTAIN=0` |
| progress_monitoring | `0=105, 1=205, 2=137, 3=53, ABSTAIN=0` | `0=31, 1=121, 2=76, 3=53, ABSTAIN=0` |
| retrieval_practice | `0=0, 1=146, 2=165, 3=0, ABSTAIN=0` | `0=5, 1=105, 2=104, 3=2, ABSTAIN=0` |

## Collapse and stability flags

| Action | Flags | Mode share | E[R] std | Mean confidence |
|---|---|---:|---:|---:|
| assessment_recovery | `hard_label_collapse,unstable_across_seeds` | 0.957447 | 0.120655 | 0.968901 |
| re_engagement | `unstable_across_seeds` | 0.368000 | 1.130935 | 0.839231 |
| study_planning | `unstable_across_seeds` | 0.510000 | 0.502366 | 0.761669 |
| progress_monitoring | `none` | 0.410000 | 0.857312 | 0.781000 |
| retrieval_practice | `unstable_across_seeds` | 0.530547 | 0.253738 | 0.647645 |

## A4 Progress Monitoring warning

A4 has two effective sources and both are Gemini-family models.
They are not treated as fully independent annotators.
Snorkel LabelModel 0.9.9 requires at least three labeling functions, so A4 cannot use Snorkel.
The documented fallback is TWO_SOURCE_CONSENSUS: one-hot on agreement, 0.5/0.5 on conflict.
- Aggregator used: `TWO_SOURCE_CONSENSUS`.
- Quality status: `PASS_WITH_WARNING`.
- Reasons: `correlated_gemini_family, two_source_consensus_fallback`.

## A5 Retrieval Practice warning

A5 has severe source disagreement and remains in REVIEW unless a strong upgrade is justified.
Estimated LF reliability parameters below are LabelModel quantities, not true accuracy.
- Aggregator used: `SNORKEL`.
- Quality status: `REVIEW`.
- Reasons: `high_source_conflict`.
- Estimated LF reliability parameters: `{'LF_BEHAVIOR': 0.485682175250171, 'LF_GEMINI35': 0.5028214006677733, 'LF_GEMMA4': 0.5002001017229579}`.
- Mean entropy: `0.845397`; mean confidence: `0.647645`.

## Leakage

- Panel B case overlap: `0`.
- FINAL stage: excluded.
- Prediction truth / future activity / future assessment / future unregistration: not loaded.

## Phase 8 note

These silver labels are the Phase 8 EBM targets. Phase 8 was not started here.
