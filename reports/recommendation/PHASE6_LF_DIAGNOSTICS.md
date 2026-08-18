# Phase 6 LF diagnostics

Phase 6 only. No API call, Snorkel execution, silver-label generation, EBM training, Panel-B use, or manual reliability weighting was performed.

## Final action/source contract

| Action | Effective sources | Quality status |
|---|---|---|
| assessment_recovery | `LF_GEMINI35, LF_GEMMA4, LF_BEHAVIOR` | `PASS` |
| re_engagement | `LF_GEMINI35, LF_GEMMA4, LF_BEHAVIOR` | `PASS` |
| study_planning | `LF_GEMINI35, LF_GEMMA4, LF_BEHAVIOR` | `PASS` |
| progress_monitoring | `LF_GEMINI35, LF_GEMINI31` | `PASS_WITH_CORRELATED_FAMILY_WARNING` |
| retrieval_practice | `LF_GEMINI35, LF_GEMMA4, LF_BEHAVIOR` | `REVIEW_HIGH_CONFLICT` |

## Source diagnostics

| Action | Source | Cases | Coverage | ABSTAIN rate | Distribution |
|---|---|---:|---:|---:|---|
| assessment_recovery | LF_GEMINI35 | 500 | 0.282000 | 0.718000 | `{'0': 0, '1': 0, '2': 17, '3': 124, 'ABSTAIN': 359}` |
| assessment_recovery | LF_GEMMA4 | 500 | 0.282000 | 0.718000 | `{'0': 0, '1': 0, '2': 19, '3': 122, 'ABSTAIN': 359}` |
| assessment_recovery | LF_BEHAVIOR | 500 | 0.282000 | 0.718000 | `{'0': 0, '1': 0, '2': 20, '3': 121, 'ABSTAIN': 359}` |
| re_engagement | LF_GEMINI35 | 500 | 1.000000 | 0.000000 | `{'0': 192, '1': 98, '2': 122, '3': 88, 'ABSTAIN': 0}` |
| re_engagement | LF_GEMMA4 | 500 | 1.000000 | 0.000000 | `{'0': 179, '1': 53, '2': 120, '3': 148, 'ABSTAIN': 0}` |
| re_engagement | LF_BEHAVIOR | 500 | 1.000000 | 0.000000 | `{'0': 168, '1': 109, '2': 117, '3': 106, 'ABSTAIN': 0}` |
| study_planning | LF_GEMINI35 | 500 | 1.000000 | 0.000000 | `{'0': 53, '1': 190, '2': 203, '3': 54, 'ABSTAIN': 0}` |
| study_planning | LF_GEMMA4 | 500 | 1.000000 | 0.000000 | `{'0': 25, '1': 282, '2': 193, '3': 0, 'ABSTAIN': 0}` |
| study_planning | LF_BEHAVIOR | 500 | 0.492000 | 0.508000 | `{'0': 68, '1': 71, '2': 56, '3': 51, 'ABSTAIN': 254}` |
| progress_monitoring | LF_GEMINI35 | 500 | 1.000000 | 0.000000 | `{'0': 54, '1': 178, '2': 212, '3': 56, 'ABSTAIN': 0}` |
| progress_monitoring | LF_GEMINI31 | 500 | 1.000000 | 0.000000 | `{'0': 82, '1': 214, '2': 92, '3': 112, 'ABSTAIN': 0}` |
| progress_monitoring | LF_BEHAVIOR | 500 | 0.000000 | 1.000000 | `{'0': 0, '1': 0, '2': 0, '3': 0, 'ABSTAIN': 500}` |
| retrieval_practice | LF_GEMINI35 | 500 | 0.622000 | 0.378000 | `{'0': 14, '1': 117, '2': 176, '3': 4, 'ABSTAIN': 189}` |
| retrieval_practice | LF_GEMMA4 | 500 | 0.622000 | 0.378000 | `{'0': 24, '1': 176, '2': 111, '3': 0, 'ABSTAIN': 189}` |
| retrieval_practice | LF_BEHAVIOR | 500 | 0.622000 | 0.378000 | `{'0': 78, '1': 78, '2': 77, '3': 78, 'ABSTAIN': 189}` |

## Pairwise agreement

Agreement is between weak-label sources, not human annotators. Gemini 3.5 and Gemini 3.1 are distinct models in the same Gemini family and are not treated as fully independent annotators.

| Action | Pair | Overlap | Exact | Linear kappa | Quadratic kappa |
|---|---|---:|---:|---:|---:|
| assessment_recovery | LF_GEMINI35_vs_LF_GEMMA4 | 500 | 478/500 (0.956000) | 0.299774 | 0.299774 |
| assessment_recovery | LF_GEMINI35_vs_LF_BEHAVIOR | 500 | 479/500 (0.958000) | 0.347366 | 0.347366 |
| assessment_recovery | LF_GEMMA4_vs_LF_BEHAVIOR | 500 | 487/500 (0.974000) | 0.613210 | 0.613210 |
| re_engagement | LF_GEMINI35_vs_LF_GEMMA4 | 500 | 294/500 (0.588000) | 0.631515 | 0.769998 |
| re_engagement | LF_GEMINI35_vs_LF_BEHAVIOR | 500 | 293/500 (0.586000) | 0.616322 | 0.755880 |
| re_engagement | LF_GEMMA4_vs_LF_BEHAVIOR | 500 | 304/500 (0.608000) | 0.641740 | 0.772001 |
| study_planning | LF_GEMINI35_vs_LF_GEMMA4 | 500 | 282/500 (0.564000) | 0.418445 | 0.566416 |
| study_planning | LF_GEMINI35_vs_LF_BEHAVIOR | 500 | 71/500 (0.142000) | 0.156271 | 0.301978 |
| study_planning | LF_GEMMA4_vs_LF_BEHAVIOR | 500 | 65/500 (0.130000) | 0.029415 | 0.100889 |
| progress_monitoring | LF_GEMINI35_vs_LF_GEMINI31 | 500 | 281/500 (0.562000) | 0.556862 | 0.713714 |
| retrieval_practice | LF_GEMINI35_vs_LF_GEMMA4 | 500 | 319/500 (0.638000) | 0.016558 | 0.044278 |
| retrieval_practice | LF_GEMINI35_vs_LF_BEHAVIOR | 500 | 268/500 (0.536000) | 0.085946 | 0.148043 |
| retrieval_practice | LF_GEMMA4_vs_LF_BEHAVIOR | 500 | 278/500 (0.556000) | -0.014480 | -0.098432 |

## Required findings

- A1 Gemini35 vs Gemma4: exact ≈ 0.956; quadratic weighted kappa ≈ 0.300. The high exact agreement is prevalence-sensitive because both sources are dominated by the same ABSTAIN/limited numeric class pattern.
- A2 Gemini35 vs Gemma4: exact ≈ 0.588; quadratic weighted kappa ≈ 0.770.
- A3 Gemini35 vs Gemma4: exact ≈ 0.564; quadratic weighted kappa ≈ 0.566.
- A4 Gemini35 vs Gemini31: exact ≈ 0.562; linear weighted kappa ≈ 0.557; quadratic weighted kappa ≈ 0.714. Both LLM sources are non-degenerate; Behavioral A4 is ABSTAIN 500/500 and excluded from the effective list.
- A5 remains REVIEW: Gemini35/Gemma4 quadratic kappa ≈ 0.044; Gemini35/Behavior ≈ 0.148; Gemma4/Behavior ≈ -0.098.
- Historical Gemma4 A4 Progress Monitoring is `REJECTED_DEGENERATE` and excluded from Phase 7.

## A5 confusion matrices

### LF_GEMINI35_vs_LF_GEMMA4

Rows = left source; columns = right source.

| Label | 0 | 1 | 2 | 3 | ABSTAIN |
|---|---:|---:|---:|---:|---:|
| 0 | 2 | 10 | 2 | 0 | 0 |
| 1 | 8 | 65 | 44 | 0 | 0 |
| 2 | 14 | 99 | 63 | 0 | 0 |
| 3 | 0 | 2 | 2 | 0 | 0 |
| ABSTAIN | 0 | 0 | 0 | 0 | 189 |

### LF_GEMINI35_vs_LF_BEHAVIOR

Rows = left source; columns = right source.

| Label | 0 | 1 | 2 | 3 | ABSTAIN |
|---|---:|---:|---:|---:|---:|
| 0 | 0 | 2 | 5 | 7 | 0 |
| 1 | 49 | 31 | 24 | 13 | 0 |
| 2 | 29 | 45 | 46 | 56 | 0 |
| 3 | 0 | 0 | 2 | 2 | 0 |
| ABSTAIN | 0 | 0 | 0 | 0 | 189 |

### LF_GEMMA4_vs_LF_BEHAVIOR

Rows = left source; columns = right source.

| Label | 0 | 1 | 2 | 3 | ABSTAIN |
|---|---:|---:|---:|---:|---:|
| 0 | 3 | 5 | 3 | 13 | 0 |
| 1 | 40 | 53 | 41 | 42 | 0 |
| 2 | 35 | 20 | 33 | 23 | 0 |
| 3 | 0 | 0 | 0 | 0 | 0 |
| ABSTAIN | 0 | 0 | 0 | 0 | 189 |

## Quality interpretation

- A1/A2/A3: `PASS` for source comparison diagnostics.
- A4: `PASS_WITH_CORRELATED_FAMILY_WARNING`; use only Gemini35 and Gemini31 as effective LLM sources.
- A5: `REVIEW_HIGH_CONFLICT`; retain in the five-action architecture and carry the warning forward.
- Historical repeatability/prompt-v1b artifacts are `ROBUSTNESS_ONLY`, not LF columns.
