# Final scientific evaluation results

## Scope and method

This report evaluates technical correctness of `recommend_hybrid`; it does not measure educational effectiveness. The deterministic sample contains 260 pseudonymous records selected by stable record key without reading outcome labels: 20 records for each UCI dataset-stage, 20 for each canonical OULAD anchor, and 10 for each inter-stage request at 25, 36, 63 and 76. Frozen canonical hybrid OOF/seed predictions and raw pre-cutoff evidence are used without retraining or threshold adjustment. Intervention coverage excludes 20 FINAL_EVALUATION records, giving denominator 240.

## Overall results

| Metric | Result |
|---|---:|
| Actionable coverage | 92.08% (221/240) |
| Full recommendation rate | 51.92% (all 260 records) |
| Partial recommendation rate | 33.08% (all 260 records) |
| Abstention rate | 7.92% (19/240 intervention records) |
| Evaluation-only rate | 7.69% (20/260) |
| Mean / median actions per intervention record | 2.58 / 3.00 |
| Mean / median workload | 135.17 / 140 minutes |
| Evidence support / explanation lineage | 100% / 100% |
| Unique actions used / top-action share | 14 / 27.90% |
| Action-set diversity | 14.93% |
| Deterministic replay / plan hash match | 100% / 100% |

## Dataset comparison

| Dataset | Records | Intervention denominator | Coverage | Abstention | Mean actions | Mean workload | Evidence support | Unique actions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| student_mat | 60 | 60 | 95.00% | 5.00% | 2.37 | 108.67 | 100% | 9 |
| student_por | 60 | 60 | 95.00% | 5.00% | 2.63 | 131.75 | 100% | 8 |
| OULAD | 140 | 120 | 89.17% | 10.83% | 2.67 | 150.13 | 100% | 8 |

UCI and OULAD workloads are not interpreted as directly comparable educational quantities: UCI plans use assessment-business periods, while OULAD uses remaining-course periods. Detailed stage denominators, status rates and action distributions are in `DATASET_STAGE_RESULTS.csv`. OULAD prediction age averages 1.43 percentage points and reaches 13 points in inter-stage routing; every anchor remains in the past.

## Safety, constraints and explanations

Post-cutoff, future-anchor, final-intervention, G3, sensitive-feature, missing-lineage, cross-dataset and invalid model/dataset mapping violations are all 0. Action-cap, workload, duplicate, prerequisite, contraindication, unsupported-action, invalid-period and course-end violations are all 0. Unsupported-reason and missing-evidence-misuse rates are 0; reason/action consistency is 100%.

## Scenario, monotonicity and robustness

Phase 3 controlled-scenario and metamorphic pass rates are 100%. Monotonicity, resolution-responsiveness, non-material-instability and uncertainty-safety violation counts are 0. Seven controlled robustness checks cover assessment progress, inactivity, absences, study time, uncertainty, requested cutoff and remaining course time; all pass. These are bounded software/policy tests, not intervention experiments.

## Ablation

| Variant | Coverage | Abstention | Evidence support | Unsupported actions | Explanation completeness | Diversity | Constraint violations |
|---|---:|---:|---:|---:|---:|---:|---:|
| A risk class only | 100% | 0% | 0% | 100% | 0% | 3 | 7.69% |
| B risk probability only | 100% | 0% | 0% | 100% | 0% | 3 | 7.69% |
| C risk + evidence | 100% | 0% | 100% | 0% | 100% | 14 | 0% |
| D official full policy | 92.08% | 7.92% | 100% | 0% | 100% | 14 | 0% |

A/B illustrate that high coverage alone can be unsupported and unsafe at final stage. C neutralizes uncertainty while retaining minimum temporal routing needed to define evidence. D is the only release policy; it abstains under locked uncertainty controls. No expected pattern was forced and no outcome label was used.

## Bootstrap confidence intervals

Student-level percentile bootstrap (1,000 replicates, seed 20260801) gives: actionable coverage 92.08% [87.81%, 95.97%], abstention 7.92% [4.03%, 12.19%], evidence support 100% [100%, 100%], mean action count 2.58 [2.39, 2.77], mean workload 135.17 [118.91, 151.48], and top-action share 27.90% [23.81%, 32.27%]. These intervals describe this technical sample, not population educational benefit.

## Concentration finding and conclusion

`PROGRESS_MONITORING` occurs in 78.28% of actionable plans and accounts for 27.90% of actions. Every occurrence is evidence-supported and the selector does not insert it by default; concentration traces to the broad LOW-severity monitoring eligibility rule. The locked policy is not changed in Phase 5. The release supports claims of technical consistency, cutoff safety, evidence linkage and reproducibility only—not optimality, grade improvement, expert validation, user acceptance or causal effect.
