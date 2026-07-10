# Master context for thesis writing

## Title and research problem

**Thesis title:** *Xay dung mo hinh hoc ket hop de du doan thanh tich hoc tap sinh vien.*

The implemented study is a three-class prediction task on UCI Student
Performance `student-mat`: predict final mathematics performance from prior
academic and contextual variables. The scientific objective is to evaluate a
CNN-BiLSTM research architecture fairly against transparent baselines and to
produce advisory, human-reviewed learning recommendations.

## Objectives and research questions

1. Build a reproducible three-class prediction pipeline with leakage controls.
2. Evaluate CNN-BiLSTM against simple and tabular baselines under a fixed
   protocol.
3. Compare late-stage, early-warning and pre-assessment information scenarios.
4. Store data lineage, runs, predictions and recommendations in PostgreSQL.
5. Provide a deterministic rule-based advisory layer, not an automated decision
   system.

Research questions: (RQ1) How accurately can the classes be predicted under
each information scenario? (RQ2) Does CNN-BiLSTM add value beyond simple G2 and
tabular baselines? (RQ3) Are final predictions, recommendation outputs and
evidence reproducible from frozen artifacts?

## Dataset and target

The dataset is UCI Student Performance `student-mat`, 395 Portuguese
secondary-school students in mathematics. G1, G2 and G3 are first, second and
final assessment grades. The target is derived from G3: Low <=9, Medium 10-14,
High >=15. Distribution: 130/192/73. The deterministic locked split contains
316 development records and 79 test records (26/38/15 by class).

## Scenarios

| Scenario | Inputs | Intended interpretation |
| --- | --- | --- |
| late-stage | G1 and G2 available | prediction near the end of the course |
| early-warning | G2 excluded | earlier intervention setting |
| pre-assessment | G1 and G2 excluded | before recorded academic assessments |

These scenarios expose different information and must not be ranked as if they
were directly comparable tasks.

## Final model and protocol

The deployable research model is a single-seed (42) CNN-BiLSTM with G1/G2 as a
two-step sequence, 13,059 trainable parameters, no resampling and no class
weight. Selection used 5 outer stratified folds, 3 inner folds, 30 Optuna trials
per inner search and mean inner-CV Macro-F1. The locked test was not used for
model selection, calibration fitting or threshold tuning. The frozen policy is
argmax with no calibration.

## Final results and scientific conclusion

Nested outer Macro-F1 is 0.8781 +/- 0.0448. Locked-test accuracy is 0.9114 and
Macro-F1 is 0.9262. This is high absolute performance, but it does not establish
added value over the G2 threshold rule (OOF Macro-F1 0.8988; locked Macro-F1
0.9365) or HGB on the locked test (0.9463). The correct conclusion is that the
CNN-BiLSTM pipeline is technically feasible and reproducible, but added value
over simple late-stage signals is not demonstrated.

## Recommendation and PostgreSQL

`student_mat_rule_policy_v3` is a deterministic, rule-based advisory policy.
It uses model output and permitted input features, supplies risk factors,
actions, reasons, priority, confidence, disclaimer and human-review framing.
Its 79 outputs are schema-valid and contradiction-free; expert review is not
collected. PostgreSQL is the canonical source architecture. Live migration 003
for separate target storage is pending administrator execution; do not claim
live target-table verification is complete.

## Reproducibility and limits

Frozen evidence is `final-a2945d79-9845-4979-b148-159f4853eca3`; selection run
is `nested-full-20260710`. Config SHA-256 is
`cda38460197627ac1d71e764f61d784e4c03cf6f86775339d38787c6890678ad`.
The prior reproducibility run has exact prediction checksum match. Tests: 87
passed, 5 PostgreSQL integration tests pending credentials.

## Claims forbidden in the thesis

- Do not call this Vietnamese university-student data; it is Portuguese
  secondary-school mathematics data.
- Do not describe G1/G2 as a long multi-semester time series.
- Do not claim CNN-BiLSTM beats G2 or HGB baselines.
- Do not call the recommender a machine-learning model or claim educational
  effectiveness/expert review.
- Do not call the 11-seed ensemble the final deployable model.
- Do not claim calibration was used in the frozen final configuration.
- Do not use locked-test results as model-selection evidence.
- Do not claim PostgreSQL migration 003 or live DB-first verification is done.
