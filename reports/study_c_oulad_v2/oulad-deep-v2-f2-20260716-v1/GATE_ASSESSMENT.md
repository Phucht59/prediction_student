# OULAD Deep V2 — F2 gate assessment

- Gate: **FAIL**
- Tuning thật giúp H2 (H2T − H2F): +0.0096
- H3C − H2T: +0.0016
- Temporal incremental value (H3C − A0): -0.0010
- Static-context contribution (H2T − T0): +0.0010
- H3C seed wins over H2T: 2/3
- H2P parameter-matched control: **NOT OPENED — gate failed**
- Best mean inner-trial positive-weight policy: `sqrt_balanced` (descriptive; configs remained outer-specific)
- Strongest mandatory candidate by Macro-F1: `V2-A0`
- Strongest constraint-eligible operational endpoint: `V2-A0`
- Overall superiority over frozen ML: **NO**
- Operational superiority: **YES — V2-A0 only**
- A0 − MLF Macro-F1: +0.0014; A0 − MLF constrained Recall: +0.0720
- A0 paired student-bootstrap lower bounds for constrained Recall are positive for all three seeds: **YES**
- CNN–BiLSTM H3C verdict: **PRACTICAL TIE WITH ML; F2 GATE FAIL**
- Stable across seeds/modules: H3C seed SD guard PASS; worst-module guard PASS; improvement-size guard FAIL
- Future benchmark used for selection: **NO**

The F2 gate failed because H3C − H2T did not reach +0.005. The operational-superiority label is limited to aggregate-only neural control A0: its frozen inner operating points met outer Precision >= 0.75 in all seeds and improved Recall with positive paired student-bootstrap intervals. It is not evidence that the CNN–BiLSTM temporal representation beat ML. Conditional candidates, ensemble and calibration were not opened. No negative result was overwritten.
