# Feature availability and temporal leakage audit

The source of truth is `config/feature_availability.yaml`. The UCI dictionary documents attribute meaning and the period-grade order G1, G2, G3, but does not timestamp capture of questionnaire variables. This audit therefore does not infer availability from plausibility.

| Scenario | Strict allowlist | Excluded reason |
| --- | --- | --- |
| pre-assessment | none | All questionnaire timing is unknown; G1/G2 are future grades; G3 is target. |
| early-warning (after G1, before G2) | G1 | G1 is a documented first-period grade; G2 is future; all unclocked questionnaire fields are excluded. |
| late-stage (after G2, before G3) | G1, G2 | The two documented prior grades are available; G3 is target; unclocked questionnaire fields remain excluded. |

`absences`, `failures`, `studytime`, `schoolsup`, `famsup`, `paid`, `activities`, `higher`, and `internet` are all explicitly marked `unknown` at strict cutoffs because their collection/reference window is not demonstrated by the data dictionary. They are not silently promoted to baseline variables. `G1` is only allowed before G2; `G2` only after G2; `G3` and every `G3_*`/G3-derived feature are permanently forbidden.

The enforcement point is `src.evaluation.protocol.validate_scenario_features`; `src.model_selection.fit_fold_predict_proba` invokes it before preprocessing. `DataPreprocessor` also removes target and target-derived columns from feature matrices. The implementation rejects unsupported overrides rather than allowing an unlogged exception.
