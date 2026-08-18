# Phase 7 — OULAD H1 Final Endpoint

## Outcome

Phase 7 completed the preregistered single-cutoff OULAD endpoint study for
`H1_TABULAR_RESIDUAL_EXPERT`. The endpoint is
`F2_MIDDLE_OFFICIAL_SINGLE_CUTOFF` (50% cutoff), not the 75% early-warning
stage and not the mean of the four early-warning stages.

The bounded INNER study selected the preregistered CONTROL configuration:
tuning changed mean Macro-F1 from 0.795913 to 0.795840
(`delta = -0.000073`). The immutable endpoint candidate was committed before
outer access in commit `c10826271b27f222955917c2ad3b169644efa40d`.

## Final endpoint result

| Model | Macro-F1 | PR-AUC | ROC-AUC | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| H1 tabular residual | 0.798400 | 0.863039 | 0.876142 | 0.406292 | 0.130702 | 0.011988 |
| H0 historical hybrid | 0.828084 | 0.893355 | 0.908156 | 0.358778 | 0.113355 | 0.009463 |
| MLP | 0.828286 | 0.891710 | 0.907336 | 0.362018 | 0.114346 | 0.007746 |

H1 did not reach 0.83. Its Macro-F1 delta was `-0.029684` versus H0 and
`-0.029886` versus MLP. Both paired student-level bootstrap intervals exclude
zero. Phase 7 classification is:

`D — H1 DOES NOT IMPROVE H0`

No post-outer tuning was performed. The frozen early-warning evidence remains
unchanged and is a separate evidence track.

## Main thesis result table

| Dataset | Model | Accuracy | Macro precision | Macro recall | Macro-F1 | PR-AUC | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| Student-Mat | CNN-BiLSTM | 0.891139 | 0.903140 | 0.902089 | 0.901460 | 0.944184 | 0.967939 |
| Student-Por | CNN-BiLSTM | 0.889060 | 0.857284 | 0.867576 | 0.862259 | 0.914679 | 0.962791 |
| OULAD | H1 CNN-BiLSTM + Tabular Residual | 0.811809 | 0.810072 | 0.792186 | 0.798400 | 0.863039 | 0.876142 |
