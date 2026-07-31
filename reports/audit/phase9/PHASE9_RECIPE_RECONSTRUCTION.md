# Phase 9 recipe reconstruction

| Candidate | Macro-F1 | PR-AUC | ROC-AUC | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| A0 Phase 7 H1 control | 0.796297 | 0.857463 | 0.870811 | 0.413389 | 0.133071 | 0.008758 |
| A1 valid H0 recipe | 0.795184 | 0.856091 | 0.870852 | 0.414389 | 0.133410 | 0.014616 |

A1 preserves the H1 topology and 160,492 parameters. It uses H0's valid
score-free compact feature recipe, masked train-only sequence normalization,
legal score-free temporal pretraining, standard BCE with 0.15 auxiliary
weights, and fixed eight-epoch training. It did not pass the materiality gate.
