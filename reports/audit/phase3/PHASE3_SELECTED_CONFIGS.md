# Phase 3 — Selected Configurations

### Fold 0

- Trial: 23
- Epoch: 6 from [7, 5]
- LR: 0.0011484568
- Weight decay: 1.3278646e-06
- Dropout: 0.236840
- Batch: 256
- Loss: standard_bce
- Positive-weight strategy: not_applicable
- Survival/outcome weights: 0.0 / 0.2

### Fold 1

- Trial: 12
- Epoch: 6 from [7, 5]
- LR: 0.00093428899
- Weight decay: 2.3336498e-05
- Dropout: 0.249703
- Batch: 256
- Loss: standard_bce
- Positive-weight strategy: not_applicable
- Survival/outcome weights: 0.15 / 0.0

### Fold 2

- Trial: 15
- Epoch: 5 from [3, 7]
- LR: 0.0014274905
- Weight decay: 1.073162e-07
- Dropout: 0.147882
- Batch: 128
- Loss: standard_bce
- Positive-weight strategy: not_applicable
- Survival/outcome weights: 0.0 / 0.15


All selected configurations share architecture hash
`305cbeb49ae04c65a4de81e40b12fbf72cee2b7a6171136cee3a19047a407eff` and 150,202 parameters. All use standard
BCE; this is an observed search association, not authorization to change the
official frozen model.
