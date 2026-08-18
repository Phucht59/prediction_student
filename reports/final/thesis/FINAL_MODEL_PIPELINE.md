# Final H1 model pipeline

```text
Raw OULAD data
        ↓
stage/cutoff-safe feature construction
        ↓
train-only preprocessing
        ↓
temporal sequence + aggregate/static representations
        ↓
H1 CNN-BiLSTM + tabular residual expert
        ↓
probability
        ↓
threshold selected from inner OOF only
        ↓
at-risk / not-at-risk prediction
```

## Training

Preprocessors fit only on the training partition. Epoch/checkpoint and training
configuration are selected without outer labels. Future events are masked.

## Validation

Inner grouped folds select the research threshold by Macro-F1. Operational
recall-oriented thresholds remain separate and do not select the scientific
model.

## Test

The frozen estimator and inner-selected threshold are applied once to the
outer partition. Test labels calculate metrics only; they do not modify the
model, threshold or feature schema.

Early-warning and endpoint evidence remain separate. No Phase 10 training,
threshold fitting or outer evaluation is performed.
