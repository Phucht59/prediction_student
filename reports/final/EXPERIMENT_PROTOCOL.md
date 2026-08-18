# Experiment Protocol

All official model selection used inner validation only. Final metrics are
complete outer out-of-fold probability ensembles; no best seed or best fold is
reported. Student-Mat and Student-Por use five outer folds and fixed seeds 42,
1201, 2026, 3407 and 7319. OULAD uses three outer folds with the same seeds and
fixed thresholds 0.455, 0.495 and 0.500.

The release performs no training, tuning or outer evaluation. Future OULAD is
`LOCKED_NOT_EXECUTED`.
