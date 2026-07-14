# Post-reconciliation decision

The official scientific value for CNN–BiLSTM under the current protocol is the valid V2 result: `0.7984 ± 0.0526` Macro-F1, using the pre-registered five-seed fold estimator. The historical `0.8781 ± 0.0448` remains a traceable historical reference, but is not eligible for new-model ranking because it uses a different training/search-space estimator and does not have complete checkpoint/epoch provenance.

Benchmark V2 remains valid; no rerun is required. The historical result was not caused by different outer folds and the audit found no historical outer-validation leakage through early stopping/checkpoint control. The large difference is a combined protocol/config/seed-estimator effect.

Future research must use Protocol V2. CNN–BiLSTM should be retained only as a research comparator, not the primary model. There is enough evidence to plan—but not implement in this phase—a tightly controlled ordinal tabular MLP ablation. It must be benchmarked against G2 rule and small MLP under the same V2 folds/seeds; legacy-79 remains excluded.
