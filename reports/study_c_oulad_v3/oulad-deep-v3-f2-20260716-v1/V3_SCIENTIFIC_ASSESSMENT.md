# OULAD Deep V3 Scientific Assessment

Study status: **exploratory post-V2 temporal representation study**. Future benchmark: **NOT EXECUTED**.

## Answers to the registered questions

- Pooling gate: **FAIL**. P0 − H3CF = +0.000646 Macro-F1; the +0.002 gate was not reached.
- Inner-selected pooling: `2/3` folds masked attention and `1/3` folds last/mean/max.
- Dynamic channels: D0 − P0 = +0.000601; below the +0.003 registered dynamics gate.
- Sequence ordering: D0 − A1 = +0.001112. This does **not** establish incremental temporal ordering value.
- Matched ML: D0 − MLD = +0.001370; dynamics gate remains FAIL.
- Three-seed ensemble: ENS − D0 mean-single-seed = +0.002991. ENS Macro-F1 = 0.830322.
- Strongest frozen/matched comparator is V3-A0F at 0.827070; ENS delta = +0.003252, below the +0.005 superiority threshold.
- Overall superiority: **FAIL**. Operational superiority: **FAIL**. Competitive gate: **PASS**.
- H4/SSL: not opened. D0 − P0 did not reach +0.002 for H4; D0 was neither stable enough nor below A1 in mean to justify SSL under the registered rule.
- No class collapse, probability failure, checkpoint mismatch, student overlap, outer-label tuning, or future access was found.

## Verdict

**PRACTICAL_TIE** for the temporal CNN–BiLSTM family. The ensemble has the highest exploratory point estimate, but neither pooling nor temporal-dynamics incremental-value gates passed, and the superiority margin was not reached.

## Thesis claims

Allowed: V3 improved engineering and ensemble stability sufficiently to be competitive in exploratory F2 evidence; aggregate summaries explain most signal; temporal ordering value was not established; the three-seed ensemble had the highest point estimate but remained within the registered practical margin.

Prohibited: confirmatory superiority, untouched/external validation, independent future confirmation, or a claim that CNN–BiLSTM sequence ordering beat matched aggregate controls.
