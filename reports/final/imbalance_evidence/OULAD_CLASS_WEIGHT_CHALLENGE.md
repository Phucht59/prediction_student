# OULAD Class-Weight Challenge Evidence

## Status

- Phase: `OULAD_FAST_CLASS_WEIGHT_CHALLENGE`
- Status: `COMPLETE`
- Runs completed: `15/15`
- Protocol identity: `PASS`
- Data leakage check: `PASS`
- Protected frozen V1 changed: `FALSE`
- Final V2 promotion: `FALSE`
- Decision: keep the current OULAD final authority with standard BCE.

## Controlled comparison

The comparison uses the authority-equivalent OULAD H1 Tabular Residual CNN–BiLSTM protocol. The architecture is unchanged; class weighting changes only the training loss policy.

| Metric | FIXED_NONE | FIXED_CLASS_WEIGHT | Delta |
|---|---:|---:|---:|
| Macro-F1 | 0.894245 | 0.885983 | -0.008262 |
| Risk Recall | 0.784267 | 0.826008 | +0.041741 |
| PR-AUC | 0.934926 | 0.934896 | -0.000030 |

## Interpretation

Class weighting increases Risk Recall from `0.784267` to `0.826008`, so the model identifies more at-risk students. However, the primary Macro-F1 decreases from `0.894245` to `0.885983`, a drop of `0.008262`. PR-AUC is effectively unchanged.

Under the frozen evaluation objective, this trade-off is not sufficient to replace the current final model. Therefore:

- `CLASS_WEIGHT_WINS=FALSE`
- `PROMOTE_FINAL_V2=FALSE`
- OULAD final architecture remains `H1_TABULAR_RESIDUAL_EXPERT`.
- OULAD final imbalance policy remains `UNIFORM_NONE_STANDARD_BCE`.
- Frozen authority headline Macro-F1 remains `0.8940709888551659`.

## Claim boundary

This evidence supports the statement that class weighting improves risk recall but reduces balanced overall classification performance under the controlled final OULAD protocol. It does not establish that every imbalance-handling method is inferior; SMOTE/ADASYN were not promoted by this challenge.

## Provenance

The controlled NONE reproduction used for the comparison achieved Macro-F1 `0.8942454181505014`. The frozen official authority remains Macro-F1 `0.8940709888551659`; the small reproduction difference was previously audited as protocol-equivalent CUDA nondeterminism. The official frozen model remains unchanged.
