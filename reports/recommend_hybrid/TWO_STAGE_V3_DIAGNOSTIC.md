# Two-stage V3 diagnostic

## Overall decomposition

- Groups: 29043
- Positive groups: 9304
- Positive prevalence: 0.3204
- Issued groups: 13462
- Issued positive groups: 4825
- False issues: 8637
- Correct issued actions: 3650
- Stage A precision: 0.3584
- Stage A recall: 0.5186
- Stage B conditional Precision@1: 0.7565
- End-to-end Precision@1: 0.2711
- Perfect-ranker ceiling with the same gate: 0.3584

## Scalar recommendability signals

- hybrid_score: ROC-AUC=0.4569, AP=0.2836, best precision at recall>=0.50=0.3204 (recall=1.0000)
- top_margin: ROC-AUC=0.5377, AP=0.3534, best precision at recall>=0.50=0.3619 (recall=0.5026)
- risk_reduction: ROC-AUC=0.4307, AP=0.2773, best precision at recall>=0.50=0.3219 (recall=0.9902)
- certainty: ROC-AUC=0.4155, AP=0.2633, best precision at recall>=0.50=0.3223 (recall=1.0000)
- evidence_strength: ROC-AUC=0.5047, AP=0.3279, best precision at recall>=0.50=0.3308 (recall=0.6172)

## Per stage

- EARLY_20: gate precision=0.2061, gate recall=0.4909, conditional P@1=0.5695, end-to-end P@1=0.1174
- EARLY_35: gate precision=0.3863, gate recall=0.4617, conditional P@1=0.8050, end-to-end P@1=0.3110
- MIDDLE_50: gate precision=0.4495, gate recall=0.5811, conditional P@1=0.7861, end-to-end P@1=0.3534

## Scientific interpretation

- Dominant failure: `RECOMMENDABILITY_GATE`
- End-to-end 80% supported: `False`
- Conditional 80% supported: `False`

Conditional metrics must not be reported as unconditional end-to-end recommendation accuracy.
