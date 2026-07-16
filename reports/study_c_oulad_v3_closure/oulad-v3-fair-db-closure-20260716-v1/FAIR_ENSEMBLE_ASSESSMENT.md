# Fair Ensemble Assessment

This closure permanently separates single-seed metrics, mean-of-seed metrics, and record-aligned probability ensembles.

| Candidate Id | Macro F1 | At Risk Precision | At Risk Recall | At Risk F1 | Pr Auc | Operational Recall | Brier | Nll | Ece | Worst Eligible Module Macro F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| V3-A0F-ENS | 0.828672 | 0.838464 | 0.738967 | 0.785578 | 0.891824 | 0.820530 | 0.115519 | 0.367835 | 0.025173 | 0.760466 |
| V3-A1-ENS | 0.826359 | 0.829228 | 0.742890 | 0.783688 | 0.890154 | 0.819712 | 0.116989 | 0.372261 | 0.029866 | 0.731496 |
| V3-D0-ENS | 0.831126 | 0.840607 | 0.743053 | 0.788825 | 0.892692 | 0.817751 | 0.114468 | 0.363406 | 0.019528 | 0.756728 |
| V3-H2TF-ENS | 0.825465 | 0.839955 | 0.729160 | 0.780646 | 0.891217 | 0.818078 | 0.115049 | 0.364657 | 0.018284 | 0.768521 |
| V3-H3CF-ENS | 0.827344 | 0.845803 | 0.728016 | 0.782502 | 0.891761 | 0.815299 | 0.115557 | 0.367368 | 0.025760 | 0.763893 |
| V3-MLD | 0.825961 | 0.835688 | 0.734881 | 0.782049 | 0.889263 | 0.817424 | 0.115258 | 0.365903 | 0.011056 | 0.741051 |
| V3-MLF | 0.825718 | 0.828587 | 0.741909 | 0.782856 | 0.887451 | 0.741909 | 0.120314 | 0.380735 | 0.046608 | 0.759556 |
| V3-P0-ENS | 0.829173 | 0.819525 | 0.761523 | 0.789460 | 0.892292 | 0.823635 | 0.114993 | 0.364469 | 0.021436 | 0.768089 |

- Corrected verdict: **PRACTICAL_TIE**.
- D0-ENS minus strongest fair comparator (V3-A0F-ENS): `0.002454441` Macro-F1; registered superiority margin: `0.005`.
- Threshold reconstruction: 54 frozen-config replay jobs; outer labels used: `false`.
- Future benchmark: `NOT EXECUTED`.
- The earlier mixed-contract bootstrap is preserved as `historical_v3_mixed_contract_result` and is ineligible for this verdict.
