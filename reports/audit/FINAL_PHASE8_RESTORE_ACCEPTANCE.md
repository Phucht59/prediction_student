# Final Phase8 Restore Acceptance

## ACCEPTANCE

- status: `PASS_PREDICTION_RECOMMENDATION_REBUILD_REQUIRED`
- prediction final: `RECONSTRUCTED_FROM_FROZEN_PROTOCOL`
- recommendation final: `REBUILT_FROM_CORRECT_RECONSTRUCTED_PREDICTIONS; NEW_HELDOUT_REVALIDATION_REQUIRED`
- retraining performed: Hybrid (three fitted instances) and five recommendation EBMs

## PREDICTION

One public research architecture is active: `Hybrid` (`model_id=hybrid`). Its exact tree is:

```text
Static -> projector ----\
Aggregate -> projector --+-> F3 adaptive entropy gate -> masked softmax -> fusion
Temporal -> adapter -> CNN -/                                      -> binary head -> one logit -> sigmoid P(Risk)
Temporal -> adapter -> BiLSTM -/
```

UCI, OULAD Early, and OULAD FINAL-100 use the same `Hybrid` class with separate fitted dimensions/weights where required. No multiclass or legacy CNN/BiLSTM public model is active. Config and source equivalence are PASS with zero scientific mismatches.

Checkpoint summary:

| instance | path | SHA-256 | config hash | parameters | load |
|---|---|---|---|---:|---|
| UCI | `artifacts/prediction/reconstructed/uci/final_hybrid.pt` | `f724591d7037858b9aaf23b4eb25b32e7e4f47ba69092050e3d9cf9f1a26719d` | `51bec15d04a459a793c367128c9ffd292cde7585c42c598257efb4d615433c26` | 513287 | PASS |
| OULAD Early | `artifacts/prediction/reconstructed/oulad_early/final_hybrid.pt` | `15b23a10dc8bf9373ed53110dea50c133dcf0f93c2c7821403dd05c896164ec6` | `548ab4e997fad4009cbb1e2299f994aa4ac4024b385870f6c33b1269901278d3` | 514247 | PASS |
| OULAD Final-100 | `artifacts/prediction/reconstructed/oulad_final/final_hybrid.pt` | `d0c425eca51e47b6236b181ebcd036501439e30adc03ddb535b905a0362f5f20` | `548ab4e997fad4009cbb1e2299f994aa4ac4024b385870f6c33b1269901278d3` | 514247 | PASS |

The original frozen checkpoint was unavailable, so these are protocol-faithful reconstructions, not byte-equivalent replacements. Historical outer metrics are therefore not reassigned.

## OOF

- UCI: 2,490 rows; group overlap 0.
- OULAD Early: 66,685 rows; group overlap 0.
- OULAD Final-100: 21,728 rows; group overlap 0.
- Recommendation Panel A: 179 inner-OOF rows plus 121 rows from the final Early checkpoint held out from outer-0 development; the latter are explicitly tagged `FINAL_OUTER0_HOLDOUT_INFERENCE_NOT_OOF`.
- No outer test was rerun or consumed.

## RECOMMENDATION

The recommendation code path is reusable. The stale learned artifacts were rebuilt from corrected prediction-derived features, regenerated weak labels, and five EBMs. Risk and safety thresholds were revalidated on development/Panel A data only using the locked grids; no Panel B tuning or overwrite occurred. `seed_disagreement` remains nullable and is not zero-imputed. A new clean held-out recommendation evaluation is still required before new scientific NDCG claims.

## SCIENTIFIC STATUS

- HPO: not performed.
- Outer rerun: not performed.
- Old outer evidence overwritten: no.
- Old recommendation evidence overwritten: no.
- C:\hufit\kltn modified: no.

## TESTS

Environment dependencies were fixed (`imbalanced-learn`, `psycopg2-binary`, `sklearn-compat`, `optuna`, and `interpret`). Full suite: 66 collected, 43 passed, 23 skipped, 0 failed, 0 errors.

## OUTPUTS

- Acceptance: `artifacts/audit/final_acceptance/FINAL_ACCEPTANCE.json`
- Model equivalence: `artifacts/audit/final_acceptance/MODEL_NUMERICAL_EQUIVALENCE.json`
- Data equivalence: `artifacts/audit/final_acceptance/DATA_EQUIVALENCE_DEEP.json`
- Recommendation provenance: `artifacts/audit/final_acceptance/RECOMMENDATION_PROVENANCE_DEEP.json`
- Full tests: `artifacts/audit/final_acceptance/FULL_TEST_SUMMARY.json`
- Reconstructed checkpoints: `artifacts/prediction/reconstructed/`
- Recommendation rebuild: `artifacts/recommendation/phase8_prediction_rebuild/`

This acceptance is conditional in the requested sense: runtime prediction and recommendation artifacts are rebuilt and validated, while historical held-out evidence remains attached only to its original frozen run.
