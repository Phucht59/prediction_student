# UCI V5.1 authority restoration audit

## Finding

The actual historical UCI V5.1 implementation was found in Git commit `f51e1f43d9af768194bb34ea5ff4aeb8e4cbd502`, immediately before deletion commit `46e3af01`. The V5.1 common loader, model, training, transfer, and UCI runner were controlled-extracted under `historical_uci_v5_1/` only.

Frozen MAT checkpoint metadata confirms the authority candidate `cnn_bilstm_v5_1_transfer_selected` uses `shared_trunk_subject_specific_heads`; the final authority aggregates five seeds. The restored MAT/POR YAML files identify the original raw data, G3 prohibition, three-class target, split-manifest hashes, inner grouping policy, and seed list.

## Dependency status

| Dependency | Status |
|---|---|
| Historical V5.1 UCI source/config | AVAILABLE_HISTORICAL_GIT |
| Current raw MAT/POR source files | AVAILABLE_CURRENT |
| Frozen checkpoint/authority metadata | AVAILABLE_FROZEN_ARTIFACT |
| Historical split manifests | AVAILABLE_HISTORICAL_GIT |
| Isolated compatibility shim for deleted `src.studies.v5*` imports | MISSING |
| Isolated extraction of split artifacts at the historical protocol paths | MISSING |

## Gate

Restoration is blocked before any training. Executing the extracted runner without the two isolated compatibility dependencies would substitute the current implementation or fail; either outcome would invalidate NONE reproduction. No 50-job NONE matrix has been launched.

OULAD readiness is independent: the current repository retains H1, parameter count 160492, FINAL-stage authority metadata, three grouped outer folds, five seeds, pooled-inner-OOF threshold policy, and STRICT_REAL_TIME contract. It is not started in this phase.
