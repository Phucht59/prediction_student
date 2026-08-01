# Phase 1 targeted audit

Base SHA: `64cece4`. Audit scope was limited to the frozen authority, recommendation configs, action/policy contracts, temporal routers, observed-state builder, and related tests/reports.

| Question | Finding | Evidence |
|---|---|---|
| Prediction authority entry | Policy contexts carry `architecture_authority`; the adapter validates OULAD checkpoint/hash lineage. | `common/policy_contracts.py`, `prediction_adapter.py`, `policy_common.yaml` |
| Alternate model ingress | Generic context construction could name another authority; the Phase 1 guard now locks the dataset-specific model/authority pair. | `weak_supervision/validation.py` |
| Canonical actions | 15 unique IDs: 9 UCI, 10 OULAD, with 4 shared IDs. | `policy_uci_mat.yaml`, `policy_uci_por.yaml`, `policy_oulad.yaml` |
| Dataset scope | UCI-only, OULAD-only, and shared scope is explicit per action in the evidence map. | `action_evidence_map.yaml` |
| Stage/cutoff control | UCI stage routing and OULAD past-only anchor routing already exist; Phase 1 adds candidate-level rejection. | `uci/stage_router.py`, `oulad/cutoff_router.py` |
| Future/sensitive fields | `ObservedStateBuilder` rejects post-cutoff events and a prohibited-field set; UCI policies forbid G3. | `observed_state.py`, `policy_uci_*.yaml` |
| Reused deliverables | Existing dataset policy action sets, authority configs, temporal routers, and prohibited-field semantics were reused. | paths above |

No checkpoint, frozen prediction, metric, database schema, checksum, or runtime recommendation output was modified.
