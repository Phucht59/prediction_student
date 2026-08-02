# Configuration authority

| Config | Function | Read by | Authority |
| --- | --- | --- | --- |
| `configs/final/final_model_authority.yaml` | Frozen MAT/POR/OULAD prediction authority | prediction validation/adapter | Final prediction authority |
| `configs/final/recommendation.yaml` | Final system boundary | release documentation/registry | Final recommendation authority |
| `configs/recommend_hybrid/planning.yaml` | Action/workload constraints | policy pipeline | Locked operational configuration |
| `configs/recommend_hybrid/policy_*.yaml` | Dataset evidence policies | UCI/OULAD policy engines | Locked evidence policy |
| `configs/recommend_hybrid/scientific_labeling.yaml` | Phase 1/2 research labeling | scientific-labeling scripts | Diagnostic research only |
| `configs/recommend_hybrid/scientific_model_*.yaml` | Excluded neural-ranker study | archived diagnostic | Non-release only |
