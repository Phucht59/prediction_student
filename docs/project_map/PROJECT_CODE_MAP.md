# Project code map

## Pipeline

`Raw data → preprocessing → Hybrid CNN-BiLSTM prediction → frozen prediction artefacts → recommendation stage router → evidence policy → constraint solver → LearningPlan → evaluation → final validation`

## Prediction

| Area | Main files |
| --- | --- |
| Student-Mat / Student-Por authority | `configs/final/cnn_bilstm_mat.yaml`, `configs/final/cnn_bilstm_por.yaml` |
| OULAD authority | `configs/final/h1_tabular_residual_oulad.yaml`, `src/models/oulad_tabular_residual.py` |
| Frozen adapter | `src/recommend_hybrid/prediction_adapter.py` |
| Checkpoint authority | `configs/final/final_model_authority.yaml` |

## Recommendation

| Function | File |
| --- | --- |
| Runtime pipeline | `src/recommend_hybrid/pipeline.py` |
| UCI routing/policy | `src/recommend_hybrid/uci/stage_router.py`, `src/recommend_hybrid/uci/policy.py` |
| OULAD cutoff/policy | `src/recommend_hybrid/oulad/cutoff_router.py`, `src/recommend_hybrid/oulad/policy.py` |
| Evidence and explanations | `src/recommend_hybrid/common/evidence.py`, `src/recommend_hybrid/common/explanation.py` |
| Constraint solver | `src/recommend_hybrid/common/constraints.py` |
| LearningPlan contract | `src/recommend_hybrid/common/plan_contracts.py` |
| CLI | `scripts/recommend_hybrid/generate_plan.py` |
| Build/evaluation | `scripts/recommend_hybrid/build_final_recommendations.py`, `scripts/recommend_hybrid/evaluate_final_recommendations.py` |
| Final validation | `scripts/recommend_hybrid/validate_final_evidence_recommender.py` |

## Research-only areas

`src/recommend_hybrid/weak_supervision/` supports Phase 1/2 scientific diagnostic and
label research; it is not final recommendation authority. The neural ranker is
`NON_RELEASE_RESEARCH_DIAGNOSTIC`, excluded because silver targets are
shortcut-confounded.
