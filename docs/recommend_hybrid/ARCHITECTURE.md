# Hybrid CNN-BiLSTM Learning Support Recommender

## Scientific identity

- Module: `recommend_hybrid`
- Vietnamese name: **Mô hình khuyến nghị hỗ trợ học tập dựa trên CNN-BiLSTM hybrid**
- Architecture family: `HYBRID_CNN_BILSTM_RECOMMENDER`
- Prediction backbone: `FROZEN_HYBRID_CNN_BILSTM`
- Recommendation component: `HYBRID_ACTION_RANKER`
- Plan component: `HYBRID_LEARNING_PLAN_BUILDER`
- Authority: `RECOMMEND_HYBRID_MODEL_AUTHORITY`

## Architecture

```text
Pre-cutoff student data
  -> Frozen hybrid CNN-BiLSTM
       -> prediction class and probabilities
       -> calibrated confidence and uncertainty
       -> 64-D student-state embedding
       -> 32-D tabular-expert embedding
  + Observed learning state with evidence mask and lineage
  -> recommendation adapter
  -> hybrid action ranker
  -> hybrid constraint solver
  -> hybrid multi-week learning-plan builder
  -> advisor review
```

The prediction checkpoint is immutable. Recommendation training cannot alter model weights, parameter count, architecture semantics, cutoff policy or prediction results. The recommendation component only ranks actions; it does not predict academic outcome again.

## Stage-aware behavior

`EARLY_20` is screening with evidence-gated recommendations; `EARLY_35` allows early intervention; `MIDDLE_50` is the primary recommendation stage; `LATE_75` supports late intervention/escalation; `FINAL_EVALUATION` is evaluation-only and cannot generate a new plan. One shared canonical checkpoint per fold/seed covers the four intervention stages. Dedicated endpoint checkpoints are used only for final evaluation.

## Frozen outputs

Read the 64-D fused student-state representation and 32-D tabular-expert representation already returned by the model. Probabilities include the complete hybrid risk path, including the bounded tabular residual contribution. Do not recompute the prediction from an incomplete embedding. Ensemble disagreement across five locked seeds is the principal uncertainty input; calibrated confidence requires its own versioned calibration provenance.

## Future components, not implemented in Phase 1

`HybridPredictionAdapter` will expose detached frozen outputs. `HybridStudentRepresentation` will combine embeddings, probabilities, uncertainty and cutoff-safe observed state. `HybridActionEncoder` and `HybridActionRanker` will score candidate relevance using only real expert labels. `HybridConstraintSolver` will enforce evidence, workload, action caps, uniqueness, prerequisites, incompatibility, abstention and escalation. `HybridLearningPlanBuilder` will schedule safe selected actions with goals, rationale, success criteria and review points. `HybridRecommendationService` and `HybridRecommendationValidator` are later integration components.

## Thesis description

“Mô hình Hybrid CNN-BiLSTM Learning Support Recommender là mô hình khuyến nghị hỗ trợ học tập được xây dựng trực tiếp trên biểu diễn sinh viên và kết quả dự đoán của mô hình hybrid CNN-BiLSTM. Mô hình sử dụng xác suất rủi ro, độ bất định, biểu diễn ẩn của sinh viên và trạng thái học tập quan sát được trước thời điểm dự báo để xếp hạng các hành động hỗ trợ. Các hành động được kiểm tra bằng những ràng buộc về khối lượng học tập, điều kiện tiên quyết, tính an toàn và độ đầy đủ của bằng chứng trước khi được sắp xếp thành lộ trình học tập theo từng giai đoạn.”

Historical experiment aliases and paths appear only in provenance metadata; they are not production names, authority IDs or scientific conclusions.

## Phase 2 foundation status

Phase 2 implements only the frozen-output adapter, immutable data contracts, cutoff-safe observed-state builder, controlled action catalog, eligibility-only candidate generator, and blinded expert-label export/import boundary. It does not implement or train `HybridActionRanker`, select Top-K actions, solve plan constraints, build a learning plan, or change a production API/database.

The adapter reads `binary_logit`, `student_state_embedding`, and `tabular_expert_embedding` already returned by the frozen forward path. For the locked five-seed ensemble it averages seed probabilities, reports population standard deviation as seed disagreement, computes binary predictive entropy from the mean risk probability, and averages the two representation tensors across the same seeds. Raw maximum class probability is exposed as classification confidence with `confidence_source=RAW_MAX_CLASS_PROBABILITY`; it is not mislabeled as calibrated confidence because no recommender-specific calibrator provenance is frozen.
