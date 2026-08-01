# recommend_hybrid data contracts

These Phase 2 contracts are specifications only. All student-derived values require stage, cutoff, source, transformation version and maximum source time. Missing is never encoded as zero.

| Contract | Required fields | Validation and source |
|---|---|---|
| `PredictionContext` | record/student key; dataset; standardized stage; cutoff; model/checkpoint/config/architecture hashes; logits optional; probabilities; class; calibrated confidence; uncertainty/seed disagreement | frozen authority; finite probabilities; calibration lineage; no final-stage plan generation |
| `HybridStudentRepresentation` | context ID; 64-D student state; 32-D tabular expert state; dtype; eval/detached flags; layer names | existing frozen forward outputs; exact dimensions; no gradients into predictor |
| `ObservedLearningState` | activity; inactivity; assessment progress; grade trend; course progress; evidence mask; per-field lineage | pre-cutoff only; availability timestamp required; reject sensitive/outcome fields |
| `CandidateAction` | action ID; category; description; workload; applicable stages; required evidence; prerequisites; contraindications; incompatibilities; human-review flag | immutable catalog version; unique ID; valid dependency graph |
| `ActionScore` | context/action pair; raw score; calibrated relevance optional; ranker/adapter versions; evidence mask; abstention signal | finite; real-label training lineage; no outcome-prediction target |
| `SelectedAction` | score reference; unique rank; week; workload; reasons; rejected constraints; human-review flag | deterministic solver; catalog workload; no duplicate |
| `LearningPlan` | plan/revision IDs; context; status; selected actions; goals; rationale; success criteria; review points; weekly workload; all component versions | empty actions only for abstention; complete supersession/version lineage |
| `RecommendationEvidence` | evidence ID/type; field/band; source path/hash; max source time; cutoff; transformation version | source time must satisfy field cutoff rule |
| `AdvisorReview` | review/plan/revision IDs; reviewer; status; reason; modified-action diff; timestamp; safety/escalation flags | status is APPROVED/MODIFIED/REJECTED/NEEDS_MORE_EVIDENCE; append-only |

The adapter input is exactly: 64-D student state + 32-D tabular expert state + probabilities + uncertainty + observed learning state. A separate prediction model is prohibited.
