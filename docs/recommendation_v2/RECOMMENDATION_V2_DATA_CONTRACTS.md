# Recommendation V2 data contracts

All timestamps are UTC ISO-8601; IDs and hashes are non-empty strings; probabilities are finite in `[0,1]`. Required fields cannot silently default. Every student-derived field must declare cutoff and lineage.

## PredictionContext

| Field | Type | Req. | Validation/source | Cutoff and lineage |
|---|---|---:|---|---|
| context_id | string | yes | unique deterministic ID | hashes all versions below |
| record_id, student_key | string | yes | prediction record; pseudonymized downstream | same locked record |
| dataset, stage | enum | yes | selected baseline manifest | stage defines cutoff |
| cutoff_at/cutoff_day | timestamp/int | yes | pipeline cutoff manifest | all input max times `<` or approved `<=` rule |
| model_id, model_version | string | yes | model registry | exact authority |
| checkpoint_sha256, architecture_hash, config_hash | SHA-256 | yes | baseline lock | immutable lineage |
| logits | float array | optional | frozen forward | never persisted if policy forbids |
| probabilities | float array | yes | sigmoid/softmax; sum≈1 | calibration lineage required |
| predicted_class | int/string | yes | threshold/argmax | threshold version required |
| calibrated_confidence | float | yes | locked calibration | calibration split/hash required |
| uncertainty, seed_disagreement | float | optional | ensemble/entropy contract | `NOT_IMPLEMENTED` distinct from zero |

## StudentRepresentation

| Field | Type | Req. | Validation/source | Cutoff and lineage |
|---|---|---:|---|---|
| context_id | string | yes | joins PredictionContext | exact join |
| embedding | float32 array | yes | frozen pre-head fusion | `[64]` for selected OULAD H1 |
| embedding_layer | string | yes | `student_state_embedding` | architecture hash bound |
| embedding_dim, dtype, device_class | int/string | yes | runtime assertions | replay metadata |
| eval_mode, detached | bool | yes | both true | no gradients to predictor |
| embedding_sha256 | SHA-256 | optional | canonical byte encoding | request/replay trace |

## ObservedLearningState

| Field | Type | Req. | Validation/source | Cutoff and lineage |
|---|---|---:|---|---|
| record_id, stage, cutoff | string | yes | matches PredictionContext | exact match |
| activity_level | float | optional | `[0,1]`, VLE | pre-cutoff source max day |
| inactivity_streak | int | optional | `>=0`, VLE | pre-cutoff source max day |
| assessment_progress | float | optional | `[0,1]`, due/submitted | both due/submission cutoff-safe |
| grade_trend | float | optional | `[-1,1]`, released/available score only | availability timestamp required |
| course_progress, weeks_remaining | float/int | optional | course calendar | computed at cutoff |
| available_evidence_mask | map[string,bool] | yes | keys equal state fields | missing ≠ zero |
| feature_lineage | map[string,Lineage] | yes | source, transform, max time, hash | reject post-cutoff/sensitive |

## CandidateAction

| Field | Type | Req. | Validation/source | Cutoff and lineage |
|---|---|---:|---|---|
| action_id, catalog_version | string | yes | unique immutable catalog | catalog hash |
| category, description | string | yes | approved catalog | localized text version |
| workload_minutes | int | yes | positive and within catalog maximum | no student outcome data |
| applicable_stages | enum array | yes | non-empty | current stage included |
| required_evidence | string array | yes | defined state keys | checked against mask |
| prerequisites, incompatible_actions | ID array | yes | all IDs defined; acyclic prerequisites | catalog lineage |
| contraindications | rule array | yes | deterministic safe rules | only allowed evidence |
| human_review_required | bool | yes | catalog safety owner | immutable per version |

## ActionScore

| Field | Type | Req. | Validation/source | Cutoff and lineage |
|---|---|---:|---|---|
| context_id, action_id | string | yes | unique pair | joins exact inputs |
| raw_score | float | yes | finite ranker output | ranker version/hash |
| calibrated_relevance | float | optional | `[0,1]` if calibrated | validation split lineage |
| ranker_version, adapter_version | string | yes | model registry | training data/label manifest |
| evidence_mask | map | yes | input mask snapshot | cutoff-safe only |
| abstention_signal | bool | yes | uncertainty/data policy | reason code required if true |

## SelectedAction

| Field | Type | Req. | Validation/source | Cutoff and lineage |
|---|---|---:|---|---|
| action_score | ActionScore ref | yes | exact candidate | immutable ref |
| selected_rank | int | yes | positive unique | deterministic tie-break |
| scheduled_week | int | yes | within plan horizon | after cutoff only as future intervention |
| workload_minutes | int | yes | equals catalog unless advisor modification | solver trace |
| selection_reasons, rejected_constraints | code array | yes | known vocabulary | evidence refs required |
| requires_human_review | bool | yes | catalog/uncertainty solver | cannot be downgraded silently |

## LearningPlan

| Field | Type | Req. | Validation/source | Cutoff and lineage |
|---|---|---:|---|---|
| plan_id, revision | string/int | yes | unique; revision `>=1` | supersession chain |
| context_id | string | yes | exact prediction/state | complete version lineage |
| status | enum | yes | DRAFT/ABSTAINED/REVIEW_REQUIRED/ACTIVE/REJECTED | transition validation |
| selected_actions | SelectedAction array | yes | empty only for abstention | solver version |
| goals, rationale, success_criteria, review_points | structured arrays | yes | non-causal language | evidence refs |
| weekly_workload | map[int,int] | yes | each week within cap | deterministic sum |
| abstention_reasons | code array | optional | required when abstained | uncertainty/data lineage |
| model/adapter/ranker/catalog/solver versions | strings | yes | registered | exact hashes |

## RecommendationEvidence

| Field | Type | Req. | Validation/source | Cutoff and lineage |
|---|---|---:|---|---|
| evidence_id, context_id | string | yes | unique | exact join |
| evidence_type | enum | yes | prediction/observed/constraint | allowed vocabulary |
| field_name, value_band | string | yes | do not expose sensitive raw values | source mapping |
| source_artifact, source_sha256 | string | yes | exists/hash matches | immutable |
| source_max_time, cutoff | timestamp/int | yes | `source_max_time <= cutoff` per feature contract | reject violation |
| transformation_version | string | yes | registered | full lineage |

## AdvisorReview

| Field | Type | Req. | Validation/source | Cutoff and lineage |
|---|---|---:|---|---|
| review_id, plan_id, plan_revision | string/int | yes | exact existing plan | immutable reviewed snapshot |
| expert_id/advisor_id | string | yes | authorized reviewer | pseudonymous audit key |
| status | enum | yes | APPROVED/MODIFIED/REJECTED/NEEDS_MORE_EVIDENCE | no free-form substitute |
| reason, comment | string | yes | non-empty for non-approval | audit record |
| modified_actions | structured diff | optional | required for MODIFIED | before/after hashes |
| reviewed_at | timestamp | yes | server time | append-only |
| safety_concern, escalation_required | bool | yes | explicit | cannot be null |

Contract changes require a new schema version and migration in a later approved phase; Phase 1 changes no production schema.
