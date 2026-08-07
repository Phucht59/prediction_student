# LLM Annotation Instructions

## Relevance Scale
- 0: Unsuitable or harmful
- 1: Weakly relevant
- 2: Relevant with adequate evidence
- 3: Highly relevant with direct evidence

## Required Response Fields
case_id, action_id, relevance_score, abstain, evidence_ids, rationale,
contraindication_detected, safety_flag, reviewer_id, reviewer_type,
provider, model_name, request_id.
reviewer_type must be: REAL_EXTERNAL_LLM_REVIEW
