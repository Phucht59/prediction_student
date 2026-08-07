# LLM Annotation Instructions for Student Action Ranking

## Relevance Scale
- **0**: Unsuitable or potential harm (e.g. recommending quiz practice when no quizzes exist).
- **1**: Weakly relevant (generic advice, low specificity).
- **2**: Relevant (direct alignment with observed student behavioral gaps).
- **3**: Highly relevant (urgent action matching specific missing assessment or inactivity streak).

## Required Response Provenance Fields
Each response MUST contain authentic external provider metadata:
`case_id`, `panel_id`, `action_id`, `relevance_score` (0-3 or abstain=true),
`evidence_ids`, `rationale`, `contraindication_detected`, `safety_flag`,
`reviewer_id`, `reviewer_configuration_id`, `reviewer_type`="REAL_EXTERNAL_LLM_REVIEW",
`provider`, `model_name`, `request_id`, `batch_id`, `prompt_version`, `prompt_sha256`, `created_at`.
