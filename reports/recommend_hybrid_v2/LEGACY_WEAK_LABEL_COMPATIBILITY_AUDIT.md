# Legacy Weak-Label Infrastructure Compatibility Audit

## 1. Executive Summary
- **Legacy Pipeline Status**: Existing pipeline scripts and artifacts found.
- **Legacy Snorkel Model Found**: True
- **Exported Expert Cases Count**: 60
- **Real Human Reviews**: 0
- **Real LLM Reviews**: 0
- **Snorkel Cardinality**: 4 (Ordinal scores 0..3)

## 2. Answers to Audit Questions

### Q1-Q3: Existing Infrastructure & Runs
- Found 5 legacy scripts for exporting expert cases, building candidates, generating silver labels, and fitting Snorkel LabelModel.
- Total canonical cases exported: `60`.

### Q4-Q6: Review Verification
- Real human review count: `0`
- Real LLM review count: `0`
- Note: `expert_cases.json` is an input case export package, NOT an annotation response file.

### Q7-Q9: Taxonomy & Cardinality Mapping
- Legacy Snorkel LabelModel cardinality: `4` (Relevance scores: 0 = Unsuitable/Harmful, 1 = Low, 2 = Suitable, 3 = Highly Relevant).
- Legacy Taxonomy mapped to V2 5 Canonical Actions:
  - `ASSESSMENT_COMPLETION` → `ASSESSMENT_COMPLETION`
  - `VLE_ENGAGEMENT` → `RECOVER_ENGAGEMENT` (only when evidence shows engagement drop/inactivity)
  - `STUDY_SCHEDULE` → `STUDY_REGULARITY`
  - `LEARNING_CONSOLIDATION` / `CONTENT_REVIEW` / `TARGETED_REVISION` → `TARGETED_CONTENT_REVIEW`
  - `RETRIEVAL_PRACTICE` / `PRACTICE_EXERCISES` / `ASSESSMENT_PREPARATION` → `QUIZ_RETRIEVAL_PRACTICE`

### Q10-L12: Reusability & Risk Controls
- **Circular Labeling Risk**: Mitigated by excluding prediction scores from LF inputs.
- **Action-Stage Shortcut Risk**: Mitigated by training 5 separate EBM regressors without `action_id` feature.
- **Post-Cutoff Leakage Risk**: Enforced by filtering features strictly at or before `cutoff_day`.
- **Student Overlap Risk**: Grouped CV splits strictly by student ID.
- **Missing Artifacts for Final Run**: Real LLM annotation responses.
