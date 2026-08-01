# recommend_hybrid expert-label protocol

Current status: `PENDING_REAL_EXPERT_LABELS`; training status: `LOCKED_UNTIL_REAL_EXPERT_LABELS`. Existing templates and prepared cases are not labels. Reviewers submitted: 0; cases scored: 0.

Each real rating requires `case_id`, `action_id`, `expert_id`, `relevance_score`, `approval_status`, `missing_action`, `safety_concern`, `escalation_required`, `reason_support` and `comment`.

Ordinal relevance scale: 3 highly suitable; 2 suitable; 1 may be considered; 0 unsuitable; -1 unsafe. A -1 rating must set `safety_concern=true` and enter adjudication. Plan decisions use `APPROVED`, `MODIFIED`, `REJECTED`, `NEEDS_MORE_EVIDENCE`.

Cases must be blinded to outcome, exact probability, internal record ID, model aliases and sensitive attributes. Candidate order is randomized. Sampling covers standardized stages, risk class, uncertainty/abstention and evidence completeness. Students never cross train/validation/test splits.

Training remains blocked until at least two real reviewers, the approved overlap count, schema/completeness checks, ordinal and safety agreement reporting, safety adjudication and immutable raw/adjudicated label manifests exist. Rules, model predictions and language-model judgments cannot be converted into expert labels.
