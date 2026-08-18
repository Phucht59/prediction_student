# Phase 9 score-feature authority

Decision: **SCORE_PROXY_REJECTED**

Official endpoint semantics remain `F2_MIDDLE`: events are legal only when
`0 <= event_day < floor(module_presentation_length * 0.50)`.

Historical score availability used `max(date_submitted, assessment_due_date) < cutoff_day`. This proves
that submission and due dates precede cutoff, but not that a marker released
the score before cutoff. Raw OULAD has no score-release timestamp. Therefore
score values and the two score-based pretraining tasks are excluded. This
decision was made without using performance as an authorization criterion.
