# V6.2 blinded expert review instructions

## Purpose and scope

Evaluate whether each proposed plan is relevant, supported by the displayed
historical pre-cutoff evidence, safe, and feasible. This is a scientific review
of decision-support output. It is **not** evidence that an action causes better
student outcomes, and no student outcome is shown.

The cases hide model identity, exact prediction probabilities, source record
identifiers, student identifiers, and outcomes. Reviewer IDs must remain
pseudonymous (`E01`, `E02`, ...). Do not add names, email addresses, or other
personal information.

## Independent review

Complete the assigned randomized order independently before discussing cases
with another reviewer. Do not infer a missing behavior from a risk band. A
reason such as `LOW_VLE_ENGAGEMENT` is acceptable only when the displayed
observed pre-cutoff evidence supports it.

## Questions

1. `q1_plan_score` — overall plan quality, integer 1 (very poor) to 5 (very good).
2. `q2_action_relevance` — one value for every proposed action:
   `APPROVE`, `PARTIAL`, `UNSURE`, or `REJECT`.
3. `q3_missing_action` — `YES` or `NO`; if `YES`, describe the omitted action.
4. `q4_escalation` — `CORRECT`, `OVER_ESCALATED`, `UNDER_ESCALATED`, or `UNSURE`.
5. `q5_reason_support` — `SUPPORTED`, `PARTIAL`, `UNSUPPORTED`, or `UNSURE`.
6. `q6_safety_workload` — `SAFE`, `CONCERN`, `UNSAFE`, or `UNSURE`; a note is
   required for `CONCERN` or `UNSAFE`.

Do not change `schema_version`, `reviewer_id`, `case_id`, `action_id`, or
`randomized_order`. Blank templates are intentionally pending and contain no
synthetic labels.
