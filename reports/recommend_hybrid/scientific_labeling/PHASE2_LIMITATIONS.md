# Phase 2 limitations

- Silver labels are probabilistic weak-supervision outputs, not expert gold labels.
- `ASSESSMENT_COMPLETION` and `ATTENDANCE_IMPROVEMENT` remain evidence gaps and are capped at `CONDITIONAL` unless another rule marks them inappropriate.
- UCI MAT/POR keys are dataset-scoped because cross-dataset identity has not been verified.
- Human-review actions retain their review flag and cannot retain `APPROPRIATE` as an autonomous hard action label.
- Label agreement is not educational effectiveness, user acceptance, or a causal outcome estimate.
