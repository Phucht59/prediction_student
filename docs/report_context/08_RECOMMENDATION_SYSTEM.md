# Recommendation system

The final component is `student_mat_rule_policy_v3`, a deterministic rule-based
advisory policy. Inputs are permitted feature context, predicted class and
confidence. Outputs contain risk factors, prioritized actions, reasons,
confidence, disclaimer and human-review framing. It is not a machine-learning
recommender and does not make automatic educational decisions.

For 79 locked-test outputs, valid-schema, explanation, specific-action,
no-contradiction, no-sensitive-metadata-leak and cautious-low-confidence rates
are all 1.0. Risk-band counts are High 31, Medium 33, Low 15. Expert evaluation
is `not_collected`; structural correctness is not evidence of effectiveness.

True G3 and sensitive variables (sex, school, address, guardian, paid,
alcohol and going-out) are excluded from automated rules. Artifact-derived
frequent factors include prior-grade gap (42), partial support gap (34),
attendance absences (26), low study time (20), and failure history (14).
