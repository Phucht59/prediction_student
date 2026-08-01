# Phase 2 silver-label results

The generated dataset contains 984,855 `student_state x candidate_action` rows with three soft probabilities, expected relevance, a conservative hard-label policy, LF diagnostics, split, and lineage fields. The full Parquet files are local artifact-policy outputs; committed manifests, checksums, and a redacted sample provide reproducible audit evidence.

The locked operating policy is confidence `0.75` and at least `2` independent LF families, selected without test-split fitting or tuning. These are project operating thresholds, not universal Snorkel thresholds. The report contains label quality only and no recommendation accuracy, ranking metric, expert validation, deployment, or causal claim.
