# Two-Stage V4 feasibility audit

## Current held-out result

- End-to-end Precision@1: 0.6589
- Positive-group coverage: 0.4980
- Stage A precision: 0.6953
- Conditional action Precision@1: 0.9476
- Stage A precision required for 80% end-to-end at current conditional precision: 0.8443

## Exact global score frontiers

- direct_gate_probability: best P@1=0.6463, coverage=0.5000, Stage A precision=0.6884
- action_any_probability: best P@1=0.6308, coverage=0.5015, Stage A precision=0.6785
- joint_gate_probability: best P@1=0.6363, coverage=0.5004, Stage A precision=0.6833
- joint_x_top_action_probability: best P@1=0.6613, coverage=0.5000, Stage A precision=0.6990
- direct_x_top_action_probability: best P@1=0.6579, coverage=0.5000, Stage A precision=0.6925

## Optimistic registered-grid oracle

This section uses held-out labels to choose thresholds and is diagnostic only. It cannot authorize release.

- Current-ranking oracle: {"precision": 0.6622596153846154, "coverage": 0.5, "stage_a_precision": 0.6989182692307693, "conditional_precision": 0.9475494411006019, "issued": 6656, "issued_positive": 4652, "correct": 4408, "direct_action_blend": 0.25, "minimum_action_probability": 0.6, "minimum_action_margin": 0.0, "stage_thresholds": [0.6231342517399049, 0.7017532341467202, 0.7529927117809194]}
- Perfect-ranking gate oracle: {"precision": 0.6989182692307693, "coverage": 0.5, "stage_a_precision": 0.6989182692307693, "conditional_precision": 1.0, "issued": 6656, "issued_positive": 4652, "correct": 4652, "direct_action_blend": 0.25, "minimum_action_probability": 0.6, "minimum_action_margin": 0.0, "stage_thresholds": [0.6231342517399049, 0.7017532341467202, 0.7529927117809194]}

## Target stability

- Learners with mixed positive/non-positive stage targets: 5360
- Mixed-target rate: 0.4235

## Interpretation

If the post-hoc registered-grid oracle remains below 0.80 at coverage 0.50, no further threshold or calibration tuning on the current V4 scores can satisfy the original gate. A new representation/target boundary would be required. Conditional action-ranking evidence remains separate from end-to-end recommendability evidence.

Claim boundary: `OFFLINE_PREDICTIVE_RELEVANCE_NOT_CAUSAL_EFFECT`
