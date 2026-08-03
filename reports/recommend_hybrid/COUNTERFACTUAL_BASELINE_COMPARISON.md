# Counterfactual Baseline Comparison

{
  "claim_boundary": "MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT",
  "paired_note": "Comparisons are descriptive paired orderings on the same eligible ranked candidate rows; they are not causal improvements.",
  "same_ranked_candidate_set": true,
  "schema_version": "full_cohort_baseline_comparison_v1",
  "status": "PASS",
  "strategies": {
    "existing_policy_ordering": {
      "action_diversity": 5,
      "coverage_over_all_rows": 0.6632866853258697,
      "mean_risk_reduction": 0.11139249478619152,
      "median_risk_reduction": 0.07388505339622489,
      "records": 41472,
      "success_at_0_01": 1.0,
      "success_at_0_05": 0.6363088348765432,
      "top_action_concentration": 0.5804398148148148,
      "workload_mean_minutes": 56.64171006944444
    },
    "fixed_seed_random_ordering": {
      "action_diversity": 5,
      "coverage_over_all_rows": 0.6632866853258697,
      "mean_risk_reduction": 0.11535763511077404,
      "median_risk_reduction": 0.07532075047492981,
      "records": 41472,
      "success_at_0_01": 1.0,
      "success_at_0_05": 0.6333912037037037,
      "top_action_concentration": 0.5287181712962963,
      "workload_mean_minutes": 77.21028645833333
    },
    "risk_reduction_ordering": {
      "action_diversity": 5,
      "coverage_over_all_rows": 0.6632866853258697,
      "mean_risk_reduction": 0.14015983764145432,
      "median_risk_reduction": 0.09440132603049275,
      "records": 41472,
      "success_at_0_01": 1.0,
      "success_at_0_05": 0.6891396604938271,
      "top_action_concentration": 0.8331163194444444,
      "workload_mean_minutes": 95.59027777777777
    },
    "workload_only_ordering": {
      "action_diversity": 5,
      "coverage_over_all_rows": 0.6632866853258697,
      "mean_risk_reduction": 0.10640591485643208,
      "median_risk_reduction": 0.0691078901290893,
      "records": 41472,
      "success_at_0_01": 1.0,
      "success_at_0_05": 0.6085792824074074,
      "top_action_concentration": 0.7102141203703703,
      "workload_mean_minutes": 47.547019675925924
    }
  }
}
