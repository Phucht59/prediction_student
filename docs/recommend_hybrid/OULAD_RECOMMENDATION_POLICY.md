# OULAD recommendation policy

## Arbitrary-cutoff behavior

Requests may occur anywhere in 0–100% course progress. The prediction anchor is the latest validated cutoff not later than the request; observed evidence is aggregated independently through the actual request. Prediction age is reported but never used to recalibrate probability or invent confidence decay.

Before 20%, the policy abstains with `NO_VALIDATED_PREDICTION_ANCHOR`. At 100%, it returns `EVALUATION_ONLY` and zero intervention actions. Invalid percentages are rejected.

## Evidence and actions

OULAD evidence may include activity level/trend, inactivity streak, due assessment count/completion, course progress, verified released-grade trend, explicit knowledge-gap evidence and missingness. All event-derived evidence must end strictly before the requested cutoff. A score trend without verified release timing is rejected.

The branch declares the ten OULAD actions from `policy_oulad.yaml`. Assessment completion requires both a due assessment and incomplete progress. Targeted practice requires explicit knowledge-gap evidence and never invents a topic. Critical direct evidence may require human contact; risk alone cannot create escalation.

Increasing inactivity or worsening completion cannot reduce related priority. Completion at 100% or zero due assessments removes the completion action. Increasing uncertainty cannot raise automation.
