# Final Recommendation Held-Out Evidence

## Status

`FAIL_HOLDOUT_CONTAMINATION`

The final recommendation evaluation was not opened. The immutable protected set contains 121 unique queries, 36 students, and 36 groups. Its ID-set SHA-256 is `931f1230cf0603891cacdb568c9f9b3e3be4704bdfd3121d86c88cef7fc7044a`.

## Contamination proof

The rebuilt recommendation feature table contains 300 queries, including all 121 protected rows. `finish_recommendation_phase8.py` passes the full feature frame to `fit_ebms`; the EBM fit path has no protected-holdout exclusion. The weak-label fit also consumes the full Panel A vote matrix before the protected set is removed. Risk threshold revalidation and the development NDCG artifact were computed over the 300-query scope.

Therefore the required zero-overlap condition fails at the direct recommendation training/development-frame level. The clean nonholdout partition itself has zero student/group overlap with the protected set, but that does not repair the fact that the protected rows were actually present in the fitted recommendation workflow.

## Holdout access

- holdout opened: `false`
- holdout run count: `0`
- outcome/relevance values opened before the decision: `false`
- final NDCG@3: not computed
- secondary/action/baseline metrics: not computed

The old Panel B evidence remains historical and was not merged. No repair or same-run evaluation was performed.

## Required next scientific action

Create a genuinely untouched recommendation holdout, rebuild or refit the recommendation artifacts without any protected-row access, freeze them, and then perform one final evaluation. Do not assign this run any new held-out evidence.
