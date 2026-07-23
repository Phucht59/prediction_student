# Database Protocol Amendment 001

Decision time: 2026-07-23, during disposable validation and before target
cutover.

The first disposable schema validation counted 24 non-primary indexes because
PostgreSQL indexes created by UNIQUE constraints correctly count toward the
budget. Four explicit indexes duplicated the left-prefix access paths already
provided by UNIQUE constraints.

Migration `009_prune_redundant_indexes.sql` removes only these redundant
indexes:

- `catalog.dataset_version_dataset_idx`
- `ml.model_dataset_idx`
- `ml.artifact_run_kind_idx`
- `recommendation.plan_risk_profile_idx`

The eight previously applied migrations are not edited. No table, row,
prediction, metric, model selection, recommendation, split, seed, or Future
OULAD state changes. The amendment exists solely to enforce the preregistered
maximum of 20 non-primary indexes.
