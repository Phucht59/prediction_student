# V2.1 model selection

Nested grouped selection uses all outer folds (0, 1, 2), learner-grouped inner validation, explicit action-by-state interactions, pairwise ranking, boosted-tree comparison, and LambdaMART when XGBoost is available. Selection is frozen per outer fold and is not runtime integration.
