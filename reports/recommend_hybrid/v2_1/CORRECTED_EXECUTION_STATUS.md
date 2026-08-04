# Corrected V2.1 execution status

The corrected nested evaluator completed all three outer folds and produced held-out predictions, model-selection ledgers, and 1,000 random-baseline repetitions under `final_oof/`. Four model families were actually fitted: interaction logistic, pairwise logistic, LambdaMART through XGBoost 3.2.0, and boosted tree.

Learner-cluster bootstrap completed 2,000 replicates for separate group-weighted and learner-weighted estimands. Temporal support remains `COMPLETE_INSUFFICIENT_SUPPORT` for the registered 2014J candidate cohort.

Retrained negative controls and ten ablations were started with resumable batches but did not complete within this execution. Therefore the scientific release gate remains closed and the status is not `INCONCLUSIVE` or validated:

`V2_1_IMPLEMENTATION_SCAFFOLD_COMPLETE`  
`V2_1_FULL_EVALUATION_NOT_COMPLETED`

Claim boundary: `OFFLINE_PREDICTIVE_RELEVANCE_NOT_CAUSAL_EFFECT`.
