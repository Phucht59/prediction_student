# Hybrid-only final recommendation architecture

## Final decision

The final thesis recommendation module uses exactly one learned model:

```text
Frozen residual CNN–BiLSTM hybrid predictor
```

No XGBoost, LightGBM, LambdaMART, Logistic Regression ranker, pairwise ranker or other learned recommendation model is permitted in the final runtime path.

Outcome-grounded V2.1 with XGBoost/LambdaMART is retained only as a research experiment. Its artifacts are not deleted, but it is not the final architecture and cannot become runtime authority.

## Final pipeline

```text
OULAD data observed before the stage cutoff
        ↓
Frozen preprocessing authority
        ↓
Frozen residual CNN–BiLSTM
        ↓
Baseline risk, risk band and uncertainty
        ↓
Deterministic stage-aware candidate policy
        ↓
Hybrid-only counterfactual simulation
        ↓
The same frozen CNN–BiLSTM predicts each simulated state
        ↓
Deterministic transparent utility score
        ↓
Prerequisite, conflict, workload and safety constraints
        ↓
Top recommendation plan or abstention/fallback
```

## Deterministic recommendation score

For an eligible action `a`:

```text
risk_reduction(a) = max(p_baseline - p_counterfactual(a), 0)

utility(a) =
    risk_reduction(a)
    × observed_evidence_strength(a)
    × (1 - hybrid_uncertainty(a))
    ÷ workload_factor(a)
```

This score is a deterministic calculation. It is not a second machine-learning model.

The ranker must use only:

- baseline risk from the frozen hybrid predictor;
- counterfactual risk from the same frozen hybrid predictor;
- uncertainty from the frozen hybrid ensemble/checkpoint authority;
- observed pre-cutoff behavioral evidence;
- workload, prerequisites, conflicts and safety priority.

## Scientific meaning

The module can be evaluated for:

1. Correct and reproducible use of the frozen hybrid model.
2. Deterministic replay and checkpoint authority.
3. Temporal leakage prevention.
4. Constraint and protected-feature safety.
5. Stability across folds, stages and seeds.
6. Coverage, fallback, abstention, workload and action diversity.
7. Observational alignment between fixed hybrid-only recommendations and later behavior/outcomes.
8. Comparison with deterministic baselines such as random feasible action, lowest workload and rule-only policy.

The valid claim is:

> The system converts multi-stage risk predictions from the residual CNN–BiLSTM into transparent, constrained learning-action recommendations and evaluates their offline observational alignment with later OULAD trajectories.

The module must not claim:

- a causal treatment effect;
- guaranteed improvement in grades;
- expert validation;
- production readiness;
- that model-estimated risk reduction is independently proven educational benefit.

## Status of previous experiments

### Counterfactual V1

- Engineering implementation: complete.
- Internal model consistency: available.
- External observational validation: failed.
- Final-runtime authority: no, until corrected hybrid-only validation is completed.

### Outcome-grounded V2/V2.1

- These experiments introduced a learned action ranker.
- V2.1 used LambdaMART through XGBoost.
- This violates the final hybrid-only architecture requirement.
- Their artifacts remain historical research evidence only.
- Remaining V2.1 XGBoost controls and ablations are no longer required for the final thesis module unless explicitly retained as an appendix experiment.

## Required next work

Build and validate a corrected hybrid-only release without a learned action ranker:

1. Freeze the action-policy and utility formula before evaluation.
2. Audit action-to-feature mutations against the canonical 47 sequence and 165 aggregate feature authorities.
3. Use only frozen hybrid checkpoints for baseline and counterfactual inference.
4. Recompute full-cohort recommendations.
5. Evaluate deterministic baselines on the same candidate sets.
6. Run leakage, stability, shortcut, safety and deterministic replay audits.
7. Run observational future-trajectory alignment as secondary evidence.
8. Keep causal and expert-validation claims disabled.

Runtime integration and PR merge remain forbidden until the hybrid-only release gate passes.
