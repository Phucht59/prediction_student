# Final V6 integrated system review

System verdict: **INTEGRATED_SYSTEM_PASS_EXPERT_PENDING**

## Git and immutable bases

- Branch: `codex/project-v6-integrated-risk-recommendation-system`
- Repository integration base: `24cca2b7f0904504e6f1c937af04589938e1a73f`
- Scientific V5.1 source: `308370cf6c6f16e65cc0f0aaa3f38393ae141e16`
- Recommendation V5.2 source: `b9087ceb1600582ad1351b134a2f4c4d9af77d89`
- Report-generation HEAD: `8ee9f43751aa9eeb5e6b06eb00e707e6c65074ee`
- Future OULAD: `LOCKED_NOT_EXECUTED`

## Knowledge audit

- Order destruction: `TEMPORAL_ORDER_LOW_VALUE`
- Residual signal: `RESIDUAL_SIGNAL_HIGH`
- Oracle complementarity gain: 0.026808
- Survival feasibility: `WITHDRAWAL_SURVIVAL_FEASIBLE`
- Graph feasibility: `GRAPH_CONTEXT_PASS`; graph was skipped because Candidate D failed its guardrail.

## Prediction evidence

- V5.1 reproduction: **PASS**
- Selected: `C_TEMPORAL_MULTITASK_W0` (P1 pretraining + withdrawal/outcome heads)
- Parameters: 100,938
- Outer matrix: 3 folds x 5 fixed seeds (15 checkpoints)
- Macro-F1: 0.828084
- At-risk F1: 0.782639
- PR-AUC: 0.893355
- Brier: 0.113355
- ECE before outer reporting calibration: 0.008683
- Recall@10%: 0.250409
- Survival C-index: 0.641216
- Withdrawal recall: 0.000000
- Outcome Macro-F1: 0.615295
- Total recorded training runtime: 3080.5 seconds
- Peak CUDA allocation: 125.2 MiB
- Calibration temperature: 1.043603; fit on inner OOF only
- Domain conclusion: `NO_GENERALIZATION_ADVANTAGE`

CNN-BiLSTM is retained as the temporal thesis model; XGBoost is retained as an
operational cross-check. The integration value lies in progression modelling,
risk prioritization, calibrated uncertainty and governed recommendations.

## Risk profiles

- Schema: `student_risk_profile_v1`
- Records / coverage: 15378 / 100.000%
- Abstention: 20.913%
- Confidence: `{'HIGH_CONFIDENCE': 7825, 'MEDIUM_CONFIDENCE': 4337, 'LOW_CONFIDENCE': 3216}`
- Top-k distribution: `{'OUTSIDE_TOP_20_PERCENT': 12302, 'TOP_20_PERCENT': 1538, 'TOP_5_PERCENT': 769, 'TOP_10_PERCENT': 769}`
- Mean deep-ML disagreement: 0.074542
- Sensitive demographics in payload: False

## Recommendation and governance

- Plans / coverage: 15378 / 100.000%
- Escalation: 46.859%
- Conflicts / duplicates / workload violations: 0 / 0 / 0
- Missing lineage: 0
- Deterministic replay: True
- Linkage/stability: **PASS**
- Database: `SKIP_NO_DISPOSABLE_DSN`; production write = False

## Expert evidence and claims

- Status: **PENDING_EXPERT_LABELS**
- Experts / cases scored: 0 / 0
- Action F1 / Top-3 recall / approval / escalation F1 / agreement: pending real labels
- A blinded 60-case, two-expert package exists; no labels were fabricated.
- Recommendation technical validation is not evidence of causal student-outcome improvement.

## Validation

- V6 validation: **PASS** (26/26)
- Protected V4-V5.4 hashes: **PASS**
- Future lock: **PASS**
