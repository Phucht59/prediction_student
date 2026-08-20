# Final Model Architecture and Pipeline

Status: **PASS**. Release scope: previous recommendation release.

1. The frozen Hybrid CNN–BiLSTM predicts learner risk. Its architecture and weights were not retrained or tuned by previous recommendation release or Panel B.
2. The frozen router decides whether recommendation processing is justified, using exactly `RECOMMEND`, `INSUFFICIENT_EVIDENCE`, `HUMAN_REVIEW`, and `NO_FEASIBLE_ACTION`.
3. The canonical V4 feasibility policy removes impossible actions before ranking.
4. Five action-specific EBMs score feasible interventions. Native ordinal predictions use the 0–3 scale; the single public adapter produces `clip(native / 3, 0, 1)`.
5. The highest-scoring valid action is recommended only when evidence and ambiguity gates pass.
6. Explanations and plans are generated from observed pre-cutoff evidence.
7. The plausibility simulator reports a **model-implied risk delta** only. It does not estimate a causal treatment effect.

The unavailable `seed_disagreement` value is nullable and is never silently replaced with zero. The frozen router therefore applies no disagreement threshold when no real finite value exists. Panel B was evaluated once after development freeze and was not used for tuning.
