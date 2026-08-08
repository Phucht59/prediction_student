# Official recommendation module

The only release recommendation implementation is `explainable_v2`.

Runtime flow:

1. consume the frozen Hybrid CNN–BiLSTM risk output;
2. apply the frozen evidence and risk router;
3. remove infeasible actions with the canonical V4 policy;
4. score feasible actions with the five frozen action-specific EBMs;
5. normalize once with `clip(native_prediction / 3, 0, 1)`;
6. return one of `RECOMMEND`, `INSUFFICIENT_EVIDENCE`, `HUMAN_REVIEW`, or
   `NO_FEASIBLE_ACTION`;
7. produce an evidence-grounded plan and a model-implied risk delta.

Release authority is recorded in
`artifacts/recommend_hybrid/explainable_v2/frozen/final_release_v1/FINAL_RELEASE_MANIFEST.json`.
The selected configuration is `a70599afad40`; the five official model files are
under `artifacts/recommend_hybrid/explainable_v2/frozen/ranker_panel_a_v2/final_models/`.

`hybrid_only_final`, `two_stage_v4`, `prediction_adapter.py`, and the repository's
frozen prediction checkpoints are retained as prediction authority. Recommendation
V2 does not retrain, replace, or modify those checkpoints.

Superseded recommendation implementations and experiment-only tooling are archived
locally under `test_lab/recommend_hybrid_v2_legacy_20260808/`. Public runtime code
must not import from `test_lab`.

Scientific boundary: simulator output is a **model-implied risk delta**, not a
causal treatment effect.
