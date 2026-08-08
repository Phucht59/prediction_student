# Official Recommendation V2 release tools

Only the `explainable_v2` release tools below remain in the public tree:

- `verify_hybrid_oof_authority.py`: read-only verification of the frozen Hybrid
  CNN–BiLSTM prediction authority;
- `audit_phase9_end_to_end.py` and `audit_phase10_final.py`: deterministic release
  audits;
- `evaluate_panel_b_final_heldout_v1.py`: preregistered evaluator retained for hash
  provenance only; Panel B must not be evaluated again;
- `freeze_final_release_v1.py`: deterministic final release manifest builder;
- `run_plausibility_simulator.py`: reports model-implied risk delta only.

Training, model-selection, challenger, provider-dispatch, weak-label development,
causal/counterfactual, and superseded pipeline scripts are archived locally under
`test_lab/recommend_hybrid_v2_legacy_20260808/` and are not release authority.
