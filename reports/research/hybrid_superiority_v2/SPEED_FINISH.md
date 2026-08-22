# SPEED_FINISH — documented budget cuts

User requested a crash finish after the PC looked idle (OULAD DT HPO hung ~2h, then killed).

This is **not** the preregistered protocol budget.

| Item | Preregistered | SPEED_FINISH |
|---|---|---|
| OULAD baseline trials | 28 / model | 4 (XGB/CatBoost only) |
| Skipped HPO | none | DT, SVM, MLP, RF (defaults; DT previously hung) |
| Lock folds × seeds | 3 × 3 | [0] × [42, 1201] |
| Hybrid screen trials | 24 / candidate | 6 C0-R only |
| Hybrid epochs | 24 / patience 8 | 10 / 4 |
| OULAD diagnose | required | skipped (GPU reserved for C0-R) |
| Ablation | independent 3×3 | UCI fold-0 seed-42 only |

GPU: CatBoost/XGB `task_type/device=GPU` on RTX 2060; Hybrid tensors pinned; process HIGH priority.

Protocol hash still `eb5f4cfbf4e1629281386367400970241ff68fdaec6c0f7905b0e3a6f33646a2`. Outer test unused. Serving Hybrid **not** promoted.
