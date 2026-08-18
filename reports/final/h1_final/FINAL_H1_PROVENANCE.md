# Final H1 Provenance

- Phase 5 selected candidate commit: `2c8baa96f563d7a4a5188abfd0c700828a91e301`
- Pre-outer freeze commit: `234e0c6d7e058d5e7d855b8060ddfe2d59093085`
- Final candidate hash: `56f57a59652bbd002dc64c12a6a3f8f35bdab4c620e76c63752a3c72493ce3de`
- Evaluation protocol: `h1_final_outer_v1`
- Outer folds: **3**
- Inner folds: **2**
- Seeds: `42, 1201, 2026, 3407, 7319`
- Stages: `20%, 35%, 50%, 75%`
- Runs: **45**
- Same checkpoint across stages: **YES**
- Threshold source: `PHASE5_POOLED_INNER_OOF_SEED42`
- Outer labels used for epoch/threshold selection: **NO**
- Optuna trials: **0**
- Old official evidence preserved: **YES**

H0 and MLP were recomputed under the same frozen folds, stages, seed aggregation,
and Phase 5 inner-only threshold authority because historical evidence was not
silently assumed protocol-compatible.
