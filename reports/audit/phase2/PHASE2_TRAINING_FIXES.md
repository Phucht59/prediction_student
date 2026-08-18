# Phase 2 — Training Fixes

| Area | Repair | Validation |
| --- | --- | --- |
| Fixed refit metadata | `epochs_trained`, `selected_epoch`, and `checkpoint_epoch` all equal N; selection is `final_fixed_epoch` | T1 PASS |
| Early stop metadata | Separates epochs trained from selected checkpoint epoch | T1 PASS |
| Run identity | One canonical hash includes dataset, model, fold, seed, protocol, stage policy, config hash, and training mode | T2 PASS |
| Inner→outer budget | Median of positive inner-only selected epochs; missing corrected inner evidence raises instead of silently using four | T4/T7 PASS |
| Concat auxiliary heads | Heads use authoritative `representation_dim` | T6 PASS |
| Gated model | Representation remains 64 and parameter count remains 150,202 | T5 PASS |
| Loss | Weighted BCE + 0.15 survival + 0.15 outcome; zero aux restores risk-only | T11 PASS |
| Provenance | Requested/executed/checkpoint/strategy are explicit | T12 PASS |

New checkpoints created by the corrected path serialize `config_version`,
`config_hash`, `architecture_hash`, parameter count, training mode, trained
epochs, selected/checkpoint epoch, and pretraining provenance. Legacy frozen
checkpoints remain read-only and were not migrated in place.
