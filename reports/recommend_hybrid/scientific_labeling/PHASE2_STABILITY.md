# Phase 2 stability

Each dataset has a separate Snorkel Label Model fitted on its train rows only. Stability artifacts record five fixed seeds (`42`, `1201`, `2026`, `3407`, `7319`) on deterministic train resamples and one-family-at-a-time ablations. The largest observed sampled family-ablation label change was `0.052`.

The canonical seed remains `2026`; alternate seeds are diagnostics and were not selected for a favorable outcome. No single ablation exceeded the project warning threshold of `0.10`.
