# Benchmark V2 compute estimate

## Smoke run

`smoke-v2-20260713` ran one outer fold, one evaluation seed, and one CNN-tuning trial. Wall time was **47.2 seconds** on the current workstation. It exercised DB-first load, shared manifest membership, basic baselines, MLP, CNN-only, BiLSTM-only, legacy CNN–BiLSTM, and one 3-inner-fold CNN tuning trial.

## Full-run estimate

The full V2 plan has five outer folds, five evaluation seeds for neural models, and 30 CNN–BiLSTM tuning trials per outer fold. The dominant cost is the tuned CNN: approximately 150 trials × 3 inner fits plus 25 outer refits. A conservative linear extrapolation from smoke is **35–45 minutes**, plus roughly 5–10 minutes for the remaining multi-seed neural refits. Checkpoint storage is expected to remain below 100 MB because models are small.

No fold, inner-fold count, seed list, or trial budget will be reduced. The full job is therefore launched as one immutable benchmark run; partial outputs are not interpreted or ranked.
