# Phase 4 — Protocol

Phase 4 changed only fusion. The temporal projection, CNN kernels/channels/dilation,
BiLSTM, masks, pooling, aggregate/static inputs, targets, heads, checkpoint objective,
epoch cap, and threshold policy remained frozen.

- Dataset: OULAD unified 20/35/50/75%.
- Stage A: four architectures, three outer-train partitions, seed 42, inner validation only.
- Stage B: A0 plus top two non-controls, seeds 1201 and 2026.
- Checkpoint: minimize mean-stage validation NLL, maximum 15 epochs.
- Research threshold: pooled inner OOF only.
- Outer labels: unavailable to runner and unused.
- Stage conditioning: validly skipped because progress_fraction, observed_week_count, weeks_remaining, assessment_available_fraction already
  provide explicit legal cutoff context.
- Micro-tuning: not triggered.
