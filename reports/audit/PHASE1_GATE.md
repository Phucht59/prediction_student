# Phase 1 — Gate

## Status

**PASS**

## Required questions

1. **Why is OULAD `selected_epoch=1`?**

   Fixed outer refit executes four epochs, bypasses best-epoch updates, and
   serializes the initialization value 1. Confirmed metadata bug.

2. **What objective selects checkpoints?**

   Inner diagnostic states maximize mean-stage Macro-F1 at threshold 0.5.
   Final unified outer checkpoints are not selected by validation; they are
   last-state four-epoch refits.

3. **Where is threshold selected and with what data?**

   `_threshold()` receives pooled inner-OOF labels/probabilities per outer
   fold/stage and maximizes recall subject to risk precision ≥0.75.

4. **Is there leakage?**

   No audited record/future/OULAD group/threshold leakage. UCI frozen outer
   folds have quasi-group overlap and are a potential group-safety issue, not
   confirmed identity leakage.

5. **Is preprocessing train-only?**

   Yes for OULAD aggregate/static and UCI context; temporal masks are safe.

6. **Is same-checkpoint-across-stage correct?**

   PASS for all OULAD and UCI mappings.

7. **What is final unified OULAD architecture?**

   47→48 projection; parallel CNN 2/3/5×32; residual; BiLSTM 64×2; masked
   mean+max; 165 aggregate and 13 static branches; two scalar gated residuals;
   risk/survival/outcome heads; 150,202 parameters.

8. **Was final architecture actually Optuna-tuned?**

   No. Historical Optuna tuned a predecessor. Unified uses `frozen_default`.

9. **Is there config/provenance mismatch?**

   Yes: pretraining, parameter count, threshold authority, epoch metadata,
   architecture audit input dimension, and run IDs.

10. **How is hybrid calibration versus ML?**

    Worse, especially early. Hybrid ECE 0.127/0.098 at 20/35% versus HGB
    0.019/0.017.

11. **What advantage do aggregate features give ML?**

    A 161-feature, cutoff-safe summary of totals, moments, extrema, recency,
    slopes, and halves—strong tabular inductive advantage, not leakage.

12. **Should UCI CNN be deepened for length 2?**

    No. Kernel 2 already spans the sequence; S0 has no temporal input.

13. **Is there a latent fusion/multitask bug?**

    Yes. Concatenation returns 3×fusion hidden but auxiliary heads accept 1×.
    Frozen gated residual is unaffected.

14. **What historical architecture gains exist?**

    Capacity +0.0017 over small CNN but still −0.0024 versus BiLSTM; best
    dilation gain about +0.0011; skip/parallel failed the replacement gate.

15. **What must be fixed before Optuna VNext?**

    Epoch/run identity, unified config authority, checkpoint/objective
    propagation, calibration/threshold reporting, concat head dimensions, and
    UCI group-safety protocol wording.

## Preservation gate

| Check | Result |
| --- | --- |
| Baseline preserved | YES |
| Final checkpoints modified | NO |
| Final evidence overwritten | NO |
| Outer labels used for tuning | NO |
| New Optuna run | NO |
| Full final experiment rerun | NO |
| Diagnostic result promoted to official | NO |
