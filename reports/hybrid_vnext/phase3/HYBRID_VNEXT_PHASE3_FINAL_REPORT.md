# HYBRID VNEXT PHASE 3 — FINAL REPORT

**Status:** `NOT_READY_FOR_FINAL_EVAL`

Topology remained **C0**: parallel CNN ∥ BiLSTM, corrected availability, 3-way masked softmax, binary risk head.
Outer labels were not used for HPO, threshold, seed, architecture, or calibration decisions.
Mode: `FAST_COMPLETION` (no new HPO after locked best; remaining started 3×3 only; temperature once; bootstrap 1000 if outer opened).

## A. Executive conclusion

`NOT_READY_FOR_FINAL_EVAL`

- OULAD Hybrid robust macro PR-AUC = `0.8275` ± `0.0056` vs strongest inner baseline `XGB_robust` = `0.8273` (Δ `0.0003`, positive stages `2`, ok=`False`).
- UCI Hybrid robust macro PR-AUC = `0.7288` ± `0.0241` vs `RF_robust` = `0.7320` (Δ `-0.0033`, ok=`True`).
- Inner ready: `False`
- Outer opened: `False`
- Authority `src/prediction` updated: `False`
- Stop reason: OULAD is only +0.00027 vs XGB and positive on **2/4** stages (need ≥3). Outer was not opened.

## B. Architecture lock

- topology_hash: `b2377c362eeca6deba96903cc5a9375ac05e4f9110767583ae6539490ca1b08f`
- temporal_path = parallel; fusion = softmax_3way; public class = Hybrid
- Phase 2 SELECTED_TOPOLOGY / PROTOCOL_LOCK hashes verified before training
- Availability unit tests passed (S0/no-temporal mass = 0; BiLSTM not gated by aggregate)
- No dataset-specific fork; no C0 topology change; no post-outer retune

## C. Shared structural HPO

- Selected shared tuple: `128/64/128`
- Shared across UCI and OULAD: `True`
- Reason: `lexicographic_oulad_then_uci_guardrail`
- Screened `{64,96,128}^3` then confirmed the selected tuple with 3×3. No further structural search in FAST_COMPLETION.

## D. OULAD training HPO

- Complete trials: `10`; pruned: `25`
- Best 1-fold macro: `0.8359` (saturated at the first complete trial; no material later gain)
- Locked numerics: `{"batch_size": 128, "dropout": 0.31959818254342154, "entropy_floor_coefficient": 0.005, "lr": 0.00011844319751820385, "pos_weight_multiplier": 0.7790418060840998, "weight_decay": 0.0007114476009343421}`
- No additional HPO trials were launched after the locked best.

## E. UCI training HPO

- Complete trials: `25`; pruned: `3`
- Best 1-fold macro: `0.7103`
- Locked numerics: `{"batch_size": 32, "dropout": 0.4061978796339918, "entropy_floor_coefficient": 0.002, "lr": 8.605034792033103e-05, "pos_weight_multiplier": 1.1830880728874675, "weight_decay": 0.0032859708169642424}`
- Same C0 graph; only training numerics differ from OULAD.

## F. Overfit / variance analysis

- OULAD 3×3: mean `0.8275`, std `0.0056`, min `0.8222`, worst-stage mean `0.7649`, generalization-gap mean `0.0271`, median best epoch `29.0`
- UCI 3×3: mean `0.7288`, std `0.0241`, min `0.6912`, worst-stage mean `0.4547`, generalization-gap mean `0.0603`, median best epoch `14.0`
- OULAD stage means: `{"20pct": 0.7648925424627283, "35pct": 0.8073284349736313, "50pct": 0.8490449134415323, "75pct": 0.8888305711136967}`
- UCI stage means: `{"S0": 0.454744821940755, "S1": 0.8214149119441972, "S2": 0.9101038055944977}`
- Seeds were never selected; all three seeds remain in the confirmation pool.

## G. Gate diagnostics

See `GATE_DIAGNOSTICS.csv`. UCI S0 must keep tabular_mass=1 and zero CNN/BiLSTM mass when temporal is absent.
OULAD temporal mass increases with later prefixes, consistent with corrected availability.

## H. Baseline fairness

Fixed Phase-2 strong configs, same cutoff-safe parity features, same FIT/STOP/VALID 3×3. No extra baseline HPO.

```json
{
  "oulad": {
    "DT_robust": 0.7778078195140466,
    "DT_robust_std": 0.0060595942329471925,
    "LR_robust": 0.8211406153784279,
    "LR_robust_std": 0.005307021704896947,
    "MLP_robust": 0.7670157556135968,
    "MLP_robust_std": 0.005544575178855857,
    "RF_robust": 0.8177495330488171,
    "RF_robust_std": 0.004189666136932686,
    "XGB_robust": 0.8272537188232377,
    "XGB_robust_std": 0.0050015270694077725
  },
  "uci": {
    "DT_robust": 0.6682073502839286,
    "DT_robust_std": 0.020551309019107542,
    "LR_robust": 0.7120087568345516,
    "LR_robust_std": 0.03411219566596775,
    "MLP_robust": 0.6953138062473472,
    "MLP_robust_std": 0.047449650341875635,
    "RF_robust": 0.7320438916039742,
    "RF_robust_std": 0.03550767348748176,
    "XGB_robust": 0.7187061357061777,
    "XGB_robust_std": 0.033550045261749026
  }
}
```

## I. Inner acceptance gate

```json
{
  "oulad": {
    "best_baseline": 0.8272537188232377,
    "best_baseline_name": "XGB_robust",
    "delta": 0.0002703966746593345,
    "hybrid": 0.8275241154978971,
    "ok": false,
    "positive_stages": 2,
    "stage_delta": {
      "20pct": -0.0014384866185427336,
      "35pct": 0.0007856074663922241,
      "50pct": 0.003062021346723509,
      "75pct": -0.0013275554959353286
    }
  },
  "outer_test_used": false,
  "ready": false,
  "reason": "OULAD Hybrid macro PR-AUC is only +0.00027 vs XGB 3x3 and is positive on 2/4 stages (need >=3). Protocol forbids opening outer. UCI guardrail would have passed (Hybrid 0.7288 vs RF 0.7320, within 0.005).",
  "status": "NOT_READY_FOR_FINAL_EVAL",
  "authority_updated": false,
  "uci": {
    "best_baseline": 0.7320438916039742,
    "best_baseline_name": "RF_robust",
    "delta": -0.0032893784441575535,
    "hybrid": 0.7287545131598167,
    "ok": true,
    "std": 0.024140540307607195
  }
}
```

## J. Threshold / calibration

- Threshold policy: `STOP-only F1 then recall then |t-0.5|` on the existing 0.05–0.95 / 0.01 grid. No finer grid search.
- Thresholds: `{"oulad": {"20pct": 0.29, "35pct": 0.32, "50pct": 0.34, "75pct": 0.47}, "uci": {"S0": 0.46, "S1": 0.88, "S2": 0.27}}`
- Temperature scaling tested once (`used=False`). Other calibrators were not tried.
- T=`1.109`; ECE `0.0991` → `0.1086` (gain `-0.0094`); Brier `0.1288` → `0.1274` (gain `0.0013`).
- Decision: `no_clear_ece_brier_gain_dropped`

## K. Final outer results

Outer evaluation was **not** opened. Inner gate did not pass, or lock was not written.

## L. Hybrid vs strongest baseline

Final acceptance was not computed because outer was not opened.

## M. Paired bootstrap

Bootstrap was not run (no outer predictions).

## N. FINAL-100 shortcut analysis

FINAL-100 was diagnostic only. It was not used for structural HPO, training HPO, threshold, or lock.
Summary: `{"hpo": false, "used_for_selection": false}`

## O. Leakage / provenance audit

- `outer_test_used=false` on HPO, robust confirmation, inner baselines, threshold, and calibration
- split hashes inherited from Phase 2 PROTOCOL_LOCK
- FIT-only preprocessing unchanged
- no best-seed selection; no post-outer retune
- FAST_COMPLETION flag present: `True`

## P. Final authority decision

`src/prediction` is updated only if status is FINAL_HYBRID_ACCEPTED.
Current authority update: `False`.

## Q. Remaining limitations

- UCI T≤2 limits temporal inductive advantage versus trees on aggregate/grade features
- FINAL-100 length≈Withdrawn shortcut remains; not used for acceptance
- C0 softmax can down-weight tabular on UCI S1/S2
- HPO batch sizes were locked (OULAD 128 / UCI 32); they were not enlarged after selection because that would change the locked training numerics
- AMP kept; DataLoader pin_memory/workers were not introduced after a previous host-side hang on this Windows box

