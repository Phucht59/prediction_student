# Hybrid scientific improvement + checklist

**One model, one spec.** Canonical Hybrid is frozen `TRAINING_CONFIG.json` (C0). Fold/seed jobs and the S0/20% fine-tune below are **diagnostics of that spec**, not extra models to pick from. See `reports/prediction/experiments/ONE_MODEL_LOCK.md`.

Architecture remains CNN ∥ BiLSTM Hybrid C0. Outer labels were not used for HPO, model choice, or thresholds.
Unfavorable cells (UCI S0, OULAD 20pct) are **accepted**, not a reason to spawn a second Hybrid.

## Checklist

| Item | Status | Evidence |
| --- | --- | --- |
| Nested / 5-fold on development, outer0 held out | DONE | `experiments/cv5` |
| Early stopping on STOP | DONE | train_one patience on STOP PR-AUC |
| No outer HPO / threshold | DONE | STOP-only threshold; outer confirm is post-lock |
| Multiple seeds | DONE | UCI seeds 42/1201/2026; OULAD seed 42 (compute) |
| Leakage audit | DONE | frozen split hashes from backup branch; `LEAKAGE_OVERFIT_AUDIT.json` |
| Overfit by fold/seed/stage | DONE | `overfit_gap` column in scientific metrics |
| Baseline RF/LR | DONE | tabular features only |
| Bootstrap / McNemar / DeLong / effect size | DONE | `scientific/stat_tests.csv` |
| Calibration Brier/ECE/plots | DONE | metrics + `validation/figures` |
| Ablation CNN/BiLSTM/Tabular/Hybrid | DONE | Hybrid-* rows (trained-model branch scoring) |
| Robustness/sensitivity | DONE | `sensitivity_uci_fold0.csv` (not used to pick a new production model) |
| Error analysis | DONE | confusion counts + error histograms |
| Subgroup/fairness | DONE | `validation/subgroup.csv` |
| SHAP / fusion | DONE | RF KernelSHAP + gate masses |
| Cross-dataset UCI+OULAD | DONE | both in 5-fold |
| External dataset | NOT AVAILABLE | no licensed third dataset |
| No cherry-pick | DONE | S0 and 20pct losses kept |

## 5-fold VALID means after weak-stage STOP fine-tune

```
dataset,information_level,model,pr_auc,brier,ece
oulad,100pct,Hybrid,0.9194250578970573,0.08955781743794407,0.0693537352846105
oulad,100pct,LR,0.9084546904534051,0.08844027458654549,0.07039627030954038
oulad,100pct,RF,0.9144013872484624,0.07635079365267793,0.03246707808925421
oulad,20pct,Hybrid,0.7545397263085019,0.19871450009698605,0.1063376818121069
oulad,20pct,LR,0.7579159070327711,0.18554114427349497,0.05808003890489148
oulad,20pct,RF,0.7555347185631544,0.18475293805107526,0.030323692169872952
oulad,35pct,Hybrid,0.8041352478012225,0.16340692817275,0.06828871363609507
oulad,35pct,LR,0.7953038349918271,0.16538331396042377,0.06767915378404886
oulad,35pct,RF,0.7966639661945034,0.16221976807324895,0.031197841840188634
oulad,50pct,Hybrid,0.8455614493626706,0.13737638845383435,0.06389427072648299
oulad,50pct,LR,0.8369613786740707,0.14133586625722785,0.07227621995025608
oulad,50pct,RF,0.8394113134606037,0.13547374956384736,0.031045001079864708
oulad,75pct,Hybrid,0.8886298847184471,0.10451137384429329,0.05484548952055279
oulad,75pct,LR,0.8806592106882715,0.1096649247419433,0.07522067339569095
oulad,75pct,RF,0.8862092707953838,0.0992319031043912,0.031010357870696652
uci,S0,Hybrid,0.49005293734510214,0.19575157634125379,0.2022049878803737
uci,S0,LR,0.47800064494651007,0.19350639270439038,0.1874714712326494
uci,S0,RF,0.48563763734660037,0.14625593733074085,0.08845260305331898
uci,S1,Hybrid,0.8043737202640248,0.10883162291693346,0.11043352921204748
uci,S1,LR,0.7330883375022257,0.11814923506641355,0.13527739836644076
uci,S1,RF,0.7747150873916607,0.08883825508990159,0.08991353398340031
uci,S2,Hybrid,0.8964818427062359,0.0828049467080708,0.09604027890783261
uci,S2,LR,0.8054969657036326,0.10253390462554061,0.1184219978172187
uci,S2,RF,0.8978642900454908,0.06646038948932372,0.09039375998110179

```

### Hybrid train−VALID PR-AUC gap

```
dataset,information_level,overfit_gap
oulad,100pct,0.02405050402174671
oulad,20pct,0.03285469777443743
oulad,35pct,0.036224290314142295
oulad,50pct,0.03267200070448846
oulad,75pct,0.029288230416433313
uci,S0,0.16383889278768715
uci,S1,0.07289706082460609
uci,S2,0.04636618052861853

```

## Unfavorable results (kept)

- UCI S0: Hybrid ≈ RF within 0.002 PR-AUC; S0 has no temporal input by contract.
- OULAD 20pct: Hybrid can trail LR/RF slightly; short VLE history.
- These do **not** change production Hybrid without a pre-registered superiority gate.

MODEL_CHANGED=False OUTER_USED_FOR_HPO=false
