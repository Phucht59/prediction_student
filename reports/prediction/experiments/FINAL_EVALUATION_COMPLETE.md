# Final evaluation complete — one Hybrid CNN–BiLSTM

**Status:** complete for this phase.  
**Production model:** unchanged.  
**Git push:** not done.  
**Rule:** one architecture, one frozen spec. Folds/seeds evaluate that spec; they are not extra models.

Canonical numbers are the **5-fold development CV, seed 42, frozen `TRAINING_CONFIG.json`**, not the S0-fine-tune diagnostic suite.

## Locked spec

| Field | Value |
| --- | --- |
| Public model | Hybrid CNN–BiLSTM |
| `architecture_id` | C0 |
| Widths | d_fuse 128, CNN 64, BiLSTM 128 |
| Strategy | L1_control |
| Hparams | `artifacts/prediction/final/TRAINING_CONFIG.json` |
| UCI vs OULAD | same topology; lr/batch/dropout differ by dataset scale only |
| Stages | views of one fitted instance each |

See `ONE_MODEL_LOCK.md`.

## Protocol

- FIT / STOP / VALID; early stop and threshold from STOP only.
- Official outer fold 0 excluded from 5-fold train/STOP/VALID.
- Split parquets restored from `origin/codex/backup-hybrid-phase8-2026-08-17`; hashes verified.
- LR/RF: tabular `static∥aggregate∥progress`. Hybrid: ordered temporal via CNN∥BiLSTM.
- No outer HPO. No cherry-pick of seed or fold.

## Canonical 5-fold PR-AUC (seed 42)

| Dataset | Level | Hybrid | LR | RF | Hybrid vs best baseline |
| --- | --- | ---: | ---: | ---: | --- |
| UCI | S0 | 0.479 | 0.478 | 0.481 | **−0.002 accepted** |
| UCI | S1 | **0.804** | 0.733 | 0.777 | +0.027 |
| UCI | S2 | **0.895** | 0.805 | 0.894 | +0.002 |
| UCI | **macro** | **0.726** | 0.672 | 0.717 | **+0.009** |
| OULAD | 20% | 0.752 | 0.758 | 0.756 | **−0.006 accepted** |
| OULAD | 35% | **0.805** | 0.795 | 0.797 | +0.008 |
| OULAD | 50% | **0.846** | 0.837 | 0.839 | +0.007 |
| OULAD | 75% | **0.890** | 0.881 | 0.886 | +0.004 |
| OULAD | 100% | **0.923** | 0.908 | 0.914 | +0.008 |
| OULAD | **macro** | **0.843** | 0.836 | 0.838 | **+0.005** |

UCI outer fold 0 confirm (same spec, not HPO): S0 0.500, S1 0.792, S2 0.914.

## Checklist (all closed)

| Item | Result |
| --- | --- |
| Nested / 5-fold, outer held out | Done |
| Early stopping | Done (STOP PR-AUC) |
| No outer HPO / model / threshold | Done |
| Multiple seeds | Done (locked Phase-4 3×3; UCI 5-fold also 42/1201/2026 as diagnostic) |
| Leakage audit | `LEAKAGE_FREE=true`, hashes match backup |
| Overfit by stage | Locked S0 HIGH; OULAD LOW; 5-fold gaps recorded |
| Baseline RF, LR | Done |
| Bootstrap CI, McNemar, DeLong, effect size | Done |
| Brier, ECE, calibration plots | Done (84 calib/CM + 56 summary figures) |
| Ablation CNN / BiLSTM / Tabular / Hybrid | Done (branch-only of trained Hybrid) |
| Sensitivity | Done; **not used for selection** |
| Error analysis FP/FN | Done |
| Subgroup / fairness | Done |
| SHAP / fusion gates | Done |
| Cross-dataset UCI + OULAD | Done |
| External dataset | **NOT AVAILABLE** (none invented) |
| Unfavorable results kept | S0, 20% accepted |
| One model | Locked |

## Artifacts

- `artifacts/experiments/cv5/hybrid_vs_baselines.csv` — canonical comparison
- `artifacts/experiments/validation/LEAKAGE_OVERFIT_AUDIT.json`
- `artifacts/experiments/validation/figures/` and `plots/`
- `artifacts/experiments/cv5/scientific/` — stats, sensitivity, outer confirm (diagnostic)
- Reports in `reports/prediction/experiments/`

## Conclusion

One Hybrid CNN–BiLSTM, one frozen parameter spec, beats LR/RF on **dataset macro PR-AUC** for UCI and OULAD. UCI S0 and OULAD 20% slightly lower is **accepted**. Nothing here promotes a second Hybrid or changes production weights.
