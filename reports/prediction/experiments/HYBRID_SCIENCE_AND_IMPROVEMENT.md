# Hybrid CNN–BiLSTM: improvement + scientific checklist

Production Hybrid **was not modified**. **Not pushed.**

## One model, one spec (scientific lock)

There is **one** public model: Hybrid CNN–BiLSTM (`architecture_id=C0`).

| Locked | Value |
| --- | --- |
| Class | `Hybrid` |
| Topology | Residual CNN ∥ BiLSTM + tabular + 3-way softmax, binary logit |
| Widths | `d_fuse=128`, `cnn_channels=64`, `bilstm_hidden=128` |
| Strategy family | `L1_control` |
| Parameters | `artifacts/prediction/final/TRAINING_CONFIG.json` only |
| UCI vs OULAD numerics | dataset **scale** exception (lr/batch/dropout), **not** a second architecture |
| Stages | views of the **same** fitted instance, not S0-model / 20%-model / 100%-model |

Fold, seed, and STOP checkpoints are **evaluation copies of that one spec**. They are not a menu of models to pick from.

Forbidden: extra HPO, stage-specific networks, “Hybrid-S0-tuned” as production, selecting a seed because it looked better.

Sensitivity / S0 fine-tune jobs are **diagnostics**. They do **not** replace `TRAINING_CONFIG.json` and are not promoted.

## How Hybrid was “optimized higher” without cheating

- Same students, labels, FIT-only preprocess, STOP early-stopping and STOP thresholds.
- Outer fold 0 **never** used for HPO, architecture choice, or threshold.
- Baseline LR/RF trained on **tabular** `static∥aggregate∥progress` only. Hybrid uniquely sees ordered temporal tensors. Flattening sequences into RF is not used (that would copy Hybrid’s sequence view into trees).
- Weak-stage fine-tune (S0 / 20%) updates FIT of that stage only; the checkpoint is still selected on **all-stage STOP** PR-AUC (mean − 0.2·std). If STOP does not improve, the extra updates are discarded.

This is a representation/protocol advantage, not a different prediction task.

## Nested 5-fold / outer

- Development 5-fold: grouped stratified CV, outer fold 0 excluded (`experiments/cv5`).
- Early stopping: STOP macro PR-AUC, patience 8, max 24 (official numerics).
- Outer fold 0 confirmatory scoring is **post-lock** (`scientific/outer_confirm_uci.csv` when present) and is not a search loop.

## 5-fold seed-42 PR-AUC (complete run)

| Dataset | Level | Hybrid | LR | RF | Δ vs best baseline |
| --- | --- | ---: | ---: | ---: | ---: |
| UCI | S0 | 0.479 | 0.478 | **0.481** | **−0.002** (unfavorable, essentially tie) |
| UCI | S1 | **0.804** | 0.733 | 0.777 | +0.027 |
| UCI | S2 | **0.895** | 0.805 | 0.894 | +0.002 |
| UCI | **macro** | **0.726** | 0.672 | 0.717 | **+0.009** |
| OULAD | 20% | 0.752 | **0.758** | 0.756 | **−0.006** (unfavorable) |
| OULAD | 35% | **0.805** | 0.795 | 0.797 | +0.008 |
| OULAD | 50% | **0.846** | 0.837 | 0.839 | +0.007 |
| OULAD | 75% | **0.890** | 0.881 | 0.886 | +0.004 |
| OULAD | 100% | **0.923** | 0.908 | 0.914 | +0.008 |
| OULAD | **macro** | **0.843** | 0.836 | 0.838 | **+0.005** |

Hybrid wins **both macros** and **6/8** stages.

**Accepted (do not keep optimizing these two cells):** UCI S0 is −0.002 vs RF; OULAD 20% is −0.006 vs LR. Both are early states with little or no sequence. They stay in the table. They are **not** a failure of CNN ∥ BiLSTM and **not** a trigger to change the task or the production model.

## Checklist

| Item | Status | Where |
| --- | --- | --- |
| Nested / 5-fold outer held out | **Done** | `experiments/cv5`, outer0 firewall tests |
| Early stopping | **Done** | STOP PR-AUC |
| No outer HPO / model / threshold | **Done** | frozen `TRAINING_CONFIG.json`; STOP thresholds |
| Multiple random seeds | **In progress / partial** | Phase-4 locked 3×3 seeds 42/1201/2026; 5-fold table is seed 42; scientific suite adds UCI 3 seeds |
| Leakage audit | **Done** | Split files restored from `codex/backup-hybrid-phase8-2026-08-17`, hashes match; `LEAKAGE_FREE=true` |
| Overfit by fold/seed/stage | **Done (locked) + in progress (5-fold gaps)** | `OVERFIT_AUDIT.json` (S0 HIGH 0.125); suite writes `overfit_gap` |
| Baseline RF, LR | **Done** | 5-fold table |
| Bootstrap CI | **Done** | `validation/stat_tests.csv`; suite refreshes on 5-fold scores |
| McNemar | **Done** | same |
| DeLong | **Done** | same |
| Effect size + CI | **Done** | Cohen’s g; bootstrap ΔPR-AUC CI |
| Brier / ECE / calibration plot | **Done** | metrics + 56+ figures |
| Ablation CNN / BiLSTM / Tabular / Hybrid | **Done** | trained-Hybrid branch scoring (not a separately retrained net except where suite adds it) |
| Robustness / sensitivity | **Partial → suite** | Optuna DB not re-run as HPO; UCI fold-0 dropout/lr probe, `used_for_selection=false` |
| Error analysis FP/FN | **Done** | confusion + `error_hist_*.png` |
| Subgroup & fairness | **Done** | gender/disability/IMD/module/sex/school |
| Feature importance / SHAP | **Done** | gate masses; KernelSHAP on RF of packed UCI S2 (explains the tabular comparator, not the neural gate) |
| Cross-dataset UCI + OULAD | **Done** | |
| External validation | **NOT AVAILABLE** | no third licensed dataset; none invented |
| No cherry-pick / no fabricated metrics | **Done** | S0 and 20% reported |
| Conclusions from evidence only | **Done** | this file |

## Leakage / overfit (locked, still the authority)

- G3 never a predictor; S0 has no G1/G2.
- OULAD `observation_start ≤ t < cutoff`.
- FIT/STOP/VALID disjoint; groups disjoint; outer fold 0 disjoint from development 5-fold.
- UCI S0 generalization gap **HIGH** (0.125). OULAD gaps **LOW** and shrink with cutoff.
- OULAD 100% length↔Withdrawn confounder remains; not a feature leak, but a validity limit.

## Statistical notes (inner 3×3 VALID, previous pass)

- UCI S2 vs LR: ΔPR-AUC CI above 0 (Hybrid better).
- UCI vs RF at S1/S2: CI includes 0.
- OULAD 35%+ vs RF: small Hybrid ΔPR-AUC (~0.01) with CI above 0.
- p-values are not used to rewrite production.

## What we will not claim

- Hybrid does not dominate RF at UCI S0.
- Hybrid does not dominate LR at OULAD 20%.
- Phase-8 `outer_test_final` is **not** C0 evidence (wrong generation, includes XGB).
- Fine-tuning S0 cannot be declared a production upgrade unless STOP **and** 5-fold VALID both improve without harming S2; that comparison is in `artifacts/experiments/cv5/scientific/`.

## Conclusion

On the pre-registered one-model metric (**macro PR-AUC**), Hybrid CNN–BiLSTM is above LR and RF on both UCI and OULAD (5-fold development CV).

UCI S0 and OULAD 20% being slightly lower is **accepted**. Report them; do not hide them; do not retune the Hybrid until those two cells win. Production Hybrid stays CNN ∥ BiLSTM (`architecture_id=C0`).
