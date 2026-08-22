# 00 INPUT AUDIT — C4-STAR v2.1

Do not treat prior Markdown as ground truth. Key tables were recomputed from OOF.

## Git / worktree

| Field | Value |
|---|---|
| Branch | `research/hybrid-superiority-v2` |
| HEAD (scaffold) | `ae883396b15294a075aecf47cd8c70998cd213f1` |
| Parent authority commit (untouched serving) | `0cb02479154a734240b55bf5525a96e11a72e863` |
| Worktree | **dirty** (v2 SPEED_FINISH + C4-STAR files) |
| Serving Hybrid | not mutated |

## Protocols

| Protocol | Hash |
|---|---|
| `hybrid_superiority_v2.0` (parent) | `eb5f4cfbf4e1629281386367400970241ff68fdaec6c0f7905b0e3a6f33646a2` |
| `c4_star_v2.1` | `ce758268ce0c834624a76f847864e4f31f553d85d1bb6458d453d07b8f8ee9ac` |

Amendment: joint-domain candidate selection. Outer splits **not** regenerated.

## Locked splits (must remain)

| Domain | outer sha256 | inner sha256 |
|---|---|---|
| UCI | `4bf33619395c360442531d396575f42d3dae99e646da3d6418bf1070e8228d0b` | `ce4550ada4b5f6a70cee7525fba1a82ca0c890786e491933e27d9b006e390cca` |
| OULAD | `8ad606ebe805cc0f6c9e742823f8db56122a1d8d6e932caf6d2cf36de09bcbec` | `a83bf6f864227d535bc939b2fb5b780c09868304854afc8222c42df51dd56845` |

## Hardware

- Windows, i5-10400F 6c/12t, ~16 GB RAM
- RTX 2060 6 GB, CC 7.5, CUDA 12.8, torch 2.11+cu128
- AMP FP16 selected
- GPU power limit 125–192 W; `nvidia-smi -pl` **insufficient permissions** → software thermal pause at **80°C**
- PostgreSQL Optuna schema `optuna_hs_v2`

## Verified from raw OOF (mean-of-run AP)

UCI 3×3 CatBoost ceiling: S0 **0.5010** / S1 **0.7694** / S2 **0.9067** — VERIFIED vs lock.

OULAD SPEED fold0×2seed: LR 20% **0.7684**, LR 35% **0.8087**, XGB 50% **0.8563**, LR 75% **0.8989**, XGB 100% **0.9260** — VERIFIED vs lock, **not confirmatory** (truncated HPO).

OULAD C0-R hybrid OOF seeds 42/1201/2026 match SPEED robust JSON.

## Missing / UNVERIFIED

- UCI Hybrid per-row OOF parquet (only robust JSON)
- OULAD C1/C2/C3 joint screen
- OULAD shuffle/reverse diagnostics
- Full OULAD 3×3 baseline
- Outer test records (correctly absent)
- `PHUONG_PHAP_CAI_TIEN_C4_STAR_VA_BUOC_TIEP_THEO.md` is design, not experimental confirmation

## Discrepancies

- Report LR UCI S1 0.7448 vs OOF mean-of-run **0.7417** (CatBoost ceiling unaffected)
- Pooled-row AP ≠ mean-of-run AP; protocol uses mean-of-run
- Ablation `full` AP~0.32 is 8-epoch under-convergence, not a synergy result
- `lambda_kd=0` made SPEED `full_no_kd` a non-ablation

## Reproducible now

Parent v2 pipeline + OOF parquets + split locks. C4-STAR code is new and must pass tests before HPO.
