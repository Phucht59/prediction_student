# FINAL_DECISION

NOT_READY_FOR_DEFENSE

Chương trình `hybrid_superiority_v2` **không** mutate `reports/CURRENT_REPORTS.md` và **không** promote Hybrid serving. Confirmation **từ chối** vì development gate không `pass=true` trên mọi warm stage.

SPEED_FINISH (wall 833s, RTX 2060 HIGH priority) đã hoàn tất lock OULAD + Hybrid C0-R + gate + ablation UCI. Đây **không** phải budget preregister 28-trial / 3×3 OULAD — xem `SPEED_FINISH.md`.

## 1. Lineage

| Field | Value |
|---|---|
| Time | 2026-08-20T22:58:45Z |
| Branch | `research/hybrid-superiority-v2` |
| Scaffold commit | `ae883396b15294a075aecf47cd8c70998cd213f1` |
| Protocol | `hybrid_superiority_v2.0` |
| Protocol hash | `eb5f4cfbf4e1629281386367400970241ff68fdaec6c0f7905b0e3a6f33646a2` |
| Outer test used for selection | `false` |
| Serving authority | không đổi |

## 2. Candidate

Public class `SuperiorityHybrid`. Ladder C0-R / C1-R / C2-S / C3-G.

- UCI screen fold-0 J: C0-R −3.15, C3-G −5.12, C1-R −5.88, C2-S −7.81 → survivor **C0-R**
- OULAD SPEED: chỉ C0-R (6 trial, 10 epoch, batch 512)

## 3. Hybrid vs trần baseline (AP)

### UCI — lock 3 fold × 3 seed, trần CatBoost

| | S0 | S1 | S2 |
|---|---:|---:|---:|
| CatBoost (ceiling) | 0.501 | 0.769 | 0.907 |
| C0-R 3×3 | 0.461 | **0.811** | **0.913** |
| Δ vs CatBoost | −0.040 | **+0.041** | +0.006 |
| Material cần | cold ≤0.05 | 0.023 | 0.010 |
| Gate | cold pass | **material pass** | **material fail 0.004** |

### OULAD — SPEED lock fold 0 × 2 seed

| | 20pct | 35pct | 50pct | 75pct | 100pct |
|---|---:|---:|---:|---:|---:|
| Ceiling | LR 0.768 | LR 0.809 | XGB 0.856 | LR 0.899 | XGB 0.926 |
| C0-R 3 seed | 0.761 | 0.809 | 0.858 | 0.897 | 0.923 |
| Δ | −0.007 | +0.0002 | +0.0013 | −0.002 | −0.003 |
| Material cần | cold ≤0.02 | 0.019 | 0.014 | 0.010 | 0.010 |
| Gate | cold pass | pos, **not material** | pos, **not material** | **thua** | **thua** |

Hybrid OULAD bám sát trần tree/linear (Δ trong ±0.003 trên warm) nhưng **không** vượt material margin.

## 4–8. Gate / ablation / shortcut

- Combined development gate: **FAIL**
- UCI: S2 thiếu 0.004 AP so với material 0.010
- OULAD: 4 warm stage fail material
- Confirmation: không mở
- Ablation SPEED UCI 8 epoch fold-0: `serial_no_tabular` S2=0.904 (G1/G2 temporal); `full` 8-epoch undertrain AP~0.32 — **không** so với C0-R 3×3 0.913. Xem `ABLATION.md`
- OULAD 100% operational: 22522 records, **94 Withdrawn**. Length→Withdrawn là sensitivity, không phải early-warning
- Bootstrap 10 000 / Holm: không chạy confirmation vì gate fail

## 9. Reproduce

```bash
python -m experiments.hybrid_superiority_v2 audit
python -m experiments.hybrid_superiority_v2 prepare --dataset all
python -m experiments.hybrid_superiority_v2 baselines --resume
python -m experiments.hybrid_superiority_v2 diagnose --candidate C0-R
python -m experiments.hybrid_superiority_v2 optimize --candidate C3-G --resume
python -m experiments.hybrid_superiority_v2 confirm --frozen-protocol eb5f4cfbf4e1629281386367400970241ff68fdaec6c0f7905b0e3a6f33646a2
python -m experiments.hybrid_superiority_v2.fast_finish
```

## 10. Gemini

Quota code có. HPO dự đoán **không** gọi Gemini. Weak label ≠ expert gold.

## 11. Claim được phép / cấm

**Cấm hiện tại:** viết Hybrid serving vượt trội; giấu SPEED_FINISH; gọi AP là PR-AUC; OULAD 100% là early-warning; roster không XGB/CatBoost.

**Được phép:** protocol đã khóa; XGB/CatBoost trong roster; AP primary; G1/G2 không vào tabular Hybrid; UCI C0-R thắng CatBoost S1 material nhưng **fail S2 material**; OULAD C0-R không thắng trần; **NOT_READY_FOR_DEFENSE**.

## 12. Files

`SPEED_FINISH.md`, `BASELINE_CEILING.md`, `DEVELOPMENT_GATE.md`, `ABLATION.md`, `STATS.md`, `THESIS_READY_TABLES.md`, `00_SOURCE_AND_SCOPE_AUDIT.md`.
