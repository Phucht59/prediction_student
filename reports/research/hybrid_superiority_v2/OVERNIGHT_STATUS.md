# OVERNIGHT_STATUS — hybrid_superiority_v2

- Updated: `2026-08-20T22:58:45Z`
- Branch: `research/hybrid-superiority-v2`
- Commit: `ae883396b15294a075aecf47cd8c70998cd213f1`
- Protocol hash: `eb5f4cfbf4e1629281386367400970241ff68fdaec6c0f7905b0e3a6f33646a2`
- Phase: **SPEED_FINISH done — NOT_READY_FOR_DEFENSE**

## Completed

- UCI baseline lock 3×3 (CatBoost trần)
- UCI ladder screen C0-R/C1-R/C2-S/C3-G; robust C0-R + C3-G
- UCI development gate: S1 pass, S2 material fail
- OULAD SPEED lock (XGB/CatBoost GPU 4 trial; skip HPO DT/SVM/MLP/RF)
- OULAD C0-R screen 6 trial + robust 3 seed
- OULAD development gate: fail 4 warm
- UCI SPEED ablation fold-0
- Reports written (no CURRENT_REPORTS promotion)

## Evidence

- `artifacts/research/hybrid_superiority_v2/runs/baseline_lock_uci.json`
- `artifacts/research/hybrid_superiority_v2/runs/baseline_lock_oulad.json`
- `artifacts/research/hybrid_superiority_v2/runs/development_gate.json`
- `reports/research/hybrid_superiority_v2/FINAL_DECISION.md`
- `reports/research/hybrid_superiority_v2/SPEED_FINISH.md`
- Wall SPEED_FINISH = 833s, RTX 2060 HIGH priority

## Decision

**NOT_READY_FOR_DEFENSE.** Không promote Hybrid serving. Confirmation từ chối.

## Next

Đọc `FINAL_DECISION.md`. Nếu muốn protocol đủ 28-trial / 3×3 OULAD thì chạy lại ngoài SPEED_FINISH — không bắt buộc để kết luận hiện tại (gate đã fail).

## Blockers

- Development gate fail (UCI S2 material; OULAD 4 warm)
- SPEED_FINISH cắt budget OULAD — đã document
