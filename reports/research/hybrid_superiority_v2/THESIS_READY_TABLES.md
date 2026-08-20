# Thesis-ready tables (research namespace)

Các bảng dưới đây **không** thay `reports/prediction/final/*` cho đến confirmation.

Protocol: `hybrid_superiority_v2.0`  
Hash: `eb5f4cfbf4e1629281386367400970241ff68fdaec6c0f7905b0e3a6f33646a2`

Primary metric: AP (`average_precision_score`).

## Baseline ceiling

Điền sau `baseline_lock_uci.json` / `baseline_lock_oulad.json`. Roster: LR, DT, RF, SVM, XGB, CatBoost, MLP.

## Hybrid vs max baseline

Warm stages bắt buộc thắng với material margin. Cold: S0/20% được phép thua trong guardrail.

## Ghi chú N / prevalence

UCI Combined: 1044 records, 662 groups, 366 MAT∩POR, prevalence 0.2203 (230 risk).  
OULAD: điền sau prepare (32593 enrollments / 28785 students ở bước static).
