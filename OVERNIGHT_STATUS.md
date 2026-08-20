# OVERNIGHT_STATUS — hybrid_superiority_v2

- Updated: in-progress
- Branch: `research/hybrid-superiority-v2`
- Protocol hash: `eb5f4cfbf4e1629281386367400970241ff68fdaec6c0f7905b0e3a6f33646a2`
- Phase: P2 done; P4 UCI baselines running

## Completed

- P0 source/scope audit (đề cương + bài báo; markdown dán missing)
- P1 hardware/DB: RTX 2060 6GB, AMP FP16 19.3 TFLOPS, PostgreSQL 18.4 migrate `research`/`optuna_hs_v2`
- P2 raw→stage: UCI 1044/662; OULAD risk-set 26697→22522; checksums khớp
- Integrity tests: 18 passed
- UCI C3-G smoke 3 epoch: 75k params, 0.035 GB VRAM
- Optuna RDBStorage trên PostgreSQL đang chạy (LR/DT xong, RF đang tune)

## Evidence

- `reports/research/hybrid_superiority_v2/00_SOURCE_AND_SCOPE_AUDIT.md`
- `reports/research/hybrid_superiority_v2/01_DATA_LEAKAGE_AND_COHORT_AUDIT.md`
- `reports/research/hybrid_superiority_v2/02_HARDWARE_PERFORMANCE_AUDIT.md`
- `artifacts/research/hybrid_superiority_v2/manifests/data_lock.json`
- `tests/research/hybrid_superiority_v2` 18 passed

## Decision

Pipeline không phụ thuộc `C:\hufit\kltn`. G1/G2 chỉ temporal Hybrid. OULAD 100% operational gần như loại Withdrawn (94 còn lại) — không dùng length shortcut làm claim.

## Next

1. Đóng băng baseline UCI (XGB/CatBoost trong roster)
2. Diagnose C0-R UCI
3. Screen C0-R…C3-G UCI
4. Baseline + Hybrid OULAD
5. Gate phát triển; confirmation chỉ nếu pass

## Blockers

- `Đã dán markdown (1).md` missing
- Docker không có (dùng PostgreSQL local)
- Confirmation chưa mở (đúng protocol)
- Serving Hybrid authority chưa đụng
