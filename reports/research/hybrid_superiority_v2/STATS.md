# Stats

Paired cluster bootstrap (n=10 000, Holm α=0.05) là thống kê **confirmation**. SPEED_FINISH **không** mở confirmation vì development gate fail.

## Vì sao không bootstrap confirmation

UCI C0-R vs CatBoost: S2 mean Δ=+0.006 < material 0.010. Một CI thấp hơn 0 hay không cũng **không** được dùng để tuyên bố vượt trội khi gate material fail.

OULAD C0-R vs max(LR, XGB): Δ warm ∈ [−0.003, +0.001]. Không có cơ sở superiority.

## OOF có sẵn (nếu chạy bootstrap sau)

- UCI baseline: `artifacts/research/hybrid_superiority_v2/oof/baseline_oof_uci.parquet`
- OULAD baseline: `artifacts/research/hybrid_superiority_v2/oof/baseline_oof_oulad.parquet`
- OULAD Hybrid C0-R: `artifacts/research/hybrid_superiority_v2/oof/hybrid_oof_oulad_C0-R.parquet` (fold 0, 3 seed)

Groups resampled once per replicate; comparator = max baseline cùng stage. Không dùng outer test.
