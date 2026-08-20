# 02 — Hardware and performance audit

Nguồn: `artifacts/research/hybrid_superiority_v2/manifests/hardware_manifest.json` (không ghi secret).

## Máy

| Hạng mục | Giá trị |
|---|---|
| OS | Windows 10.0.26220 |
| CPU | Intel Core i5-10400F, 6 nhân / 12 luồng |
| RAM | 15.86 GB |
| Disk free | ~50 GB |
| GPU | NVIDIA GeForce RTX 2060, 6.0 GB, CC 7.5 (Turing) |
| Driver/CUDA | NVIDIA-SMI 610.47, CUDA UMD 13.3; PyTorch CUDA runtime 12.8 |
| Python | 3.10.0 |
| torch | 2.11.0+cu128 |
| cuDNN | 91900 |
| sklearn | 1.7.2 |
| Optuna | 4.8.0 |
| XGBoost | 3.2.0 |
| CatBoost | 1.2.10 |
| PostgreSQL | 18.4 @ localhost:5432, db `student_db` |
| Docker | không có |
| LightGBM | không cài (không trong roster chính) |

Luồng: `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`. Hybrid fail-fast nếu CUDA tắt. Baseline chạy CPU.

Env names hiện diện (không log value): `GEMINI_API_KEY`, `DB_*` qua `.env` gitignored, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`.

## Microbenchmark matmul 2048²

| Mode | seconds/iter | TFLOPS |
|---|---:|---:|
| FP32 | 0.00380 | 4.52 |
| AMP FP16 | 0.00089 | 19.34 |
| AMP BF16 | 0.00627 | 2.74 |

Chọn **AMP FP16 + GradScaler**. TF32 không dùng (Ampere+). BF16 trên Turing chậm hơn FP16.

`torch.compile` sẽ được probe trên model thật; BiLSTM packed sequence dễ graph-break — fallback eager nếu không nhanh hơn ≥5%.

## Batch / VRAM

Mục tiêu 80–90% của 6 GB. Probe powers-of-two, OOM thì giảm batch, không đổi hyperparameter khác. Channels/hidden ưu tiên bội số 8.

## PostgreSQL

Migrate `009_research_hybrid_superiority_v2.sql` idempotent: schema `research`, `optuna_hs_v2`, `recommendation`. Không drop bảng serving `raw/catalog/prediction/recommendation`.

## Quyết định train

- Hybrid: `cuda:0`, AMP FP16, `zero_grad(set_to_none=True)`, pin memory khi dataloader dùng.
- Confirmation: `cudnn.deterministic`, không `benchmark`.
- HPO: `cudnn.benchmark` OK.
- Một GPU một worker train nặng.
