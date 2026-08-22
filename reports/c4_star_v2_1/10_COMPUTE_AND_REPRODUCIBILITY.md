# 10 Compute and reproducibility

## Machine

i5-10400F 6c/12t, 16 GB RAM, RTX 2060 6 GB, Windows. Torch 2.11+cu128 AMP FP16.

## Thermal policy (user: max 80°C)

- Hard cap **80°C** GPU (`wait_if_hot`): pause until 74°C; soft sleep from 76°C.
- `nvidia-smi -pl` **failed** (Insufficient Permissions). No overclock. No disabled protections.
- Process priority **AboveNormal** (not High) to reduce extra heat.
- CatBoost/XGB GPU never concurrent with PyTorch.
- Heartbeat every 10 minutes in `OVERNIGHT_STATUS.md`.

## Resume

```bash
py -3.10 -u -m experiments.c4_star overnight
```

State: `artifacts/research/c4_star_v2_1/state.json`. Optuna studies resume from PostgreSQL.

## Protocol

`c4_star_v2.1` hash `ce758268ce0c834624a76f847864e4f31f553d85d1bb6458d453d07b8f8ee9ac`. Outer splits not regenerated.
