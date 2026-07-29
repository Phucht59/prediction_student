# Phase 1 — Configuration and Provenance

## Conclusion

Status: **PROVENANCE MISMATCH; single source of truth not established**.

`configs/final/cnn_bilstm_oulad.yaml` describes the official single-cutoff
model. `configs/final/oulad_prediction.yaml` and hardcoded
`_deep_config()` describe the unified stage-aware checkpoints. Both use the
same public model ID, which makes fields appear contradictory unless protocol
authority is stated explicitly.

## Field comparison

| Field | Canonical config | Selected evidence | Training manifest | Actual unified checkpoint | Status |
| --- | --- | --- | --- | --- | --- |
| Pretraining | `P1_MASKED_AND_NEXT_WEEK` | null / frozen default | prohibited | no pretrained state/metadata | **BEHAVIOR MISMATCH** |
| Augmentation | unspecified | none | synthetic resampling prohibited | none | CONSISTENT |
| Kernels | multi-kernel, values absent | frozen default | frozen default | `[2,3,5]` | CONSISTENT |
| Dilation | absent | frozen default | frozen default | default `1` | UNKNOWN provenance |
| Conv channels | absent | frozen default | frozen default | 32 | UNKNOWN provenance |
| LSTM hidden/layers | absent | frozen default | frozen default | 64 / 1 | UNKNOWN provenance |
| Pooling | masked, type absent | frozen default | frozen default | masked mean+max | CONSISTENT behavior |
| Fusion | gated residual | frozen default | frozen default | gated residual | CONSISTENT |
| Dropout / branch dropout | absent | 0.20 / implicit | frozen default | 0.20 / 0.10 | partial UNKNOWN |
| Optimizer | absent | absent | absent | AdamW from source | PROVENANCE MISMATCH |
| LR / WD | absent | 0.0005 / 0.00001 | frozen default | same | CONSISTENT |
| Loss | absent | aux weights only | absent | weighted BCE + survival + outcome | PROVENANCE MISMATCH |
| Aux weights | 0.15 / 0.15 | 0.15 / 0.15 | absent | 0.15 / 0.15 | CONSISTENT |
| Epochs | absent | max 4 | selected 1 | executed fixed 4, metadata 1 | PROVENANCE MISMATCH |
| Patience | absent | 2 | absent | 2, unused in outer refit | CONSISTENT value |
| Threshold | fold 0.455/0.495/0.500 | fold+stage inner OOF | separate CSV | not in checkpoint | **BEHAVIOR MISMATCH** |
| Parameters | 100,938 | audit says 150,234 | 150,202 | 150,202 | PROVENANCE MISMATCH |

The `architecture_freeze_audit.json` count of 150,234 instantiates
`static_dim=14`; actual checkpoints record `static_dim=13` and 150,202
parameters. The 32-parameter difference is exactly one additional input into
the first 32-unit static layer.

## Run identity mismatch

All 45 deep unified payload `training_run_id` values differ from their manifest
IDs because two different hash payloads are used. Checkpoint paths and SHA-256
values remain correct, so same-checkpoint stage mapping is unaffected.

## Required Phase 2 provenance action

Create an explicit versioned unified config that contains every architecture,
training, objective, epoch, preprocessing, and threshold field; record its
hash in both payload and manifest; retain the official single-cutoff YAML as a
separate authority. Do not mutate existing frozen artifacts.

Full structured rows are in
`artifacts/audit/phase1/config_provenance.json`.
