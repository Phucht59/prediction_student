# Phase 1 — Baseline Identity

## Repository identity

| Field | Frozen value |
| --- | --- |
| Repository | `Phucht59/prediction_student` |
| Remote | `https://github.com/Phucht59/prediction_student.git` |
| Baseline branch | `codex/final-unified-project-cleanup` |
| Audit branch | `codex/hybrid-optimization-vnext` |
| Baseline commit | `ead4a76c6901bc3a8def18f617ec64810fb24851` |
| Baseline commit date | `2026-07-30T01:42:20+07:00` |
| Audit date | `2026-07-30` |
| Baseline local/remote divergence before branch creation | `0 / 0` |
| Working tree before branch creation | Clean |

The audit branch was created directly from the baseline SHA after
`git fetch origin --prune`. No history was rewritten and no final checkpoint,
prediction, metric, report, or checksum artifact was changed.

## Runtime identity

| Field | Value |
| --- | --- |
| Operating system | Windows `10.0.26220` |
| Python | `3.10.0` |
| Python executable | `.venv-oulad-v2/Scripts/python.exe` |
| PyTorch | `2.7.1+cu118` |
| CUDA runtime reported by PyTorch | `11.8` |
| CUDA available | Yes |
| cuDNN | `90100` |
| GPU | NVIDIA GeForce RTX 2060, 6,144 MiB |
| NVIDIA driver | `610.47` |

The bare `python` command in the host PATH resolves to a broken Windows app
alias. All audit execution therefore used the project virtual environment
explicitly.

## Dataset artifact identity

| Dataset | Rows / protocol | Frozen artifact identity |
| --- | --- | --- |
| Student-Mat | 395 | `student-mat.csv`, SHA-256 `e47f9ee225e1ee6e69b7564e6dac7123e80b8486677fe111f351964cef5dec80` |
| Student-Por | 649 | `student-por.csv`, SHA-256 `a7594a11d7771c0efe1a740824e0e833da9c4cad07c39a9766a874575563fb3f` |
| OULAD official F2 | `F2_MIDDLE` | Snapshot manifest SHA-256 `8482dbf51c1fe25940a8eef871daeaf07515ac9698fc36b5620263d52b9a1110` |
| OULAD split | 3 frozen outer folds | Split manifest SHA-256 `5e5e4d9ab48a6049f86cdf2451f4302bc44f2896a2f23a23494bc95f549b97d4` |
| OULAD unified | 20%, 35%, 50%, 75% cutoff views | Rebuilt from raw tables with `event_day < cutoff_day`; M1 is the exact F2 cutoff anchor |

## Current final configuration files

- `configs/final/cnn_bilstm_mat.yaml`
- `configs/final/cnn_bilstm_por.yaml`
- `configs/final/cnn_bilstm_oulad.yaml`
- `configs/final/uci_prediction.yaml`
- `configs/final/oulad_prediction.yaml`
- `configs/final/model_registry.yaml`

The official OULAD YAML and the unified OULAD protocol are not interchangeable.
The former identifies the official single-cutoff result; the latter identifies
the separately trained four-stage `frozen_default` checkpoints. This
distinction is a central provenance finding of Phase 1.
