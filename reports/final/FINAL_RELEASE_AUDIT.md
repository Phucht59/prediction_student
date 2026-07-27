# Final Release Audit

## Git

| Field | Value |
|---|---|
| Branch | `codex/final-release-cleanup` |
| Base commit | `f51e1f43d9af768194bb34ea5ff4aeb8e4cbd502` |
| Release commit | Resolved by `git rev-parse v1.0-thesis-final^{commit}` after publication |
| Tag | `v1.0-thesis-final` (created only after validated main) |
| Working tree | Must be CLEAN at release handoff |

## Final models and metrics

| Model | Macro-F1 | Balanced Accuracy | PR-AUC |
|---|---:|---:|---:|
| CNN-BiLSTM MAT | 0.9015 | 0.9021 | 0.9442 |
| CNN-BiLSTM POR | 0.8623 | 0.8676 | 0.9147 |
| CNN-BiLSTM OULAD | 0.8281 | 0.8203 | 0.8934 |

CNN-BiLSTM OULAD additionally retains risk precision 0.8522, risk recall
0.7236, risk F1 0.7826 and ECE 0.0087.

## Dependency audit and relocation

Before relocation, public final code referenced versioned source, config and
artifact paths. Runtime/replay/checksum dependencies were first copied to
non-versioned final paths. Evidence-only and historical paths were then
archived locally.

| Old path | New canonical path | Local test_lab path | Classification / reason | SHA-256 preserved? |
|---|---|---|---|---|
| `src/studies/v5_1/common/uci_model.py` | `src/models/_uci.py` | `test_lab/v5_1/release_cleanup/src_studies_v5_1` | A runtime; refactored architecture | checkpoint behavior preserved |
| `src/studies/v5_1/oulad/models.py` | `src/models/_oulad.py` | `test_lab/v5_1/release_cleanup/src_studies_v5_1` | A runtime; refactored architecture | checkpoint behavior preserved |
| `src/studies/v6/multitask.py` | `src/models/oulad_multitask.py` | `test_lab/v6/release_cleanup/src_studies_v6` | A runtime; final inference class | checkpoint behavior preserved |
| `artifacts/v5_1/student_mat/checkpoints` | `artifacts/final/models/cnn_bilstm_mat` | `test_lab/v5_1/release_cleanup/artifacts_v5_1` | B/C replay and checksum | YES |
| `artifacts/v5_1/student_por/checkpoints` | `artifacts/final/models/cnn_bilstm_por` | `test_lab/v5_1/release_cleanup/artifacts_v5_1` | B/C replay and checksum | YES |
| `artifacts/v6/prediction/final/checkpoints` | `artifacts/final/models/cnn_bilstm_oulad` | `test_lab/v6/release_cleanup/artifacts_v6` | B/C replay and checksum | YES |
| `artifacts/v5_1/*/final_metrics.json` | `artifacts/final/metrics` | corresponding local archive | D canonical evidence | YES |
| `artifacts/v6/prediction/final/metrics.json` | `artifacts/final/metrics/cnn_bilstm_oulad.json` | `test_lab/v6/release_cleanup/artifacts_v6` | D canonical evidence | YES |
| tuning/search paths | `artifacts/final/tuning_evidence` | corresponding local archive | D defense evidence | YES |
| architecture-diagnosis artifacts | `artifacts/final/ablation_evidence` | `test_lab/v6_1/release_cleanup/artifacts_v6_1` | D controlled diagnostic | YES |
| corrected recommendation validation | `artifacts/final/recommendation` | `test_lab/v6_2/release_cleanup/artifacts_v6_2` | A/D final semantics/evidence | YES |
| remaining versioned reports/code | final reports where needed | `test_lab/v4` … `test_lab/v6_2` and `archived_experiments` | E historical only | canonical evidence copied first |

## Evidence preservation

- Optuna/tuning: MAT and POR focused-search/screening/multi-seed evidence; OULAD
  trial CSV has 72 recorded COMPLETE trials and no fabricated PRUNED rows.
- Ablation: CNN-only, BiLSTM-only, hybrid, capacity, dilation, serial/parallel,
  CNN skip, aggregate/static and temporal-order diagnostics retained.
- Bootstrap, seed results, calibration and Top-k evidence retained.
- Checkpoints: 25 MAT + 25 POR + 15 OULAD = 65 final ensemble checkpoints.
- Recommendation: 15,378 plans; 10,953 GENERATED, 1,209 PARTIAL_EVIDENCE and
  3,216 ABSTAINED; deterministic replay PASS.
- Database: frozen cutover evidence reports 15,378 risk profiles, 15,378 plan
  objects and 27,355 actions.
- `artifacts/final/evidence_manifest.json` and
  `artifacts/final/checksums/checkpoint_manifest.json` retain public hashes and
  original-path provenance.

## Validation

| Check | Status |
|---|---|
| `project.py final status` | PASS |
| `project.py final report` | PASS |
| `project.py final validate` | PASS |
| pytest | 23 passed, 0 failed |
| Final checkpoint load | PASS (65/65) |
| Checksum replay | PASS |
| Prediction/metric replay | PASS |
| Recommendation deterministic replay | PASS |
| Future OULAD | LOCKED_NOT_EXECUTED |
| Outer-test selection | NOT USED |
| Frozen database cutover audit | PASS |
| Live database strict-public audit | PASS (21/21 checks; permissions PASS) |

The live audit used an administrative test connection supplied at release
time. Credentials are not stored in the repository. The persisted database
snapshot retains 15,378 risk profiles, 15,378 traceable plan objects and
27,355 actions; the corrected recommendation-validation package remains a
separate immutable evidence set.

## Scientific freeze

- Canonical metrics changed: **NO**
- Prediction model retrained: **NO**
- New Optuna search run: **NO**
- Future OULAD accessed: **NO**
- Outer test used for tuning: **NO**
- Final labels/splits/thresholds changed: **NO**
