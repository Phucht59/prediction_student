# Final prediction model

**Authority:** Phase 4 Hybrid C0, selected by the project owner as the thesis-final model.  
**Evaluation:** robust inner 3×3 only.  
**Outer test used:** `false`.  
**Historical Phase 4 gate:** `NOT_READY_FOR_FINAL_EVAL` (unchanged in the research record).

## A. Final model

| Field | Value |
|---|---|
| `model_id` | `hybrid` |
| `display_name` | Hybrid |
| `public_class` | Hybrid |
| `architecture_id` | C0 |
| Topology | parallel CNN ∥ BiLSTM, corrected availability, 3-way masked softmax, binary risk logit |
| Shared widths | `d_fuse=128`, `cnn_channels=64`, `bilstm_hidden=128` |
| Training family | `L1_control` (Phase 3 numerics; one strategy family) |

UCI and OULAD share this architecture. They differ only in input dimensions, FIT-only preprocessing, vocabulary, class prior, and learned weights.

## B. One-architecture contract

- One UCI fitted Hybrid is evaluated at `S0 → S1 → S2`.
- One OULAD fitted Hybrid is evaluated at `20 → 35 → 50 → 75 → 100`.
- There is no `oulad_early` / `oulad_final` model split and no separate 100% model.
- Availability: CNN and BiLSTM are gated by `temporal_available`; aggregate is independent.

## C. UCI S0 / S1 / S2 (Hybrid, robust inner 3×3)

| State | Accuracy | PR-AUC | F1 | Recall |
|---|---:|---:|---:|---:|
| S0 | 0.5213 | 0.4547 | 0.4291 | 0.8421 |
| S1 | 0.8553 | 0.8214 | 0.6899 | 0.7587 |
| S2 | 0.9094 | 0.9101 | 0.8010 | 0.8545 |
| macro | 0.7620 | 0.7288 | 0.6400 | 0.8184 |

S0 has no G1/G2. It is the weak state and is reported as a limitation, not a win.

## D. OULAD 20 / 35 / 50 / 75 / 100 (Hybrid, robust inner 3×3)

| State | Accuracy | PR-AUC | F1 | Recall |
|---|---:|---:|---:|---:|
| 20% | 0.6854 | 0.7624 | 0.6781 | 0.7769 |
| 35% | 0.7456 | 0.8058 | 0.7001 | 0.7464 |
| 50% | 0.8006 | 0.8483 | 0.7306 | 0.7207 |
| 75% | 0.8627 | 0.8885 | 0.7807 | 0.7221 |
| 100% | 0.9088 | 0.9204 | 0.8372 | 0.7807 |
| macro_early | 0.7736 | 0.8262 | 0.7224 | 0.7415 |
| macro_5stage | 0.8006 | 0.8451 | 0.7453 | 0.7493 |

100% must be read with the length≈Withdrawn confounder. Length is not an explicit predictor.

## E. Baseline comparison (PR-AUC)

Active roster: Hybrid, LR, DT, RF, SVM, MLP. XGBoost is not active.

### UCI

| Model | S0 | S1 | S2 | macro |
|---|---:|---:|---:|---:|
| Hybrid | 0.4547 | **0.8214** | **0.9101** | 0.7288 |
| LR | 0.4754 | 0.7794 | 0.8812 | 0.7120 |
| DT | 0.4169 | 0.7330 | 0.8547 | 0.6682 |
| RF | **0.4995** | 0.7895 | 0.9072 | **0.7320** |
| SVM | 0.4970 | 0.7936 | 0.8866 | 0.7258 |
| MLP | 0.4486 | 0.7595 | 0.8778 | 0.6953 |

### OULAD

| Model | 20 | 35 | 50 | 75 | 100 | early | 5-stage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hybrid | 0.7624 | **0.8058** | **0.8483** | **0.8885** | **0.9204** | **0.8262** | **0.8451** |
| LR | **0.7632** | 0.7986 | 0.8399 | 0.8828 | 0.9114 | 0.8211 | 0.8392 |
| DT | 0.7084 | 0.7548 | 0.7954 | 0.8530 | 0.8862 | 0.7779 | 0.7996 |
| RF | 0.7522 | 0.7940 | 0.8402 | 0.8847 | 0.9154 | 0.8178 | 0.8373 |
| SVM | 0.7534 | 0.7835 | 0.8257 | 0.8723 | 0.9018 | 0.8087 | 0.8273 |
| MLP | 0.6799 | 0.7388 | 0.7998 | 0.8556 | 0.8964 | 0.7685 | 0.7941 |

Full Accuracy / F1 / Recall: `uci_final.csv`, `oulad_final.csv`.

## F. Information-growth interpretation

UCI PR-AUC rises sharply once grades appear: S0 0.4547 → S1 0.8214 → S2 0.9101.  
OULAD PR-AUC rises with cutoff: 0.7624 → 0.8058 → 0.8483 → 0.8885 → 0.9204.  
The same checkpoint is scored at every state.

## G. Dataset-nature robustness

The same C0 remains usable on both a small grade-sequence dataset and a large VLE dataset. Raw PR-AUC is not comparable across tasks. The honest contrast is the margin versus the strongest active baseline: OULAD Hybrid beats LR; UCI Hybrid loses the macro to RF because of S0.

## H. Leakage audit

Phase 4 leakage audit passed: G3 is never a predictor; S0 has no G1/G2; OULAD forbids `final_result`, `date_unregistration`, and future events; FIT-only scaling; group-safe partitions; outer fold excluded from inner work.

## I. Overfitting audit

Stage-level 3×3 (9 runs). Gaps are **not** copied from the dataset macro.

| Dataset / state | PR-AUC mean | PR-AUC std | train PR-AUC | gap mean | class |
|---|---:|---:|---:|---:|---|
| UCI S0 | 0.4547 | (see audit) | (see audit) | **0.1254** | HIGH |
| UCI S1 | 0.8214 |  |  | 0.0352 | MODERATE |
| UCI S2 | 0.9101 |  |  | 0.0203 | MODERATE |
| OULAD 20% | 0.7624 |  |  | 0.0339 | LOW |
| OULAD 35% | 0.8058 |  |  | 0.0312 | LOW |
| OULAD 50% | 0.8483 |  |  | 0.0238 | LOW |
| OULAD 75% | 0.8885 |  |  | 0.0203 | LOW |
| OULAD 100% | 0.9204 |  |  | 0.0155 | LOW |

UCI S0’s generalization gap is larger than S1 and S2 on the official 3×3. Full numbers: `artifacts/prediction/final/OVERFIT_AUDIT.json`.

## J. Limitations

- UCI S0 underperforms RF by about 0.045 PR-AUC. S0 has no temporal academic-grade input.
- OULAD 100% history length is strongly associated with Withdrawn.
- The Phase 4 strict superiority gate required a win on both datasets. That gate failed. This report does not rewrite that fact.
- No Phase 4 outer evaluation exists. None is invented here.

## K. Finalization decision

Phase 4 strict superiority gate originally returned `NOT_READY_FOR_FINAL_EVAL`.  
The project owner subsequently selected Phase 4 C0 as the thesis-final model authority.  
No outer results were fabricated or used in this decision.
