# HYBRID VNEXT PHASE 2 — DESIGN + INNER-ONLY PROTOTYPE

**Decision:** `READY_FOR_PHASE3_HPO`  
**Frozen topology:** `C0` — Phase8 parallel CNN ∥ BiLSTM + **corrected** availability + 3-way softmax + binary risk head.  
**One architecture** for UCI S0/S1/S2 and OULAD 20/35/50/75. FINAL-100 was diagnostic only.  
**Outer test used:** `false`  
**Authority overwritten:** `false`  
**Namespace:** `experiments/hybrid_vnext/` (not `src/prediction/`)

---

## A. What was tested

Gate 0: CUDA fail-fast on RTX 2060 6 GB; split-hash lock; four-case availability unit tests for C0–C3.

Gate 1: Feature parity (XGB static / static+agg / full; RF, LR, DT, MLP; SVM on UCI) and Hybrid tabular-only vs full C0. Temporal-order falsification (identity / reverse / shuffle) on trained C0. FINAL-100 length shortcut diagnostic (no architecture selection).

Gate 2: Preregistered topologies C0–C3, 1 fold × 1 seed, both domains.

Gate 3: Survivors C0 and C3, 3 folds × 3 seeds.

Gate 4: Learning-rate check on C0 only (`8e-4`, `2e-4`, `6e-5`), 1 fold × 1 seed, max 40 epochs.

Gate 5 (loss): not run — topology already decided on PR-AUC; no evidence that loss was the bottleneck.

No outer labels entered FIT/STOP/VALID scoring. Historical outer numbers were not used to pick C0.

---

## B. Integrity / leakage status

| Check | Status | Evidence |
|---|---|---|
| CUDA required for Hybrid | PASS | 52/52 Hybrid jobs on `cuda:0`; peak VRAM 135 MB |
| Split hashes | PASS | UCI/OULAD inner match Phase8 expected hashes |
| Outer firewall | PASS | outer fold-0 IDs loaded only for exclusion |
| Group-safe partitions | PASS | StratifiedGroupKFold FIT/STOP; disjoint from VALID |
| UCI G3 / absences | PASS | forbidden as predictors |
| UCI stage views | PASS | S0 has no G1/G2; aggregate_available=0 |
| OULAD cutoff | PASS | events `< cutoff` and `>= observation_start`; D3_both_safe |
| Preprocess FIT-only | PASS | context/aggregate/temporal scalers fit on FIT ids |
| FINAL-100 not used to select | PASS | `used_for_architecture_selection=false` |
| Availability mapping | PASS | Gate 0: temporal=1 & aggregate=0 still gives BiLSTM mass > 0 |

Machine lock: `artifacts/hybrid_vnext/phase2/PROTOCOL_LOCK.json`.

---

## C. Feature parity result

Same cutoff-safe columns for trees and Hybrid tabular path:

- static/context (train-fitted OHE/scale)
- aggregates
- last / mean / max of each temporal channel
- progress

XGB 1-fold inner VALID PR-AUC:

| Domain / stage | static | static+agg | full |
|---|---:|---:|---:|
| UCI S0 | 0.439 | 0.435 | 0.428 |
| UCI S1 | 0.439 | 0.714 | 0.719 |
| UCI S2 | 0.439 | 0.859 | 0.866 |
| OULAD 20% | 0.567 | 0.764 | 0.769 |
| OULAD 35% | 0.542 | 0.813 | 0.813 |
| OULAD 50% | 0.528 | 0.855 | 0.855 |
| OULAD 75% | 0.493 | 0.898 | 0.900 |

**XGB(B→C):** UCI S2 +0.007; OULAD ≤ +0.005. Almost all tree signal is already in static+aggregate. Extra last/mean/max is a small increment.

Hybrid 1-fold:

| | UCI macro | OULAD macro | 20% | 75% |
|---|---:|---:|---:|---:|
| D tabular-only (C3, g=0) | 0.632 | 0.826 | 0.760 | 0.891 |
| E full C0 | 0.685 | 0.834 | 0.769 | 0.897 |

**Hybrid(D→E):** OULAD +0.008 macro. Temporal branch is small but not zero. UCI gain is concentrated in S1/S2 (grades as sequence), while S0 full C0 (0.416) is *below* tabular-only (0.446) on this fold — softmax without temporal leaves a weaker tabular path than the parity encoder.

Strong 1-fold full baselines (PR-AUC): OULAD 20% XGB 0.769 / LR 0.765 / C0 0.769; 75% XGB 0.900 / C0 0.897. Trees were not weakened.

---

## D. Temporal-order result

Eval-only permutation of a trained C0 (masks and marginals preserved):

| Domain | identity | reverse | shuffle | Δ reverse | Δ shuffle |
|---|---:|---:|---:|---:|---:|
| OULAD macro | 0.8340 | 0.8197 | 0.8245 | **−0.0143** | −0.0095 |
| UCI macro | 0.6851 | 0.6843 | 0.6845 | −0.0009 | −0.0006 |

OULAD reverse drop exceeds the ~0.01 guideline: **C0 uses temporal order**. Shuffle is slightly weaker but consistent in sign. UCI drop is negligible (T ≤ 2): order is not a meaningful inductive source there.

---

## E. Candidate topology definitions

All candidates: CNN + BiLSTM, same input contract, one topology across datasets.

| ID | Temporal path | Fusion | Tabular | Target size |
|---|---|---|---|---|
| **C0** | parallel CNN ∥ BiLSTM | corrected 3-way softmax | Phase8 static+agg | ~514k |
| C1 | parallel | residual add (`h_tab + 1[T>0] Δ`) | parity summaries | ~186k |
| C2 | **serial CNN→BiLSTM** | residual add | parity | ~169k |
| C3 | **serial CNN→BiLSTM** | gated residual `h_tab + g Δ` | parity | ~169k |

C0 is the forensic Phase8 graph with the **only** required bugfix: `available = [1, temporal, temporal]`.

---

## F. Candidate parameter counts

| ID | UCI | OULAD |
|---|---:|---:|
| C0 | 513 413 | 514 373 |
| C1 | 185 029 | 188 549 |
| C2 | 168 517 | 172 037 |
| C3 | 168 517 | 172 037 |

C1–C3 sit in the 150–300k design band. C0 remains ~514k. Inner evidence did **not** justify discarding C0 for the smaller serial residual family.

---

## G. 1-fold screen

Fold 0, seed 42, lr `2e-4`, max 24 epochs, AMP, CUDA.

| ID | OULAD macro | UCI macro | OULAD worst (20%) | UCI S0 | UCI S2 |
|---|---:|---:|---:|---:|---:|
| C0 | **0.8340** | 0.6851 | 0.7691 | 0.4156 | **0.8734** |
| C1 | 0.8314 | **0.6914** | 0.7648 | 0.4359 | 0.8609 |
| C2 | 0.8324 | 0.6907 | 0.7678 | 0.4620 | 0.8448 |
| C3 | 0.8327 | 0.6890 | 0.7677 | **0.4635** | 0.8395 |

All four are within 0.003 on OULAD. Survivors by preregistered rule: **C0, C3**.

---

## H. 3×3 robust confirmation

3 folds × 3 seeds. Mean ± std of macro PR-AUC.

| ID | OULAD | UCI | UCI gap | Mean best epoch |
|---|---|---|---:|---:|
| **C0** | **0.8255 ± 0.0061** | **0.7250 ± 0.0263** | 0.085 | 20.1 / 21.6 |
| C3 | 0.8248 ± 0.0063 | 0.7032 ± 0.0509 | 0.112 | 22.4 / 23.7 |

OULAD stage means (C0): 20% 0.7629, 35% 0.8042, 50% 0.8474, 75% 0.8874.  
C3 is statistically tied on OULAD and **worse and less stable on UCI**.

---

## I. UCI behavior

C0 3×3 stage PR-AUC: S0 0.467 ± 0.041, S1 0.811 ± 0.029, S2 0.897 ± 0.024.

Gate diagnostics (C0, fold 0 seed 42):

- S0: `g_temporal=0`, tabular mass=1. Architecture adapts; no CNN/BiLSTM contribution.
- S1/S2: temporal mass ≈ 0.91. Softmax shifts almost all weight off tabular once a grade exists.

That last point is a remaining **mechanism** issue, not a topology fork: C3 keeps a tabular residual at S1/S2 (`g≈0.67–0.70`) and wins S0 on the 1-fold screen, but loses S2 and explodes 3×3 variance. Phase 3 may add a small entropy floor or residual mix **without** changing the public C0 graph unless a later protocol re-opens topology.

Variance is still high (UCI n is small). Phase 3 must keep all seeds; no best-seed selection.

---

## J. OULAD behavior

C0 3×3 is stable (std 0.006). Temporal contribution is real:

- tabular-only 0.826 → full 0.834 (1-fold)
- reverse −0.014, shuffle −0.0095
- softmax mass: tabular 0.33 → 0.23 from 20% to 75%; BiLSTM 0.46 → 0.54; CNN ~0.21

CNN is used but secondary to BiLSTM. Early-warning ranking is not sacrificed for 75% only.

---

## K. Gate / branch diagnostics

Desired qualitative pattern is observed on C0:

| Setting | Observed |
|---|---|
| UCI S0 | temporal contribution = 0 |
| UCI S1/S2 | temporal on; order unused (T≤2) |
| OULAD 20→75 | temporal mass and BiLSTM mass rise |
| OULAD order | reverse hurts |

C3 also shows `g` rising 0.52 → 0.60 with cutoff, but that did not translate into a robust metric win.

---

## L. Overfit analysis

| Candidate | OULAD gap | UCI gap | Note |
|---|---:|---:|---|
| C0 3×3 | 0.022 | 0.085 | UCI overfit remains; OULAD mild |
| C3 3×3 | 0.016 | 0.112 | worse UCI generalization |
| G4 C0 lr 2e-4, 40 ep | 0.024 | 0.121 | longer budget widens UCI gap |
| G4 C0 lr 6e-5 | 0.005 | 0.105 | OULAD cleaner; UCI S0 weaker |

Do not use outer metrics as an overfit detector. Phase 3 should prefer moderate LR (`2e-4` default) with patience, and consider stronger UCI regularization **as a numeric HP**, not a second architecture.

---

## M. FINAL-100 shortcut diagnostic

Not used for selection.

| Quantity | Value |
|---|---:|
| n | 32 593 |
| Withdrawn mean weeks | 9.17 |
| Pass / Distinction / Fail mean weeks | 37.32 / 37.30 / 36.94 |
| Short history (≤20w) withdrawn rate | 0.999 |
| Length PR-AUC for Withdrawn | 0.991 |
| Length PR-AUC for Fail vs completed | 0.341 |
| flagged_shortcut_risk | **true** |

Length almost *is* the Withdrawn label. Fail vs Pass is not solved by length. Architecture must not be justified on FINAL-100 headline scores.

---

## N. Selected topology

**C0 — availability-corrected Phase8 Hybrid.**

```text
static + masked aggregate
        -> tabular encoder
temporal + mask
        -> adapter
           ├─ ResidualCNN  -> h_cnn
           └─ BiLSTM       -> h_lstm     (parallel)
available = [1, T>0, T>0]
softmax fusion -> binary logit
```

Same class and graph on every dataset/stage. Adaptation only via masks, lengths, progress, input dims, and fitted weights.

Why C0, not C3 (the preferred serial residual hypothesis):

1. OULAD 3×3: C0 0.8255 vs C3 0.8248 — no serial gain.
2. UCI 3×3: C0 higher mean and **half** the std (0.026 vs 0.051).
3. Serial CNN→BiLSTM was tested (C2, C3) and did not earn replacement.
4. Availability fix alone already lets C0 use temporal order on OULAD.

This is not “CNN→BiLSTM serial”. Calling C0 serial would be false. Phase 3 freezes **parallel CNN ∥ BiLSTM**.

---

## O. Rejected topologies and why

| ID | Why killed |
|---|---|
| C1 | 1-fold OULAD below C0; hard residual adds no unique evidence |
| C2 | Serial path; UCI S2 −0.029 vs C0; OULAD not better |
| C3 | Preferred hypothesis; robust OULAD tie, UCI regression + variance — fails unified eligibility |

No candidate was kept because it “looked deep”.

---

## P. Remaining hyperparameters for Phase 3

Topology is frozen. Phase 3 may tune only:

- `lr` (screen: `2e-4` best OULAD 1-fold 40-ep 0.836; `8e-4` 0.831; `6e-5` 0.835)
- `weight_decay`, `dropout`
- `pos_weight` multiplier
- `max_epochs` / `patience` (best epoch ~18–31 at `2e-4`)
- `d_fuse`, `cnn_channels`, `bilstm_hidden` **inside the same parallel+softmax graph**
- optional `entropy_floor` on the 3-way gate (not used in this prototype)
- inner threshold / calibration

Do **not**: change serial/parallel, drop a branch, fork UCI vs OULAD, HPO on outer, or promote to authority before Phase 3 acceptance.

Loss screen (plain BCE / Focal) remains optional if Phase 3 ranking metrics stall.

---

## Q. GO / NO-GO

```text
READY_FOR_PHASE3_HPO
```

| Criterion | Met? |
|---|---|
| Availability bug fixed in prototype | YES |
| Leakage tests pass | YES |
| Exactly one unified topology | YES — C0 |
| Same topology on UCI and OULAD | YES |
| OULAD temporal contribution useful | YES (D→E +0.008; reverse −0.014) |
| UCI no material regression vs C3 / variance worse | YES — C0 is the more stable unified choice |
| Outer unused for selection | YES |
| Fits GTX 2060 6 GB | YES (135 MB peak) |
| Evidence reproducible | YES — matrix, runs, hashes |

Phase 2 does **not** claim that C0 already beats XGB on a future outer test. It claims the topology is coherent enough to spend Phase 3 compute on numeric HPO instead of more architecture search.

---

## Artifacts

| File | Role |
|---|---|
| `artifacts/hybrid_vnext/phase2/SELECTED_TOPOLOGY.json` | Frozen graph |
| `artifacts/hybrid_vnext/phase2/EXPERIMENT_MATRIX.csv` | 277 inner rows |
| `artifacts/hybrid_vnext/phase2/PROTOCOL_LOCK.json` | Splits, contracts, firewall |
| `artifacts/hybrid_vnext/phase2/CUDA_EXECUTION_AUDIT.json` | 52 CUDA Hybrid jobs |
| `artifacts/hybrid_vnext/phase2/FEATURE_PARITY_MANIFEST.json` | Shared feature contract |
| `artifacts/hybrid_vnext/phase2/gate0.json` … `gate4.json` | Gate payloads |
| `experiments/hybrid_vnext/` | Prototype code only |
