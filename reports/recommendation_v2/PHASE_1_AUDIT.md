# Phase 1 audit — Recommendation V2

## Verdict

`PHASE_1_BLOCKED`. Audit and design are complete, but the baseline cannot be declared singular: the primary CLI (`project.py final validate`) freezes 65 legacy CNN-BiLSTM checkpoints including historical `cnn_bilstm_oulad`, while `configs/final/final_model_authority.yaml` and `scripts/final/validate_final_release.py` select `h1_tabular_residual_oulad` for OULAD. Recommendation evidence is a third lineage: frozen V6 `F2_MIDDLE`, not canonical V3 `FINAL`. No model, checkpoint, split, cutoff, database schema, API, or production behavior was changed.

## Prediction source of truth

| Scope | Called code/config | Checkpoints | Result authority | Finding |
|---|---|---:|---|---|
| Primary CLI | `project.py` → `src/final_release/validate.py` | 65 = MAT 25 + POR 25 + historical OULAD 15 | `artifacts/final/final_results.json` | still exposes three CNN-BiLSTM models |
| Newer final release | `scripts/final/validate_final_release.py` → `configs/final/final_model_authority.yaml` | MAT 25 + POR 25 + canonical H1 final/shared protected set | `artifacts/canonical_v3/*` | OULAD final model is H1 tabular residual expert |
| Recommendation | removed V6/V6.2 generator in Git history | frozen V6 F2 ensemble | `artifacts/final/recommendation/*` | not replayed from current canonical V3 model |

The official architecture definitions available now are `src/models/_uci.py` (`UCICNNBiLSTM`), `src/models/_oulad.py` (`_OULADCNNBiLSTMBackbone`), `src/models/oulad_multitask.py` and `src/models/oulad_tabular_residual.py`. `src/models/cnn_bilstm.py` is a facade, not a checkpoint loader. MAT checkpoints contain a transfer/shared-trunk topology for which current production construction code is not present; POR raw state dicts map more directly to `UCICNNBiLSTM` but fold configs vary.

## Architecture and inputs

- UCI temporal input is `[N,2,7]`; stages are `S0` (no grade), `S1` (G1 only), `S2` (G1+G2). Context excludes raw G1/G2/G3. CNN uses kernels 1/2, followed by bidirectional LSTM, mean/max pool, context fusion, then classification/regression/ordinal heads. Fold-specific parameter counts are locked in `BASELINE_LOCK.json`.
- OULAD temporal input is `[N,T,47]`, aggregate input 165 (161 derived plus four stage context fields), static input 13/14 depending protocol. Events satisfy `event_day < cutoff_day`; score values are excluded in canonical V3 because release timestamps are unavailable. Stages are 20%, 35%, 50%, 75%, and canonical `FINAL = presentation_length - 14 days`.
- Historical unified `CNNBiLSTMOULAD` uses multi-kernel CNN `[2,3,5]`, BiLSTM hidden 64, masked mean/max pooling, 64-dimensional gated-residual fusion and risk/survival/outcome heads. Canonical H1 adds a 178→48→32 tabular residual risk path; it has 160,492 parameters and architecture hash `df5cd885…`.

## Prediction outputs and embedding

| Output | Current model code | Persisted artefact | Status |
|---|---|---|---|
| logits | `classification` (UCI); `binary_logit`, `hazard_logit`, `outcome_logit` (OULAD) | not in final prediction parquet | extractable |
| probabilities | softmax/sigmoid in evaluation code | present | existing |
| predicted class | argmax/threshold | present | existing |
| calibrated confidence | ECE/calibration artefacts and V6 confidence bands | V6 F2 risk profile only | lineage-specific, not unified current API |
| uncertainty | predictive entropy + seed std formula in historical V6 | `uncertainty_score`, `seed_disagreement` | existing only for historical F2 recommendation profile |
| student embedding | UCI `encode()` return; OULAD `student_state_embedding` | not persisted | extractable |

For canonical OULAD H1, the preferred hook is output key `student_state_embedding`, the fused 64-D tensor after CNN/BiLSTM/aggregate/static fusion and before `backbone.head`. It has no dropout applied after fusion, but its upstream branch encoders contain dropout and branch dropout; extraction must use `model.eval()` and `torch.inference_mode()`. No hook is necessary because `forward()` already returns it. Caveat: the final H1 risk logit also uses a 32-D `tabular_expert_embedding` through a residual bypass, so the 64-D student embedding does not alone span every final-logit pathway.

Embedding extraction is prediction-invariant if the frozen forward output is read without mutation. Phase 2 must snapshot file SHA-256, state-dict tensor hashes, model parameter names/values, logits and probabilities before and after adapter attachment. Use exact equality for stored file/state hashes; for inference compare against a same-device/same-dtype reference. Expected tolerance is `0` when the identical eval forward is reused; if serialization or mixed precision changes the execution path, derive `atol/rtol` from `torch.finfo(dtype).eps`, operation depth and a repeated-run empirical envelope, and record dtype/device rather than inventing a universal tolerance.

## Current recommendation behavior

The current service only retrieves a stored plan and can append an advisor review. No generator is callable on this branch. Historical V6.2 generated candidates with fixed thresholds and rules, applied max four actions/180 minutes, duplicate protection, abstention, feature lineage, post-cutoff and sensitive-feature guards. It used class/risk probabilities, confidence, uncertainty/disagreement and observed state, but not a hidden embedding or neural action ranker.

V6.2 has 15,378 plans: 10,953 generated, 1,209 partial, 3,216 abstained, 23,192 actions and 3,216 zero-action plans. The database payload is V5.2: 15,378 active plans, 27,355 actions, zero zero-action plans, engine `v5_2`, policy `v6_risk_to_recommendation_policy_v1`, model version `v6_C_temporal_multitask_W0_seed_ensemble`. `PROGRESS_MONITORING` is present in every non-abstained V6.2 plan (12,162/12,162), so it is a default. V6.2 defines seven action types but selects only four; `STUDY_SCHEDULE`, `INSTRUCTOR_CONTACT`, and `PEER_STUDY` are unreachable in that generator.

KEEP: retrieval, normalized plan/action/review tables, honest expert gate, lineage guards, abstention semantics, workload/action/duplicate guards, advisor review and non-causal claims. REFACTOR: typed contracts, action catalog, prerequisites, review states and version lineage. REPLACE: threshold/rule action selection with a later neural ranker. REMOVE from V2 path: legacy independent MLP and weak/pseudo-label supervision. MISSING: frozen-model adapter, unified uncertainty, neural ranker, standalone solver and multi-week builder.

## Expert data and conflicts

The package has 60 prepared cases and templates for two reviewers, but reviewers = 0, cases scored = 0, plan/action metrics = null and inter-rater agreement is pending. Therefore `expert_status = PENDING_REAL_EXPERT_LABELS` and `recommendation_training_status = BLOCKED`. Migration 011 defines storage but final database audits show all three expert tables empty and review rows = 0.

Major conflicts are: (1) OULAD model authority described above; (2) recommendation F2 lineage versus canonical V3 FINAL authority; (3) V6.2 technical artefact versus V5.2 database payload/counts; (4) `reports/final/DATABASE_CURRENT_STATE_AUDIT.md` describes an older empty legacy database, while later final database artefacts record 15,378 loaded profiles/plans; (5) the main checkpoint manifest has 65 legacy checkpoints but no dedicated manifest for 15 canonical H1 final checkpoints. These are versioned differences where identifiable; the choice of OULAD baseline remains `UNRESOLVED` and blocks PASS.

## Targeted Git history

| Commit | Function | Reuse |
|---|---|---|
| `aebdd84` | independent recommendation MLP trained on rule-derived weak targets | remove; violates V2 architecture |
| `4915a12` | governed rule policy, prerequisites, workload and advisor gate | reuse governance concepts only |
| `177a87a` | V5.2 engine/taxonomy persisted in database artefact | refactor catalog; replace ranking |
| `9611bfc` | V6 risk profile with probability, entropy/seed uncertainty and ML disagreement | reuse fields after lineage is reconciled; do not require the independent ML cross-check |
| `d38f303` | V6.2 cutoff-safe observed state and zero-action abstention | reuse safety and lineage patterns |

## Missing gates before training

Architecture owner must explicitly select the OULAD baseline/cutoff lineage and resolve its validator/manifest. Then Phase 2 may add extraction-only adapter contracts and expert-label ingestion. Neural ranker training remains blocked until real ratings exist, student-group splits are frozen, labels are quality-checked, and invariance/leakage tests pass.
