# V5 Implementation Gap Audit

## Audit identity

- Base branch: `codex/project-v5-cnn-bilstm-final`
- Frozen base HEAD: `3b5ad7746d26658b48073c85d3ca716480873aa3`
- Audit branch: `codex/project-v5-1-cnn-bilstm-performance`
- Scope: implementation, selected configuration, checkpoints, final metrics, protocol, and documentation in V5.
- Evidence rule: a capability is treated as implemented only when the executed training path and/or final checkpoint metadata proves it; a YAML declaration alone is insufficient.
- V5 evidence is immutable. This audit does not change any file under `artifacts/v5/`, `reports/v5/`, or the V5 validators.

## Technical summary

V5 is more capable than the older narrative in `PROJECT.md` suggests, but several protocol declarations overstate what the final executed path supports. The UCI final path genuinely uses a context branch and trains gated fusion on most folds; a G3 regression head contributes only on the folds whose selected `multitask_alpha` is positive. The cross-subject experiment is combined-source pretraining followed by full-model fine-tuning, with no encoder freeze stage, and it was correctly rejected by its registered inner-validation stability rule.

The OULAD final path genuinely contains multi-kernel convolutions, a learned residual projection, mask-correct attention, and three-way gated fusion. However, the configured `concatenation` fusion candidate is not implemented in the V5 OULAD model factory: the model always builds the softmax gate. Selected augmentation is applied to the training partition before DataLoader construction, not sampled afresh per mini-batch. These gaps define the V5.1 work; they are not reasons to invalidate the frozen V5 evidence.

## Evidence inventory

| Evidence | What was checked |
|---|---|
| `configs/student_mat_v5.yaml`, `configs/student_por_v5.yaml`, `configs/oulad_v5.yaml` | Registered candidates and timing/data contracts |
| `src/studies/v5/common/uci_data.py` | UCI raw sequence, context allowlist, preprocessing, quasi-identity |
| `src/studies/v5/common/uci_model.py` | UCI context branch, gate, heads, and missing residual/multi-kernel path |
| `src/studies/v5/common/uci_training.py` | Actual loss, resampling, optimizer, checkpoint replay |
| `src/studies/v5/common/joint_learning.py` | Cross-subject data construction, leakage exclusion, pretraining/fine-tuning |
| `src/studies/v5/oulad/models.py` | OULAD kernels, residual, masking, pooling, and gate |
| `src/studies/v5/oulad/augmentation.py` | Training-only transforms and dependent-feature rebuilding |
| `src/studies/v5/oulad/training.py` | Actual training path, mask diagnostics, gate diagnostics, replay |
| `artifacts/v5/*/selected_configs.json` | Fold-selected executed configurations |
| `artifacts/v5/*/checkpoint_metadata.json` | Final checkpoint identity, parameter counts, replay and diagnostics |
| `artifacts/v5/*/final_metrics.csv` and OOF predictions | Origin of published metrics |
| `artifacts/v5/joint_uci/selection_decision.json` | Registered transfer decision |
| `reports/v5/final/validation_report.json` | Strict V5 evidence validation |
| `PROJECT.md`, `README.md`, `reports/v5/final/FINAL_MODEL_REVIEW.md` | Narrative consistency |

## Answers to the required implementation questions

### 1. Does the UCI context branch really enter the final model?

**Yes.** `DualBranchCNNBiLSTM` constructs a dense context encoder and `context_projection`; `uci_runner.py` fits the context transformer on each training partition and passes the transformed matrix into every neural fit. All final UCI checkpoints contain context-branch parameters. This is not a YAML-only feature.

Timing caveat: V5 includes `absences` in the primary context allowlist without a dataset-specific proof that the value is frozen at G2. V5.1 must exclude it from the primary safe context and reserve it for sensitivity analysis unless defensible timing evidence is added.

### 2. Was gated fusion really selected and trained?

**Yes, but not on every outer fold.** The selected configurations use gated fusion on 4/5 `student-mat` folds and 4/5 `student-por` folds; the remaining fold in each dataset uses concatenation. Gate layers are therefore present and trained in the corresponding final checkpoints. V5 records no UCI gate-distribution or saturation diagnostics, so it cannot establish that the learned gate avoided collapse.

### 3. Did classification plus G3 regression run?

**Partially.** The model always emits classification and regression outputs, and the training loop always computes both losses. Regression affects optimization only when `multitask_alpha > 0`:

- `student-mat`: selected on folds 1 (`0.10`) and 4 (`0.05`); classification-only on folds 0, 2, and 3.
- `student-por`: selected on folds 1 (`0.20`) and 4 (`0.05`); classification-only on folds 0, 2, and 3.

Thus the final pooled UCI ensemble mixes multitask and classification-only fold models. V5 did not implement the optional ordinal auxiliary objective.

### 4. What is the current joint-learning method?

**Combined-source pretraining followed by full-model fine-tuning.** For each Math inner split, V5 removes Portuguese records whose conservative quasi-identity overlaps the Math validation groups, fits preprocessing on Math-train plus allowed Portuguese rows, and pretrains one network on the concatenated source data. It then loads the entire state dict into a new Math fit with a reduced learning rate.

It is not simple final-data concatenation, and it is not a freeze–unfreeze transfer schedule: all parameters remain trainable during the fine-tuning call. It also uses one shared output head with a two-column subject indicator appended to context, rather than subject-specific heads. The registered stability rule selected `KEEP_STANDALONE` (mean inner delta `+0.002406`, positive in 3/5 seeds and 3/5 outer-training partitions).

### 5. Is OULAD multi-kernel CNN implemented?

**Yes.** `TemporalV5` builds a `ModuleList` of same-length Conv1D branches and concatenates their outputs. Final folds 0 and 1 selected kernels `[3, 5]`; fold 2 selected `[5]`. Therefore multi-kernel processing is present in two of three final outer-fold configurations, not uniformly across all folds.

### 6. Does a residual path exist?

**OULAD: yes. UCI: no.** OULAD adds a learned linear projection of the original sequence to the concatenated convolution output before LayerNorm. UCI V5 applies one convolution followed by LayerNorm/GELU and has no input residual projection. V5.1 must add a tested UCI residual path and make the OULAD residual contract explicit.

### 7. Is OULAD gated fusion used in the final checkpoint?

**Yes, always.** `OULADCNNBiLSTMV5` projects temporal, aggregate, and static branches, stacks them, predicts three softmax gate weights, and uses their weighted sum for the head. Final checkpoint metadata records three branch means; for example the first outer-fold checkpoints show a temporal-dominant but non-saturated allocation. The implementation has no concatenation branch even though YAML lists concatenation as a candidate.

### 8. Is masked attention correct?

**Yes for padding exclusion.** Attention logits are filled with negative infinity at invalid timesteps before softmax, and mean/max pooling also respects the validity mask. Final replay diagnostics report `attention_padding_max = 0.0` when attention exists. V5 does not report attention entropy, so it verifies padding safety but not attention concentration/collapse.

### 9. Is augmentation actually applied in training batches?

**The selected transform reaches training, but it is an offline partition transform rather than per-batch stochastic augmentation.** `fit_oulad_model` calls `augment_training_data` on training indices, rebuilds dependent dynamic and aggregate features, then constructs the DataLoader from the transformed training partition. Evaluation inputs come from the untouched data. Fold 1 selected `short_span_masking`; folds 0 and 2 selected `none`. The transform is deterministic for a seed and is not resampled each epoch or mini-batch.

### 10. Do V5 configs and `PROJECT.md` contradict each other?

**Yes.** The V5 configs and executed checkpoints show UCI context/gated/multitask use and OULAD multi-kernel/residual/gated use. Earlier final-state prose in `PROJECT.md` says that residual, multitask, imbalance, and context gates were not opened in final Study A and describes older compact architectures. Later V5 addenda report the V5 scores but do not fully reconcile the architecture narrative. V5.1 documentation must be generated from locked final artifacts and describe only the final state.

### 11. Which checkpoints produced the final metrics?

**The record-aligned outer-OOF checkpoint sets declared in each `checkpoint_metadata.json`.**

- `student-mat`: 25 `cnn_bilstm_outer_{0..4}_seed_{42,1201,2026,3407,7319}.pt` checkpoints.
- `student-por`: the analogous 25 checkpoints.
- OULAD: 15 full-hybrid checkpoints plus 15 CNN-only and 15 BiLSTM-only checkpoints across 3 outer folds and 5 fixed seeds.

The UCI ensemble is the arithmetic mean of all five fixed-seed probabilities for the record's outer fold. OULAD uses record-aligned fixed-seed probabilities with the registered per-fold inner threshold. Checkpoint metadata records file SHA-256, state-dict SHA-256, and zero replay difference. The strict V5 validator recomputes the evidence contracts and reports PASS.

### 12. Is there declared code that is never called?

**Yes, and there are declaration/implementation mismatches.**

- `fit_transform_partition` in `uci_data.py` is exported but the V5 runner duplicates preprocessing logic instead of calling it.
- OULAD YAML declares `fusion: [concatenation, gated]`, but the OULAD V5 model always constructs the gate and never dispatches on a fusion setting.
- UCI YAML suggests `joint_pretrain_then_subject_finetune`; the executed implementation pretrains then fine-tunes, but lacks the freeze–unfreeze behavior implied by a stronger transfer interpretation.
- OULAD YAML permits “one or multi small kernel residual”; final fold 2 is single-kernel, while folds 0 and 1 are multi-kernel. Documentation must not imply every final fold is multi-kernel.
- UCI regression heads exist in every checkpoint even when their loss weight is zero; their predictions are deliberately omitted from final rows for those folds.

No evidence was found that V5 silently accesses future OULAD data. The frozen strict report records `future_benchmark: NOT_EXECUTED`.

## Gap-to-V5.1 decisions

| Gap | V5.1 action |
|---|---|
| UCI raw two-value sequence only | Add deterministic two-timestep temporal feature construction from G1/G2, with no G3 input |
| No UCI residual or parallel kernels | Add registered kernel-1/kernel-2 residual temporal encoder |
| `absences` timing not adequately defended | Exclude from primary safe context; retain only a registered sensitivity candidate |
| UCI gate lacks diagnostics | Record mean, variance, saturation, and branch-dependence diagnostics |
| No FiLM | Add FiLM plus residual fusion as the third and final registered fusion candidate |
| Multitask only partial and no ordinal auxiliary | Screen the three pre-registered objective families; select only on inner validation |
| Transfer fine-tunes all layers immediately | Implement explicit freeze–unfreeze Portuguese-to-Math transfer and shared-trunk subject-head alternative |
| OULAD fusion candidate mismatch | Implement explicit concatenation and gated-residual modes |
| OULAD aggregate branch can dominate | Define compact primary aggregate features and retain full aggregate only as an oracle comparison |
| No masked-week self-supervised pretraining | Implement pretraining with train-partition-only masks and controlled screening |
| Augmentation is not epoch/batch stochastic | Preserve safe consistency rules and make application semantics explicit in artefacts |
| Documentation mixes historical and final architecture | Generate final README/PROJECT metrics and architecture statements from locked V5.1 artifacts |

## Baseline integrity and audit conclusion

The immutable baseline remains:

| Dataset | CNN–BiLSTM V5 ensemble Macro-F1 | Primary ML comparator | Comparator Macro-F1 | Delta |
|---|---:|---|---:|---:|
| `student-mat` | 0.8799168721 | Decision Tree | 0.9018875313 | -0.0219706593 |
| `student-por` | 0.8491516177 | Random Forest | 0.8605087126 | -0.0113570949 |
| OULAD | 0.8280026389 | XGBoost | 0.8283814220 | -0.0003787832 |

Audit verdict: **V5 is valid frozen evidence, but V5.1 must close implementation/protocol gaps instead of merely increasing trial count.** Performance targets remain directional and will not become validator gates.
