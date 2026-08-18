# Phase 2 — Behavior Change Matrix

| Area | Before | After | Behavior changed? | Scientific impact |
| --- | --- | --- | --- | --- |
| `selected_epoch` metadata | Fixed refit reported 1 | Reports actual final epoch N | Metadata only | Restores traceability |
| Actual epochs | Outer refit hard-coded 4 | Inner-selected median budget, then exact fixed refit | Yes, future corrected runs | Removes unvalidated budget |
| Run ID | Payload and manifest used different hashes | One canonical identity helper | Metadata/control | Prevents run collision/misattribution |
| Concat head dimensions | Aux heads expected `fusion_hidden` | All heads consume `representation_dim` | Yes for broken alternate mode | Makes supported mode executable |
| Gated residual | 64-dimensional representation | Unchanged | No | Frozen architecture behavior preserved |
| Research threshold | Conflated with generic threshold | Explicit inner-OOF Macro-F1 policy | Semantic/API | Prevents outer fitting |
| Operational threshold | Recall at precision constraint | Retained, explicitly barred from model selection | Semantic/API | Deployment objective stays separate |
| Config authority | Official/unified values conflated | Versioned unified registry + legacy distinction | Control plane | Reproducible fingerprints |
| Pretraining | Template strategy could imply execution | requested=false, executed=false, checkpoint=null | Provenance | No false pretraining claim |
