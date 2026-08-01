# recommend_hybrid data contracts

Phase 2 implements immutable dataclasses in `src/recommend_hybrid/contracts.py`. Primary records are typed; free-form dictionaries are not accepted at the predictor, observed-state, candidate, or expert-import boundaries. JSON serialization preserves canonical stage values.

| Contract | Required fields | Validation and source |
|---|---|---|
| `PredictionContext` | pseudonymous student/course key; canonical stage; cutoff; class; two probabilities; confidence and provenance; uncertainty; seed disagreement; fold; unique seeds; checkpoint references; architecture hash; parameter count | frozen authority; finite normalized probabilities; positive cutoff; immutable checkpoint lineage |
| `StudentRepresentation` | 64-D student state; 32-D tabular expert state; model authority; embedding source; dtype; device | existing frozen forward outputs only; exact dimensions and finite values |
| `ObservedLearningState` | activity fields; inactivity; assessment progress; grade trend; course progress; evidence availability/missing masks; per-field lineage; cutoff; stage | `event_day < cutoff_day`; unknown evidence is `None`, never zero; grade needs verified pre-cutoff release time |
| `CandidateAction` | ID; category; title/description; workload; stages; required evidence; prerequisites; contraindications; review requirement; success criterion; catalog version | unique controlled catalog; 1–180 minutes; no final-stage intervention; acyclic defined dependencies |
| `CandidateEvaluation` | action; eligibility status; reason codes | eligibility only; deliberately contains no relevance score or rank |
| `ExpertCase` | blinded case ID; prediction context; observed state; candidate actions; blinding metadata; export version | internal typed record; exporter removes identifiers, model internals, outcomes and exact probability |
| `ExpertActionRating` | case/action/expert; relevance; approval; missing/safety/escalation flags; reason; comment | score in `{-1,0,1,2,3}`; unsafe score requires safety flag; unique case/action/expert |
| `ExpertCaseReview` | case/expert; plan score; plan status; missing actions; safety concerns; comment | plan score 0–3; approved review-state vocabulary |

## Feature lineage

Every available or missing observed feature has `source_table`, `source_column`, `aggregation`, `observation_start`, `observation_end`, `cutoff_day`, and `missing_status`. `observation_end` must be strictly before cutoff. Sensitive attributes, future outcomes, outer labels, withdrawal outcomes and post-cutoff records are rejected.

## Adapter contract

The adapter input is the frozen model's native five tensors (`sequence`, `lengths`, `mask`, `aggregate`, `static`). Its output contains logits, probabilities, class, confidence, predictive entropy, seed disagreement, both embeddings, stage, fold, seeds, checkpoint references and architecture hash. Under the same CPU/float32 path, Phase 2 requires exact equality (`tolerance=0`) with direct frozen-model execution.
