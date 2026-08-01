# Controlled action catalog

`configs/recommend_hybrid/actions.yaml` is the Phase 2 source of truth for intervention metadata. Version `recommend_hybrid_actions_v1` contains ten active bilingual actions across engagement, assessment, practice, diagnostic, planning, human support, monitoring and consolidation.

Each action declares workload, canonical intervention stages, required evidence, prerequisites, contraindications, mandatory human review, success criterion and active/catalog version. Workload is metadata only in Phase 2; no plan or workload optimizer is implemented.

The validator enforces unique IDs, workload of 1–180 minutes, canonical stages, no intervention at `FINAL_EVALUATION`, defined evidence fields, defined prerequisites and an acyclic dependency graph. `INSTRUCTOR_CONTACT` and `ADVISOR_ESCALATION` always require human review.

`HybridCandidateGenerator` returns one of `ELIGIBLE`, `INELIGIBLE_STAGE`, `MISSING_REQUIRED_EVIDENCE`, `PREREQUISITE_NOT_MET`, `CONTRAINDICATED`, or `REQUIRES_HUMAN_REVIEW` with reason codes. These codes explain eligibility only. There is no relevance score, risk-class shortcut, ranking, Top-K selection or plan generation.

## Phase 3 dataset isolation

The Phase 2 catalog is preserved. Phase 3 declares separate policy action sets. UCI MAT/POR contain nine academic actions and exclude `VLE_ENGAGEMENT`; OULAD contains the ten platform/activity actions declared in its own config. Shared names such as `STUDY_SCHEDULE` retain dataset-specific triggers. An action appearing in both branches does not imply shared thresholds or business policy.
