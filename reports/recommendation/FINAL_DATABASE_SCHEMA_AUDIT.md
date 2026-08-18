# Final database schema audit

Live PostgreSQL was inspected before any Phase 11 write. Existing research data was not dropped.

| Logical entity | Existing object | Decision | Notes |
|---|---|---|---|
| Student | `catalog.student` | EXISTS | Unique `external_student_id`. OULAD ids use `OULAD:{id}`. |
| Course | `catalog.course` | EXISTS | Unique `(course_code, presentation)`. |
| Enrollment | `catalog.enrollment` | EXTEND | Added nullable unique `external_enrollment_id`. Existing unique `(student_id, course_id)` kept. |
| Dataset / snapshot | `data.*` | EXISTS | Not required for runtime recommendation writes. |
| Prediction model | `prediction.model` | EXISTS | `hybrid` / Hybrid / `final` already registered. |
| Prediction | `prediction.prediction` | EXISTS | Unique `(run_id, enrollment_id, stage)`. Not used to store 0–3 relevance. |
| Action catalog | `recommendation.action` | EXTEND | Table existed empty. Seeded five final actions. |
| Legacy recommendation | `recommendation.recommendation` + `_item` | DEPRECATED for EBM runtime | Item `score` is CHECK 0–1; cannot store clipped relevance in `[0,3]`. Left intact. |
| Bundle | `recommendation.bundle` | CREATE | Frozen bundle version + checksums. |
| Student state snapshot | `recommendation.state_snapshot` | CREATE | Unique `(enrollment_id, stage, state_version)`. |
| Recommendation run | `recommendation.run` | CREATE | Unique `request_key`. |
| Action score | `recommendation.score` | CREATE | Unique `(run_id, action_id)`; relevance in `[0,3]`. |
| Explanation | `recommendation.explanation` | CREATE | JSONB contributions. |
| Plan | `recommendation.plan` | CREATE | Unique `run_id`. |
| Expert review | none | DEPRECATED | Project has no human review workflow; not invented. |

Migration: `database/migrations/001_recommendation_runtime.sql` (additive, `IF NOT EXISTS`, re-runnable).
