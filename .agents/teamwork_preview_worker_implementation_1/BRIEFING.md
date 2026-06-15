# BRIEFING — 2026-06-15T10:16:15+07:00

## Mission
Implement the RA-HLPR Refactoring (Phase 1 & Phase 2) for student risk recommendation and document generation, running pipelines on all 3 datasets, and generating the final report.

## 🔒 My Identity
- Archetype: implementer, qa, specialist
- Roles: implementer, qa, specialist
- Working directory: c:\Huflit\kltn\.agents\teamwork_preview_worker_implementation_1\
- Original parent: da19f9da-92c3-4713-82c6-4444ea757405
- Milestone: RA-HLPR Refactoring

## 🔒 Key Constraints
- Do not cheat (no hardcoding, no dummy/facade implementations).
- Do not break the existing CNN-BiLSTM + Context MLP prediction model.
- Do not retrain or modify the main classifier unless necessary.
- RA-HLPR must be a downstream module, receiving inputs from the existing predictions.
- No fabricated metrics. Do not write metrics for datasets that haven't run.
- Do not call it collaborative filtering if there's no user-item interaction.
- Do not call it knowledge graph if no real graph has been built.
- Do not use risks without features in the dataset.

## Current Parent
- Conversation ID: da19f9da-92c3-4713-82c6-4444ea757405
- Updated: 2026-06-15T10:16:15+07:00

## Task Summary
- **What to build**: Phase 1: risk_rules.py, update risk_head.py, intervention_catalog.csv, update hybrid_scorer.py, candidate_generator.py, path_planner.py, explanation.py, recommender_metrics.py, path_quality.py, update run_recommender_pipeline.py. Phase 2: generate_doc.py, outputs/recommender/final_recommender_section.md.
- **Success criteria**: All pipelines run successfully for 3 datasets (`student-mat`, `student-por`, `xapi`), docx generated with updated details, no regressions in predictions, pytest passes, and handoff report generated.
- **Interface contracts**: Downstream integration from existing predictions.
- **Code layout**: Source in `src/`, data in `data/`, pipelines/scripts in `scripts/`, outputs in `outputs/`.

## Key Decisions Made
- Created modular evaluation script to split diagnosis/ranking and path quality.
- Unified the 6 risks for student and 3 risks for xapi.
- Implemented CandidateGenerator to filter interventions.

## Artifact Index
- `src/recommender/risk_rules.py` — Unified 6 academic risks and weak labeling rules.
- `src/recommender/risk_head.py` — Updated to support dynamic output dims.
- `data/recommender/intervention_catalog.csv` — Contains 12 interventions and targets.
- `src/recommender/hybrid_scorer.py` — Updated scoring weights and explanation generator.
- `src/recommender/candidate_generator.py` — Filters candidates before scoring.
- `src/recommender/explanation.py` — Generates friendly explanations in Vietnamese.
- `src/evaluation/recommender_metrics.py` — Diagnosis and ranking metrics.
- `src/evaluation/path_quality.py` — Path quality metrics.
- `outputs/recommender/final_recommender_section.md` — Section 3.5 content and tables in Vietnamese.

## Change Tracker
- **Files modified**: `src/recommender/risk_head.py`, `src/recommender/hybrid_scorer.py`, `src/recommender/knowledge_base.py`, `src/evaluation/recommender_eval.py`, `scripts/run_recommender_pipeline.py`, `generate_doc.py`, `tests/test_recommender.py`, `tests/test_challenger_recommender.py`
- **Build status**: pass (20 tests passed)
- **Pending issues**: None

## Quality Status
- **Build/test result**: 20 passed
- **Lint status**: clean
- **Tests added/modified**: Updated `test_recommender.py` and `test_challenger_recommender.py` to match the new unified risk rules and output dirs.

## Loaded Skills
- None
