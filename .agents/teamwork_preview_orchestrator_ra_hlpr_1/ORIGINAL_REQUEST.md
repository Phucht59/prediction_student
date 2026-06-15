# Original User Request

## 2026-06-14T17:00:27Z

You are the Project Orchestrator.
Your working directory (metadata folder) is: c:\Huflit\kltn\.agents\teamwork_preview_orchestrator_ra_hlpr_1
Your identity: teamwork_preview_orchestrator_ra_hlpr_1

Please review and execute the requirements in c:\Huflit\kltn\.agents\ORIGINAL_REQUEST.md.
Specifically, your task is to:
1. Decompose the requirements into milestones and create a detailed execution plan in c:\Huflit\kltn\.agents\teamwork_preview_orchestrator_ra_hlpr_1\plan.md.
2. Initialize progress.md and context.md in c:\Huflit\kltn\.agents\teamwork_preview_orchestrator_ra_hlpr_1\.
3. Spawn and manage explorer/worker/reviewer subagents to implement the downstream RA-HLPR system, keeping all existing prediction pipelines and locked test metrics intact.
4. Refactor the existing MLP model to `RiskDiagnosisHead` predicting 6 risks, with transparent weak-labeling rules, BCEWithLogitsLoss + pos_weight, and a dedicated rule explanation file.
5. Create the Intervention Knowledge Base (catalog & mapping CSVs) and a Hybrid Scorer matching the specified formula weights.
6. Build `path_planner.py` for the 4-week learning paths and evaluate risk diagnosis, ranking, and path quality metrics.
7. Implement `scripts/run_recommender_pipeline.py --dataset student-mat` running end-to-end successfully.
8. Output all target CSVs, JSONs, and `recommender_report.md` to `outputs/recommender/`.
9. Document your progress in progress.md.
10. Send a victory claim when all acceptance criteria are fully verified.
