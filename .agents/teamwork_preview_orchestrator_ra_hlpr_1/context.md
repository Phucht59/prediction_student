# Technical Context — Downstream RA-HLPR Recommender

## Datasets
- Primarily working with `student-mat` dataset.
- Other dataset references: `student-por`, `xapi` (keep compatibility if present).

## Key Files to Investigate / Modify
- `src/models.py` (has MLP and CNN-BiLSTM definitions)
- `src/recommendation.py` (implementation of the recommendation engine)
- `src/eval_recommendation.py` (recommendation evaluation pipeline)
- `src/data_pipeline.py` & `src/train_pipeline.py` (Must not modify preprocessing/resampling)
- `generate_doc.py` (report generation script)

## Output Targets
All recommendation results must be written to `outputs/recommender/`:
- `risk_predictions.csv`
- `recommendation_results.csv`
- `learning_paths.json`
- `recommender_metrics.json`
- `recommender_report.md`
