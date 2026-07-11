# Production inference schema extension (future design)

The current PostgreSQL schema is for experiment tracking and evaluation.
`ml_predictions.true_label NOT NULL` is appropriate for those evaluated runs,
but a production prediction has no outcome at prediction time. A future,
non-migration proposal is to permit later outcome updates or create an
`ml_prediction_outcomes` table, add run type `production_inference`, and add a
separate inference role/split. This does not alter the frozen scientific schema.
