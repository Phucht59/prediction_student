# Cleanup Log

## Cleanup Date

- 2026-06-17

## Final Files Kept

- `README.md`
- `reports/final/final_model_manifest.json`
- `reports/final/final_deep_results_table.csv`
- `reports/final/final_baseline_comparison.csv`
- `reports/final/final_prediction_model_report.md`
- `reports/final/final_thesis_ready_summary.md`
- `reports/final/final_recommender_report.md`
- `reports/final/final_recommender_thesis_summary_vi.md`
- `reports/final/recommender_model_design.md`
- `reports/final/FINAL_PROJECT_STATUS.md`
- `reports/final/LUAN_VAN_HOAN_CHINH_FINAL.docx`
- `outputs/recommender/xapi/`
- `outputs/recommender/student-por/`
- `models/saved/final/`
- `data/raw/` structure. Raw CSV files are not tracked in Git.
- `data/processed/final/`
- Source and pipeline folders needed for final use: `src/`, `scripts/`, `tests/`, `database/`

## Files And Folders Moved To Archive

Moved or retained in archive for audit instead of deletion:

- Old reports: `archive/old_reports/`
- Old experiment scripts: `archive/experiments/scripts/`
- Old source experiments: `archive/experiments/src_experiments/`
- Debug and scratch material: `archive/debug_runs/`

Final cleanup moves performed in this pass:

- `reports/final/README.md` -> `archive/old_reports/README.md` if present
- `reports/final/xapi_3class_final_report.txt` -> `archive/old_reports/xapi_3class_final_report.txt` if present
- `scripts/run_technical_experiments.py` -> `archive/experiments/scripts/run_technical_experiments.py` if present
- `src/experiments/` -> timestamped folder under `archive/experiments/src_experiments/` if present
- `tests/test_technical_experiments.py` -> `archive/experiments/tests/test_technical_experiments.py` if present
- Root-level legacy recommender outputs under `outputs/recommender/` -> `archive/old_reports/outputs/recommender_root_legacy/`
- Stale Student-Mat recommender output -> `archive/old_reports/outputs/recommender_student-mat_pending/`

## Files And Folders Deleted

Safe generated artifacts only:

- Python caches: `__pycache__/`
- Pytest cache: `.pytest_cache/`
- Notebook checkpoints: `.ipynb_checkpoints/`

## Reason For Cleanup

The project is now prepared for thesis submission. Final reports and final runnable pipeline surfaces are kept in the main repo. Old experiments, debug materials, and stale outputs are archived so readers do not confuse them with the selected final model.

## Final Model Status

Prediction final champions are unchanged:

- `student-mat late`: `sequence_cnn_bilstm_only + low_f1_tuned`
- `student-por late`: `sequence_cnn_bilstm_only + low_f1_tuned`
- `student-por midterm`: `sequence_cnn_bilstm_only + argmax`
- `xAPI`: `gated_fusion_v28 + low_f1_tuned`

## Recommender Status

- RA-HLPR is finalized as a prediction-aware and dataset-aware downstream module.
- Latest xAPI and student-por recommender outputs are kept in the final output folders.
- Student-Mat recommender is pending because metadata is missing: `models/saved/final/student-mat_3class_ensemble_features.json`.

## Future Safety Notes

- Do not restore V28/V29/V30/V31/V32 experiment files into `reports/final/` unless explicitly needed for audit.
- Do not use `student-combine` as a final dataset.
- Do not use direct ADASYN on Student data with label-encoded categorical features.
- Do not tune thresholds on locked-test labels.
- Do not use ML baseline outputs as teacher, distillation source, pseudo-labels, baseline probabilities, or feature importance for the deep model/recommender.
- Do not claim the regression head as a main result unless RMSE is substantially improved and validated.
