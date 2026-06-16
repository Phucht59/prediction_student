# Cleanup Log

## Final files kept

- `README.md`
- `reports/final/final_model_manifest.json`
- `reports/final/final_deep_results_table.csv`
- `reports/final/final_baseline_comparison.csv`
- `reports/final/final_prediction_model_report.md`
- `reports/final/final_thesis_ready_summary.md`
- `reports/final/LUAN_VAN_HOAN_CHINH_FINAL.docx`
- Existing final support folders kept in `reports/final/`: `explanations/`, `figures/`, `metrics/`, `predictions/`, `recommendations/`
- `outputs/recommender/` because the recommender tests validate these generated artifacts
- Protected project folders kept: `data/raw/`, `src/`, `scripts/`, `tests/`, `database/`

## Files and folders moved to archive

Moved experiment reports from `reports/final/` to `archive/old_reports/`:

- `reports/final/v28/`
- `reports/final/v29/`
- `reports/final/v30/`
- `reports/final/ablation/`
- `reports/final/imbalance/`
- `reports/final/scenarios/`
- `reports/final/baselines/`
- `reports/final/Bao_cao_tien_do.md`
- `reports/final/README.md`
- `reports/final/technical_experiment_report.md`
- `reports/final/student-mat_3class_final_report.txt`
- `reports/final/student-por_3class_final_report.txt`
- `reports/final/xapi_3class_final_report.txt`
- `results/`
- `outputs/experiments/`
- `outputs/v27/`

Moved experiment code and debug workspace files:

- `scripts/run_v28_experiments.py` -> `archive/experiments/scripts/run_v28_experiments.py`
- `scripts/run_v29_experiments.py` -> `archive/experiments/scripts/run_v29_experiments.py`
- `scripts/run_v30_experiments.py` -> `archive/experiments/scripts/run_v30_experiments.py`
- `scripts/update_final_report.py` -> `archive/experiments/scripts/update_final_report.py`
- `src/experiments/v28.py` -> `archive/experiments/src_experiments/v28.py`
- `src/experiments/v29.py` -> `archive/experiments/src_experiments/v29.py`
- `src/experiments/v30.py` -> `archive/experiments/src_experiments/v30.py`
- `scratch/` -> `archive/debug_runs/scratch/`

## Files deleted

- Python/test caches: `__pycache__/`, `.pytest_cache/`
- Temporary Word lock file: `reports/final/~$AN_VAN_HOAN_CHINH_FINAL.docx`
- Old run logs in `logs/`, including Optuna and obsolete ADASYN-era training logs

## Config cleanup

- Updated `config.yaml` so `xapi.paper_ml_imbalance` no longer points to `adasyn`; it now uses `smotenc`.

## Reason for cleanup

The project is now prepared for thesis writing around the final deep model selection. V28, V29, V30, ablation, imbalance, baseline and scenario outputs remain available for audit in `archive/`, but they are no longer mixed with final thesis artifacts under `reports/final/`.

The final model choice is unchanged:

- `student-mat late`: `sequence_cnn_bilstm_only + low_f1_tuned`
- `student-por late`: `sequence_cnn_bilstm_only + low_f1_tuned`
- `student-por midterm`: `sequence_cnn_bilstm_only + argmax`
- `xAPI`: `gated_fusion_v28 + low_f1_tuned`

## Future safety notes

- Do not restore archived V28/V29/V30 reports into `reports/final/` unless they are explicitly needed for audit.
- Do not use `student-combine` as a final dataset.
- Do not use direct ADASYN on Student data with label-encoded categorical features.
- Do not tune thresholds on locked-test labels.
- Do not claim the regression head as a main result unless RMSE is substantially improved and validated.
