# Paper-Style Student Performance Prediction

This project is now reduced to a paper-style replication pipeline for the CNN-BiLSTM student performance paper.

It keeps only:

- paper-style preprocessing for `student-mat`, `student-por`, and `xAPI`;
- scikit-learn baselines with imbalanced-learn resampling;
- PyTorch CNN-BiLSTM models matching the paper architecture as closely as the PDF specifies;
- optional PostgreSQL persistence for paper runs and predictions;
- Optuna search scaffolding for later full hyperparameter tuning.

The generated report is intentionally honest: it reports the metrics produced by this local run and does not copy the paper's numbers into the project results.

## Structure

```text
data/raw/                         Original CSV files
data/processed/paper_replication/ Paper-style processed splits
models/saved/paper_replication/   Trained paper-style models
reports/results/                  Paper result CSVs and Optuna smoke output
reports/tables/                   Paper summary table
reports/figures/paper_replication Training curves and confusion matrices
scripts/                          Paper pipeline entrypoints
src/paper_replication/            Paper preprocessing, training, reporting, Optuna
database/schema.sql               Optional PostgreSQL schema
tests/                            Paper pipeline tests
```

## Setup

```powershell
py -3 -m pip install -r requirements.txt
```

Or with conda:

```powershell
conda env create -f environment.yml
conda activate student-performance-paper
```

Raw data must exist at:

```text
data/raw/student-mat.csv
data/raw/student-por.csv
data/raw/xAPI-Edu-Data.csv
```

## Run

Run the paper-style pipeline without full Optuna:

```powershell
py scripts/run_paper_pipeline.py --stage all --epochs 100 --patience 15
```

Run individual stages:

```powershell
py scripts/run_paper_pipeline.py --stage prepare
py scripts/run_paper_pipeline.py --stage baseline
py scripts/run_paper_pipeline.py --stage deep --epochs 100 --patience 15
py scripts/run_paper_pipeline.py --stage report
```

Run only an Optuna smoke check:

```powershell
py scripts/run_paper_optuna.py --dataset student-mat --trials 2 --epochs 3
```

Full Optuna sweeps are intentionally separate:

```powershell
py scripts/run_paper_optuna.py --dataset student-mat --trials 50 --epochs 80
```

## Outputs

- `reports/paper_replication_report.md`
- `reports/results/paper_replication_results.csv`
- `reports/results/paper_replication_predictions.csv`
- `reports/tables/paper_replication_summary.csv`
- `reports/figures/paper_replication/*.png`
- `models/saved/paper_replication/*`

## PostgreSQL

PostgreSQL is optional. If `DATABASE_URL` or `POSTGRES_*` environment variables are configured, report generation attempts to persist `paper_runs` and `paper_predictions`.

If PostgreSQL is not configured, training still runs and the report records that database persistence was skipped.

## Verification

```powershell
py -m compileall src scripts tests
py -m pytest
```
