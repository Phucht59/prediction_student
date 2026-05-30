# Student Performance Prediction

Graduation thesis project for predicting student academic performance with Python, PyTorch, scikit-learn, and imbalanced-learn.

The maintained pipeline focuses on:

- preprocessing `student-mat`, `student-por`, `student-combined`, and `xAPI`;
- handling class imbalance with SMOTE, ADASYN, or class weights;
- training CNN-BiLSTM based deep models;
- training traditional baselines for comparison and regression metrics;
- summarizing Accuracy, Precision, Recall, F1, PR-AUC, RMSE, and R2.

## Project Structure

```text
data/
  raw/                 Original CSV files
  processed/           Reproducible processed splits
models/saved/          Generated model checkpoints
reports/
  results/             Raw experiment CSV outputs
  tables/              Summary tables
  figures/             Confusion matrices and training curves
report_temp/           Working thesis reports
scripts/               Maintained runnable pipeline scripts
src/
  data/                Dataset preparation
  features/            Feature selection and imbalance handling
  models/              PyTorch and sklearn models
  train/               Internal training implementations
  evaluation/          Metrics and summary scripts
  recommend/           Feature-importance based study advice
```

Generated outputs under `models/saved/`, `reports/results/`, `reports/tables/`, and `reports/figures/` are reproducible and ignored by Git.

## Setup

```powershell
py -3 -m pip install -r requirements.txt
```

Or with conda:

```powershell
conda env create -f environment.yml
conda activate student-performance
```

## Required Raw Data

Place these files in `data/raw/`:

```text
data/raw/student-mat.csv
data/raw/student-por.csv
data/raw/xAPI-Edu-Data.csv
```

Optional UCI download helper:

```powershell
py scripts/download_dataset.py --uci-student-performance
```

## Main Commands

Prepare data:

```powershell
py scripts/run_prepare.py
```

Train models from one file:

```powershell
py scripts/train_model.py
```

Useful train options:

```powershell
py scripts/train_model.py --mode baseline
py scripts/train_model.py --mode deep
py scripts/train_model.py --dataset student --scenario late
py scripts/train_model.py --dataset xapi --mode deep
```

Test/evaluate results from one file:

```powershell
py scripts/test_model.py
```

Run the full maintained pipeline:

```powershell
py scripts/run_all.py
```

## Optional PostgreSQL Storage

PostgreSQL is optional and is not required for model training. It can be used to store processed student records now and prediction results later when an inference endpoint/script is added.

Create a `.env` file from `.env.example`, then set `DATABASE_URL`:

```powershell
Copy-Item .env.example .env
```

Create the database tables:

```powershell
py scripts/init_postgres.py
```

After running preprocessing, import a processed split set:

```powershell
py scripts/import_processed_to_postgres.py --dataset student-mat --scenario late
```

Use `--dry-run` first to verify row counts without writing:

```powershell
py scripts/import_processed_to_postgres.py --dataset student-mat --scenario late --dry-run
```

The schema includes `student_records` for processed/raw student data and `prediction_results` for future model predictions.

## Direct Deep-Learning Commands

You normally do not need to call the internal `src.train.*` files directly. Use `scripts/train_model.py` unless you need a custom experimental run.

Student datasets custom run:

```powershell
py -m src.train.train_deep --dataset all --scenario all --model clsv2 --oversampling smote --feature-selection pearson_chi2 --max-features 40 --epochs 80 --patience 12 --batch-size 32 --lr 0.001 --seed 42
```

xAPI:

```powershell
py -m src.train.train_deep --dataset xapi --model cnn_bilstm_xapi --oversampling adasyn --feature-selection pearson_chi2 --max-features 56 --epochs 100 --patience 15 --batch-size 16 --lr 0.001 --seed 42
```

## Thesis Scope Notes

- `student-mat` and `student-por` contain G1/G2/G3 period grades, not a long multi-semester transcript. The BiLSTM branch should be described as learning grade progression from available grade periods unless a real multi-semester dataset is added.
- xAPI has no G1/G2. Its `CNNBiLSTMXAPI` / `CLS-XAPI` model uses all processed xAPI features as one input tensor; it no longer splits four behavior counters from static features.
- Recommendation code currently generates study advice from feature importance and rules. Quantitative recommendation metrics require labeled recommendation outcomes, which are not present in the current datasets.
- Optuna is listed in the thesis outline and dependencies, but the current clean final runs use fixed imbalance protocols. Add a dedicated Optuna training script before reporting Optuna results.
