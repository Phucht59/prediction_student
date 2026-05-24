# Student Performance Prediction

This repository is a simple graduation thesis project for predicting student performance using machine learning and deep learning.

The project uses three education datasets so the models are tested on more than one data source:

- Student Performance in Mathematics
- Student Performance in Portuguese
- xAPI Student Academic Performance

## Repository Structure

```text
student-performance-prediction/
├── config.yaml
├── environment.yml
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── notebooks/
├── src/
│   ├── data/
│   ├── models/
│   ├── train/
│   ├── evaluate/
│   ├── recommend/
│   └── utils/
├── results/
│   ├── metrics/
│   ├── figures/
│   └── reports/
├── saved_models/
│   ├── basic/
│   └── deep/
└── scripts/
```

## Create Conda Environment

```bash
conda env create -f environment.yml
conda activate student-performance
```

## Add Datasets

Put the original dataset files in `data/raw/`:

```text
data/raw/student-mat.csv
data/raw/student-por.csv
data/raw/xapi.csv
```

## Run Preprocessing

```bash
python scripts/run_prepare.py
```

This script loads each dataset, cleans column names, creates class labels, encodes categorical features, scales features, and saves train/test splits.

## Train Basic Machine Learning Models

```bash
python scripts/run_train_basic.py
```

This trains Decision Tree, Random Forest, SVM, XGBoost, and Gradient Boosting models with each imbalance method.

Results are saved to:

```text
results/metrics/basic_results.csv
```

Models are saved to:

```text
saved_models/basic/
```

## Train Deep Learning Models

```bash
python scripts/run_train_deep.py
```

This trains CNN-LSTM and CNN-BiLSTM models for each dataset.

Results are saved to:

```text
results/metrics/deep_results.csv
```

Models are saved to:

```text
saved_models/deep/
```

## View Results

```bash
python scripts/run_evaluate.py
```

This creates comparison plots and simple study advice reports.

Figures are saved to:

```text
results/figures/
```

Reports are saved to:

```text
results/reports/
```

## Run Full Pipeline

```bash
python scripts/run_all.py
```

This runs preprocessing, basic model training, deep learning training, and result evaluation in order.

