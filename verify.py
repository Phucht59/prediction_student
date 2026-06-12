import optuna
from src.data_pipeline import load_splits
from src.train_pipeline import objective
from src.config import DATASETS

# Test xAPI
spec = DATASETS["xapi"]
import pandas as pd
from src.config import RAW_DIR
df = pd.read_csv(RAW_DIR / "xAPI-Edu-Data.csv")
from src.data_pipeline import create_and_save_locked_test
create_and_save_locked_test(df, "xapi", "3class")
train_pool, _ = load_splits("xapi", "3class")

# Limit to 50 rows for fast testing
train_pool = train_pool.head(50)

def test_objective(trial):
    return objective(trial, train_pool, spec, "3class", cv_folds=2)

study = optuna.create_study(direction="maximize")
study.optimize(test_objective, n_trials=1)
print("Verification passed successfully!")
