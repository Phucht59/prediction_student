import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
REPORTS_DIR = ROOT_DIR / "reports" / "final"
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"
PREDICTIONS_DIR = REPORTS_DIR / "predictions"
RECOMMENDATIONS_DIR = REPORTS_DIR / "recommendations"
EXPLANATIONS_DIR = REPORTS_DIR / "explanations"
ABLATION_DIR = REPORTS_DIR / "ablation"
MODELS_DIR = ROOT_DIR / "models" / "saved" / "final"
PROCESSED_DIR = DATA_DIR / "processed" / "final"

def ensure_dirs():
    for d in [RAW_DIR, METRICS_DIR, FIGURES_DIR, PREDICTIONS_DIR, RECOMMENDATIONS_DIR, EXPLANATIONS_DIR, ABLATION_DIR, MODELS_DIR, PROCESSED_DIR]:
        d.mkdir(parents=True, exist_ok=True)

FIXED_SEEDS = [42, 123, 155, 156, 2025, 7, 99, 200, 300, 500, 1337]
DEFAULT_SEED = 42

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

POSTGRES_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "dbname": os.getenv("POSTGRES_DB", "student_performance"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres")
}

@dataclass
class DatasetConfig:
    name: str
    raw_file: str
    target_col: str
    kind: str  # 'student' or 'xapi'
    table_name: str
    csv_sep: str

DATASETS = {
    "student-mat": DatasetConfig("student-mat", "student-mat.csv", "G3", "student", "student_mat", ";"),
    "student-por": DatasetConfig("student-por", "student-por.csv", "G3", "student", "student_por", ";"),
    "xapi": DatasetConfig("xapi", "xAPI-Edu-Data.csv", "Class", "xapi", "xapi_edu_data", ",")
}

# Settings
LOCKED_TEST_SIZE = 0.2
CV_FOLDS = 5
OPTUNA_TRIALS = 50

# Thresholds for student-mat/por
STUDENT_G3_3CLASS_BINS = [0, 9, 14, 20] # Low, Medium, High

XAPI_CLASS_MAPPING = {"L": 0, "M": 1, "H": 2}

@dataclass
class TrainingConfig:
    max_epochs: int = 100
    batch_size: int = 32
    patience: int = 15
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    device: str = "cuda"
    use_postgres: bool = False
    use_feature_selection: bool = True
