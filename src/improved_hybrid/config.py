import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "improved_hybrid"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "improved_hybrid"
MODEL_DIR = PROJECT_ROOT / "models" / "saved" / "improved_hybrid"

STUDENT_G3_BINS = [-0.1, 4, 8, 12, 16, 20]
STUDENT_G3_CLASS_NAMES = ["0-4", "5-8", "9-12", "13-16", "17-20"]
XAPI_CLASS_MAPPING = {"L": 0, "M": 1, "H": 2}
XAPI_CLASS_NAMES = ["Low", "Middle", "High"]

@dataclass(frozen=True)
class DatasetSpec:
    name: str
    display_name: str
    raw_file: str
    kind: str
    n_classes: int
    class_names: list[str]
    csv_sep: str | None = None

DATASETS: dict[str, DatasetSpec] = {
    "student-mat": DatasetSpec(
        name="student-mat",
        display_name="Student Performance in Mathematics Dataset",
        raw_file="student-mat.csv",
        kind="student",
        n_classes=5,
        class_names=STUDENT_G3_CLASS_NAMES,
        csv_sep=";",
    ),
    "student-por": DatasetSpec(
        name="student-por",
        display_name="Student Performance in Portuguese language Dataset",
        raw_file="student-por.csv",
        kind="student",
        n_classes=5,
        class_names=STUDENT_G3_CLASS_NAMES,
        csv_sep=";",
    ),
    "xapi": DatasetSpec(
        name="xapi",
        display_name="Students' Academic Performance / xAPI Dataset",
        raw_file="xAPI-Edu-Data.csv",
        kind="xapi",
        n_classes=3,
        class_names=XAPI_CLASS_NAMES,
        csv_sep=None,
    ),
}

@dataclass
class HybridModelConfig:
    conv_channels: int = 64
    conv_kernel_size: int = 2
    bilstm_hidden: int = 64
    bilstm_layers: int = 1
    attention_dim: int = 64
    context_hidden: int = 128
    fusion_hidden: int = 128
    dropout: float = 0.3

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    label_smoothing: float = 0.05
    focal_gamma: float = 2.0
    ordinal_lambda: float = 0.1
    max_epochs: int = 100
    patience: int = 15

def ensure_dirs() -> None:
    for directory in (PROCESSED_DIR, RESULTS_DIR, TABLES_DIR, FIGURES_DIR, MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)
