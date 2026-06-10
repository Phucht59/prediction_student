"""
Configuration for V26 Scientific Hybrid pipeline.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = ROOT_DIR / "reports" / "results"
FIGURES_DIR = ROOT_DIR / "reports" / "figures" / "v26"
MODEL_DIR = ROOT_DIR / "models" / "saved" / "v26"

def ensure_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

FIXED_SEEDS = [42, 123, 155, 156, 2025]

@dataclass
class DatasetSpec:
    name: str
    raw_file: str
    target_col: str
    kind: str  # 'student' or 'xapi'
    csv_sep: Optional[str] = ';'
    
DATASETS = {
    "student-mat": DatasetSpec("student-mat", "student-mat.csv", "G3", "student", ";"),
    "student-por": DatasetSpec("student-por", "student-por.csv", "G3", "student", ";"),
    "xapi": DatasetSpec("xAPI", "xAPI-Edu-Data.csv", "Class", "xapi", ","), # xAPI usually uses ,
}

# 5-class bins: 0-4 (0), 5-8 (1), 9-12 (2), 13-16 (3), 17-20 (4)
STUDENT_G3_BINS_5CLASS = [0, 4, 8, 12, 16, 20]
# 3-class bins: <10 (Low: 0), 10-14 (Medium: 1), >=15 (High: 2)
# Since bins are left-inclusive, right-exclusive in standard pd.cut unless specified
STUDENT_G3_BINS_3CLASS = [0, 9, 14, 20] 

XAPI_CLASS_MAPPING = {"L": 0, "M": 1, "H": 2}

@dataclass
class V26ModelConfig:
    conv_channels: int = 64
    conv_kernel_size: int = 2
    bilstm_hidden: int = 64
    bilstm_layers: int = 1
    attention_dim: int = 64
    context_hidden: int = 128
    fusion_hidden: int = 128
    dropout: float = 0.3

@dataclass
class V26TrainingConfig:
    max_epochs: int = 100
    patience: int = 15
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 32
    focal_gamma: float = 2.0
    ordinal_lambda: float = 0.1
    label_smoothing: float = 0.05
    pretrain_epochs: int = 20 # Stage 1 epochs
