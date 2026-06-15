import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATASETS, MODELS_DIR, FIXED_SEEDS
from src.data_pipeline import (
    apply_feature_engineering,
    DataPreprocessor,
    FeatureSelector,
    StudentDataset,
    get_sequence_columns,
)
from scripts.run_pipeline import load_or_create_splits

dataset_name = "student-mat"
spec = DATASETS[dataset_name]
train_pool, locked_test = load_or_create_splits(dataset_name, "3class")
best_params_path = MODELS_DIR / f"{dataset_name}_3class_best_params.json"
best_params = json.loads(best_params_path.read_text(encoding="utf-8"))

seed = 42
set_seed(seed)
from sklearn.model_selection import train_test_split
labels = train_pool[spec.target_col].astype(int).to_numpy()
indices = np.arange(len(train_pool))
train_indices, val_indices = train_test_split(
    indices,
    test_size=0.15,
    stratify=labels,
    random_state=seed,
)
train_sub = apply_feature_engineering(train_pool.iloc[train_indices].copy(), spec.kind)

preprocessor = DataPreprocessor(
    target_col=spec.target_col,
    oversample_method=best_params["oversample_method"],
    smote_ratio=best_params.get("smote_ratio", 1.0),
    resampling_k_neighbors=best_params.get("resampling_k_neighbors", 5),
)
train_prep = preprocessor.fit_transform(train_sub)

selector = FeatureSelector(
    target_col=spec.target_col,
    use_feature_selection=True,
    required_features=get_sequence_columns(spec.kind),
)
train_selected = selector.fit_transform(
    train_prep,
    preprocessor.numerical_cols,
    preprocessor.categorical_cols,
)

train_ds = StudentDataset(train_selected, spec.kind, spec.target_col, preprocessor.numerical_cols, preprocessor.categorical_cols)

print("Selected features:", selector.selected_features)
print("train_ds.cat_cols:", train_ds.cat_cols)
print("train_ds.num_cols:", train_ds.num_cols)
print("preprocessor.categorical_cols:", preprocessor.categorical_cols)
print("preprocessor.numerical_cols:", preprocessor.numerical_cols)

# Load checkpoint
model_path = MODELS_DIR / f"{dataset_name}_3class_cnn_bilstm_mlp_seed{seed}.pt"
state_dict = torch.load(model_path, map_location="cpu")
print("Checkpoint embeddings shape:")
for k, v in state_dict.items():
    if "embeddings" in k or "classifier" in k or "context_mlp.0" in k:
        print(f"  {k}: {v.shape}")
