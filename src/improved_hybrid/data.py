import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, ADASYN
import json
from pathlib import Path

from .config import DatasetSpec, RAW_DIR, STUDENT_G3_BINS, XAPI_CLASS_MAPPING
from .features import apply_feature_engineering

def read_raw_data(spec: DatasetSpec) -> pd.DataFrame:
    path = RAW_DIR / spec.raw_file
    if spec.csv_sep is None:
        return pd.read_csv(path, sep=None, engine='python')
    return pd.read_csv(path, sep=spec.csv_sep)

def add_targets(raw: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    data = raw.copy()
    if spec.kind == "student":
        g3 = pd.to_numeric(data["G3"], errors="coerce").fillna(0)
        target = pd.cut(
            g3,
            bins=STUDENT_G3_BINS,
            labels=False,
            include_lowest=True,
        )
        data["target_class"] = target.astype(int)
    else:
        data["target_class"] = data["Class"].map(XAPI_CLASS_MAPPING).astype(int)
        
    # Ordinal normalized target for auxiliary loss
    max_class = spec.n_classes - 1
    data["ordinal_target"] = (data["target_class"] / max_class).astype(np.float32)
    return data

class HybridDataProcessor:
    def __init__(self, seq_cols, num_cols, cat_cols):
        self.seq_cols = seq_cols
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        
        self.seq_imputer = SimpleImputer(strategy='median')
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        
        self.num_scaler = StandardScaler()
        self.cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        
        self.cat_cardinalities = []

    def fit(self, df: pd.DataFrame):
        if self.seq_cols:
            self.seq_imputer.fit(df[self.seq_cols])
            
        if self.num_cols:
            self.num_imputer.fit(df[self.num_cols])
            self.num_scaler.fit(self.num_imputer.transform(df[self.num_cols]))
            
        if self.cat_cols:
            self.cat_imputer.fit(df[self.cat_cols])
            encoded = self.cat_encoder.fit_transform(self.cat_imputer.transform(df[self.cat_cols]))
            # +2 because we have unknown_value=-1 mapped to 0, and max value needs +1 for size
            self.cat_cardinalities = [int(max_val) + 2 for max_val in encoded.max(axis=0)]
            
    def transform(self, df: pd.DataFrame):
        X_seq = np.zeros((len(df), len(self.seq_cols)), dtype=np.float32)
        if self.seq_cols:
            # Scale sequence data as well
            X_seq = self.seq_imputer.transform(df[self.seq_cols])
            # Sequence data usually needs minmax scaling or specific scaling to preserve relations, 
            # but standard scaler is fine for deep learning
            # Wait, let's keep sequence unscaled or minmax scaled? Let's use simple scaling
            X_seq = (X_seq - X_seq.mean(axis=0)) / (X_seq.std(axis=0) + 1e-8)
            X_seq = X_seq.astype(np.float32)

        X_num = np.zeros((len(df), len(self.num_cols)), dtype=np.float32)
        if self.num_cols:
            X_num = self.num_imputer.transform(df[self.num_cols])
            X_num = self.num_scaler.transform(X_num).astype(np.float32)
            
        X_cat = np.zeros((len(df), len(self.cat_cols)), dtype=np.int64)
        if self.cat_cols:
            X_cat = self.cat_imputer.transform(df[self.cat_cols])
            # Shift by +1 so -1 becomes 0 (the unknown bucket)
            X_cat = (self.cat_encoder.transform(X_cat) + 1).astype(np.int64)
            
        y_class = df["target_class"].values.astype(np.int64)
        y_ord = df["ordinal_target"].values.astype(np.float32)
        
        return X_seq, X_num, X_cat, y_class, y_ord

def create_dataloaders(X_seq, X_num, X_cat, y_class, y_ord, batch_size, shuffle=True):
    dataset = TensorDataset(
        torch.tensor(X_seq, dtype=torch.float32),
        torch.tensor(X_num, dtype=torch.float32),
        torch.tensor(X_cat, dtype=torch.long),
        torch.tensor(y_class, dtype=torch.long),
        torch.tensor(y_ord, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def apply_resampling(X_seq, X_num, X_cat, y_class, y_ord, strategy='smote', seed=42):
    if strategy == 'none':
        return X_seq, X_num, X_cat, y_class, y_ord
        
    # Flatten features for SMOTE
    n_samples = len(y_class)
    X_flat = np.hstack([X_seq.reshape(n_samples, -1), X_num, X_cat])
    
    # Need at least 2 samples per class
    counts = np.bincount(y_class)
    min_class_count = counts[counts > 0].min()
    
    if min_class_count < 2:
        return X_seq, X_num, X_cat, y_class, y_ord
        
    k_neighbors = min(5, min_class_count - 1)
    
    if strategy == 'smote':
        sampler = SMOTE(random_state=seed, k_neighbors=k_neighbors)
    elif strategy == 'adasyn':
        sampler = ADASYN(random_state=seed, n_neighbors=k_neighbors)
    else:
        return X_seq, X_num, X_cat, y_class, y_ord
        
    try:
        X_res, y_res = sampler.fit_resample(X_flat, y_class)
        
        # Unflatten
        seq_dim = X_seq.shape[1]
        num_dim = X_num.shape[1]
        
        X_seq_res = X_res[:, :seq_dim]
        X_num_res = X_res[:, seq_dim:seq_dim+num_dim]
        X_cat_res = X_res[:, seq_dim+num_dim:]
        # Categorical values must be integers after SMOTE interpolation
        X_cat_res = np.round(X_cat_res).astype(np.int64)
        
        # Recreate y_ord
        max_class = y_class.max()
        y_ord_res = (y_res / max_class).astype(np.float32)
        
        return X_seq_res, X_num_res, X_cat_res, y_res, y_ord_res
    except Exception as e:
        print(f"Resampling failed: {e}. Returning original.")
        return X_seq, X_num, X_cat, y_class, y_ord
