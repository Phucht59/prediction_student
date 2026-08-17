"""Train-only preprocessing; CatBoost retains native categorical fields."""
from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NON_FEATURE_COLUMNS = {
    "record_id", "group_id", "global_student_group", "target", "inner_fold", "outer_fold",
}

_CONTROL_TOKENS = ("fold", "split", "partition", "control")

def feature_columns(df: pd.DataFrame):
    cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS and not any(token in c.lower() for token in _CONTROL_TOKENS)]
    if "inner_fold" in cols:
        raise RuntimeError("inner_fold is control metadata and must never be a predictor")
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in cols if c not in numeric]
    return cols, numeric, categorical

def make_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    _, numeric, categorical = feature_columns(df)
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                                   ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ], remainder="drop")

def catboost_frame(df: pd.DataFrame):
    cols, _, categorical = feature_columns(df)
    out = df[cols].copy()
    for col in categorical:
        out[col] = out[col].fillna("Unknown").astype(str)
    return out, categorical
