from __future__ import annotations

import argparse
import json
import inspect
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RAW_PATH = PROJECT_ROOT / "data" / "raw" / "xAPI-Edu-Data.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "xapi" / "xapi_behavior"
CLASS_MAPPING = {"L": 0, "M": 1, "H": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare processed xAPI dataset splits.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stratified splits.")
    return parser.parse_args()


def read_xapi(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing xAPI raw dataset: {path}")
    return pd.read_csv(path, sep=None, engine="python")


def class_distribution(y: pd.Series | np.ndarray) -> dict[str, int]:
    counts = pd.Series(y).value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def build_preprocessor(X_train: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_columns = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in X_train.columns if column not in numeric_columns]
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )
    onehot_params = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        onehot_params["sparse_output"] = False
    else:
        onehot_params["sparse"] = False
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(**onehot_params)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )
    return preprocessor, numeric_columns, categorical_columns


def get_feature_names(preprocessor: ColumnTransformer, raw_feature_columns: list[str]) -> list[str]:
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return raw_feature_columns


def to_dense(array):
    if hasattr(array, "toarray"):
        return array.toarray()
    return np.asarray(array)


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    df = read_xapi(RAW_PATH)
    if "Class" not in df.columns:
        raise ValueError("xAPI target column 'Class' was not found.")

    unknown_labels = sorted(set(df["Class"].dropna().unique()) - set(CLASS_MAPPING))
    if unknown_labels:
        raise ValueError(f"xAPI Class contains unsupported labels: {unknown_labels}")

    df = df.copy()
    df["target_class_name"] = df["Class"]
    df["target_class"] = df["Class"].map(CLASS_MAPPING).astype(int)

    feature_columns = [column for column in df.columns if column not in {"Class", "target_class_name", "target_class"}]
    X = df[feature_columns]
    y = df["target_class"]

    X_train_temp, X_test, y_train_temp, y_test, df_train_temp, df_test = train_test_split(
        X,
        y,
        df,
        test_size=0.15,
        random_state=args.seed,
        stratify=y,
    )
    val_size_from_temp = 0.15 / 0.85
    X_train, X_val, y_train, y_val, df_train, df_val = train_test_split(
        X_train_temp,
        y_train_temp,
        df_train_temp,
        test_size=val_size_from_temp,
        random_state=args.seed,
        stratify=y_train_temp,
    )

    preprocessor, numeric_columns, categorical_columns = build_preprocessor(X_train)
    X_train_processed = to_dense(preprocessor.fit_transform(X_train)).astype("float32")
    X_val_processed = to_dense(preprocessor.transform(X_val)).astype("float32")
    X_test_processed = to_dense(preprocessor.transform(X_test)).astype("float32")

    for split_name, X_split in (
        ("train", X_train_processed),
        ("val", X_val_processed),
        ("test", X_test_processed),
    ):
        if np.isnan(X_split).any():
            raise ValueError(f"xAPI processed {split_name} contains NaN.")

    feature_names = get_feature_names(preprocessor, feature_columns)
    contains_target = any(name.split("__")[-1] == "Class" or name == "Class" for name in feature_names)
    leakage_checks = {
        "contains_target_in_features": bool(contains_target),
        "passed": not contains_target,
    }
    if not leakage_checks["passed"]:
        raise ValueError(f"xAPI leakage check failed: {leakage_checks}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "X_train.npy", X_train_processed)
    np.save(OUTPUT_DIR / "X_val.npy", X_val_processed)
    np.save(OUTPUT_DIR / "X_test.npy", X_test_processed)
    np.save(OUTPUT_DIR / "y_train_class.npy", y_train.to_numpy(dtype="int64"))
    np.save(OUTPUT_DIR / "y_val_class.npy", y_val.to_numpy(dtype="int64"))
    np.save(OUTPUT_DIR / "y_test_class.npy", y_test.to_numpy(dtype="int64"))
    df_train.to_csv(OUTPUT_DIR / "train_raw.csv", index=False)
    df_val.to_csv(OUTPUT_DIR / "val_raw.csv", index=False)
    df_test.to_csv(OUTPUT_DIR / "test_raw.csv", index=False)
    joblib.dump(preprocessor, OUTPUT_DIR / "preprocessor.joblib")
    write_json(OUTPUT_DIR / "feature_names.json", feature_names)

    metadata = {
        "dataset_name": "xapi",
        "scenario": "xapi_behavior",
        "source_file": str(RAW_PATH),
        "target_column": "Class",
        "class_mapping": CLASS_MAPPING,
        "random_seed": args.seed,
        "split_ratio": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "n_rows_total": int(df.shape[0]),
        "n_train": int(X_train_processed.shape[0]),
        "n_val": int(X_val_processed.shape[0]),
        "n_test": int(X_test_processed.shape[0]),
        "n_features_raw": int(len(feature_columns)),
        "n_features_processed": int(X_train_processed.shape[1]),
        "raw_feature_columns": feature_columns,
        "class_distribution_total": class_distribution(y),
        "class_distribution_train": class_distribution(y_train),
        "class_distribution_val": class_distribution(y_val),
        "class_distribution_test": class_distribution(y_test),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "leakage_checks": leakage_checks,
    }
    write_json(OUTPUT_DIR / "metadata.json", metadata)

    print(f"xAPI processed output: {OUTPUT_DIR}")
    print(f"Train/val/test rows: {X_train_processed.shape[0]}/{X_val_processed.shape[0]}/{X_test_processed.shape[0]}")
    print(f"Raw features: {len(feature_columns)}")
    print(f"Processed features: {X_train_processed.shape[1]}")
    print(f"Class distribution total: {metadata['class_distribution_total']}")
    print(f"Leakage checks: {leakage_checks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
