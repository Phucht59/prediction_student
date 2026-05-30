from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


SCENARIOS = ("mid", "late")
CLASS_MAPPING = {"weak": 0, "average": 1, "good": 2}
CLASS_IDS = (0, 1, 2)
TARGET_COLUMNS = ("G3", "target_regression", "target_class_name", "target_class")
GRADE_ENGINEERED_COLUMNS = ("grade_trend", "grade_velocity")
ENGINEERED_COLUMNS = (
    "grade_trend",
    "grade_velocity",
    "effort_score",
    "support_index",
    "social_risk",
)
SPLIT_RATIO = {"train": 0.70, "validation": 0.15, "test": 0.15}


def load_student_dataset(path: str) -> pd.DataFrame:
    """Load a UCI Student Performance CSV without modifying the source file."""
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {source_path}")

    try:
        data = pd.read_csv(source_path, sep=";")
    except Exception as exc:
        raise RuntimeError(f"Failed to read dataset with sep=';': {source_path}") from exc

    if data.empty:
        raise ValueError(f"Dataset is empty: {source_path}")
    if data.shape[1] <= 1:
        raise ValueError(
            f"Dataset appears to have only one column after sep=';' parsing: {source_path}"
        )
    return data


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add regression and 3-class classification targets derived from G3."""
    if "G3" not in df.columns:
        raise ValueError("Column 'G3' is required to create targets.")

    data = df.copy()
    g3 = pd.to_numeric(data["G3"], errors="coerce")
    if g3.isna().any():
        bad_count = int(g3.isna().sum())
        raise ValueError(f"Column 'G3' contains {bad_count} non-numeric values.")

    data["target_regression"] = g3
    data["target_class_name"] = np.select(
        [g3 < 10, (g3 >= 10) & (g3 < 14), g3 >= 14],
        ["weak", "average", "good"],
        default="unknown",
    )

    if (data["target_class_name"] == "unknown").any():
        bad_count = int((data["target_class_name"] == "unknown").sum())
        raise ValueError(f"Could not assign class label for {bad_count} rows.")

    data["target_class"] = data["target_class_name"].map(CLASS_MAPPING).astype("int64")
    return data


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def _yes_no_series(df: pd.DataFrame, column: str) -> pd.Series:
    values = df[column]
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").fillna(0).astype(float)
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.map({"yes": 1.0, "no": 0.0}).fillna(0.0)


def apply_student_feature_engineering(df: pd.DataFrame, scenario: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create student-domain features before scaling and feature selection."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Invalid scenario '{scenario}'. Expected one of: {SCENARIOS}")

    data = df.copy()
    created: list[str] = []
    warnings: list[str] = []

    def missing(required_columns: list[str]) -> list[str]:
        return [column for column in required_columns if column not in data.columns]

    if scenario == "late":
        required = ["G1", "G2"]
        absent = missing(required)
        if absent:
            warnings.append(f"Skip engineered feature grade_trend/grade_velocity because missing columns: {absent}")
        else:
            g1 = _numeric_series(data, "G1")
            g2 = _numeric_series(data, "G2")
            data["grade_trend"] = g2 - g1
            data["grade_velocity"] = (g2 - g1) / (g1 + 1.0)
            created.extend(["grade_trend", "grade_velocity"])
    else:
        warnings.append(
            f"Skip grade_trend/grade_velocity for scenario={scenario} to avoid using unavailable future grade information."
        )

    required = ["studytime", "absences"]
    absent = missing(required)
    if absent:
        warnings.append(f"Skip engineered feature effort_score because missing columns: {absent}")
    else:
        studytime = _numeric_series(data, "studytime")
        absences = _numeric_series(data, "absences")
        data["effort_score"] = studytime * (1.0 - absences.clip(lower=0) / 30.0)
        created.append("effort_score")

    required = ["schoolsup", "famsup", "paid"]
    absent = missing(required)
    if absent:
        warnings.append(f"Skip engineered feature support_index because missing columns: {absent}")
    else:
        data["support_index"] = sum(_yes_no_series(data, column) for column in required)
        created.append("support_index")

    required = ["famrel", "freetime", "romantic"]
    absent = missing(required)
    if absent:
        warnings.append(f"Skip engineered feature social_risk because missing columns: {absent}")
    else:
        famrel = _numeric_series(data, "famrel")
        freetime = _numeric_series(data, "freetime")
        romantic = _yes_no_series(data, "romantic")
        data["social_risk"] = famrel + freetime - romantic
        created.append("social_risk")

    metadata = {
        "engineered_features_created": created,
        "feature_engineering_warnings": warnings,
    }
    return data, metadata


def get_feature_columns(df: pd.DataFrame, scenario: str) -> list[str]:
    if scenario not in SCENARIOS:
        raise ValueError(f"Invalid scenario '{scenario}'. Expected one of: {SCENARIOS}")

    excluded_columns = set(TARGET_COLUMNS)
    if scenario == "mid":
        excluded_columns.update(["G2", *GRADE_ENGINEERED_COLUMNS])

    return [column for column in df.columns if column not in excluded_columns]


def get_excluded_columns(df: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    feature_set = set(feature_columns)
    return [column for column in df.columns if column not in feature_set]


def split_data(df: pd.DataFrame, feature_columns: list[str], seed: int = 42) -> dict[str, Any]:
    required_columns = {"target_class", "target_regression", *feature_columns}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Missing columns before split: {missing_columns}")

    train_temp, test = train_test_split(
        df,
        test_size=SPLIT_RATIO["test"],
        random_state=seed,
        stratify=df["target_class"],
    )
    validation_size_from_train_temp = SPLIT_RATIO["validation"] / (
        SPLIT_RATIO["train"] + SPLIT_RATIO["validation"]
    )
    train, validation = train_test_split(
        train_temp,
        test_size=validation_size_from_train_temp,
        random_state=seed,
        stratify=train_temp["target_class"],
    )

    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)

    return {
        "X_train_raw": train[feature_columns].copy(),
        "X_val_raw": validation[feature_columns].copy(),
        "X_test_raw": test[feature_columns].copy(),
        "y_train_class": train["target_class"].copy(),
        "y_val_class": validation["target_class"].copy(),
        "y_test_class": test["target_class"].copy(),
        "y_train_reg": train["target_regression"].copy(),
        "y_val_reg": validation["target_regression"].copy(),
        "y_test_reg": test["target_regression"].copy(),
        "df_train_raw": train.copy(),
        "df_val_raw": validation.copy(),
        "df_test_raw": test.copy(),
    }


def _make_one_hot_encoder() -> OneHotEncoder:
    params = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        params["sparse_output"] = False
    else:
        params["sparse"] = False
    return OneHotEncoder(**params)


def get_column_types(X_train_raw: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_columns = X_train_raw.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in X_train_raw.columns if column not in numeric_columns]
    return numeric_columns, categorical_columns


def build_preprocessor(X_train_raw: pd.DataFrame) -> ColumnTransformer:
    numeric_columns, categorical_columns = get_column_types(X_train_raw)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
    preprocessor.metadata_numeric_columns = numeric_columns
    preprocessor.metadata_categorical_columns = categorical_columns
    return preprocessor


def _as_dense_array(data: Any) -> np.ndarray:
    if sparse.issparse(data):
        data = data.toarray()
    array = np.asarray(data, dtype=np.float32)
    if np.isnan(array).any():
        raise ValueError("Transformed feature matrix contains NaN values.")
    return array


def transform_splits(
    preprocessor: ColumnTransformer,
    X_train_raw: pd.DataFrame,
    X_val_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_train = _as_dense_array(preprocessor.fit_transform(X_train_raw))
    X_val = _as_dense_array(preprocessor.transform(X_val_raw))
    X_test = _as_dense_array(preprocessor.transform(X_test_raw))
    return X_train, X_val, X_test


def get_output_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        names: list[str] = []
        for transformer_name, transformer, columns in preprocessor.transformers_:
            if transformer_name == "remainder" or transformer == "drop":
                continue

            column_list = list(columns)
            if transformer_name == "numeric":
                names.extend([f"numeric__{column}" for column in column_list])
                continue

            onehot = transformer.named_steps.get("onehot")
            if onehot is None or not hasattr(onehot, "categories_"):
                names.extend([f"{transformer_name}__{column}" for column in column_list])
                continue

            for column, categories in zip(column_list, onehot.categories_):
                names.extend([f"{transformer_name}__{column}_{category}" for category in categories])
        return names


def processed_feature_names_contain_column(feature_names: list[str], column: str) -> bool:
    for feature_name in feature_names:
        without_prefix = feature_name.split("__", 1)[-1]
        if without_prefix == column or without_prefix.startswith(f"{column}_"):
            return True
    return False


def run_leakage_checks(
    raw_feature_columns: list[str],
    feature_names: list[str],
    scenario: str,
) -> dict[str, bool]:
    if scenario not in SCENARIOS:
        raise ValueError(f"Invalid scenario '{scenario}'. Expected one of: {SCENARIOS}")

    def contains(column: str) -> bool:
        return column in raw_feature_columns or processed_feature_names_contain_column(
            feature_names, column
        )

    expected_allowed_g1 = scenario in {"mid", "late"}
    expected_allowed_g2 = scenario == "late"
    contains_g1 = contains("G1")
    contains_g2 = contains("G2")
    contains_g3 = contains("G3")

    passed = (
        not contains_g3
        and contains_g1 == expected_allowed_g1
        and contains_g2 == expected_allowed_g2
    )

    return {
        "contains_G3_in_features": contains_g3,
        "contains_G1_in_features": contains_g1,
        "contains_G2_in_features": contains_g2,
        "expected_allowed_G1": expected_allowed_g1,
        "expected_allowed_G2": expected_allowed_g2,
        "passed": passed,
    }


def class_distribution(values: pd.Series | np.ndarray) -> dict[str, int]:
    series = pd.Series(values)
    counts = series.value_counts().reindex(CLASS_IDS, fill_value=0).astype(int)
    return {str(class_id): int(counts.loc[class_id]) for class_id in CLASS_IDS}


def save_processed_artifacts(
    *,
    dataset_name: str,
    source_file: str | Path,
    scenario: str,
    random_seed: int,
    output_dir: str | Path,
    full_df: pd.DataFrame,
    split_result: dict[str, Any],
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    preprocessor: ColumnTransformer,
    feature_names: list[str],
    raw_feature_columns: list[str],
    excluded_columns: list[str],
    leakage_checks: dict[str, bool],
    feature_engineering_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / "X_train.npy", X_train)
    np.save(output_path / "X_val.npy", X_val)
    np.save(output_path / "X_test.npy", X_test)

    np.save(
        output_path / "y_train_class.npy",
        split_result["y_train_class"].to_numpy(dtype=np.int64),
    )
    np.save(
        output_path / "y_val_class.npy",
        split_result["y_val_class"].to_numpy(dtype=np.int64),
    )
    np.save(
        output_path / "y_test_class.npy",
        split_result["y_test_class"].to_numpy(dtype=np.int64),
    )
    np.save(
        output_path / "y_train_reg.npy",
        split_result["y_train_reg"].to_numpy(dtype=np.float32),
    )
    np.save(
        output_path / "y_val_reg.npy",
        split_result["y_val_reg"].to_numpy(dtype=np.float32),
    )
    np.save(
        output_path / "y_test_reg.npy",
        split_result["y_test_reg"].to_numpy(dtype=np.float32),
    )

    split_result["df_train_raw"].to_csv(output_path / "train_raw.csv", index=False)
    split_result["df_val_raw"].to_csv(output_path / "val_raw.csv", index=False)
    split_result["df_test_raw"].to_csv(output_path / "test_raw.csv", index=False)
    joblib.dump(preprocessor, output_path / "preprocessor.joblib")

    (output_path / "feature_names.json").write_text(
        json.dumps(feature_names, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    numeric_columns = list(getattr(preprocessor, "metadata_numeric_columns", []))
    categorical_columns = list(getattr(preprocessor, "metadata_categorical_columns", []))
    metadata: dict[str, Any] = {
        "dataset_name": dataset_name,
        "source_file": str(source_file),
        "scenario": scenario,
        "random_seed": random_seed,
        "split_ratio": SPLIT_RATIO,
        "n_rows_total": int(len(full_df)),
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features_raw": int(len(raw_feature_columns)),
        "n_features_processed": int(X_train.shape[1]),
        "raw_feature_columns": raw_feature_columns,
        "excluded_columns": excluded_columns,
        "class_mapping": CLASS_MAPPING,
        "class_distribution_total": class_distribution(full_df["target_class"]),
        "class_distribution_train": class_distribution(split_result["y_train_class"]),
        "class_distribution_val": class_distribution(split_result["y_val_class"]),
        "class_distribution_test": class_distribution(split_result["y_test_class"]),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "leakage_checks": leakage_checks,
        "feature_engineering": feature_engineering_metadata or {
            "engineered_features_created": [],
            "feature_engineering_warnings": [],
        },
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return metadata
