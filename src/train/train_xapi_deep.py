from __future__ import annotations

import argparse
import copy
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.prepare_xapi import (  # noqa: E402
    CLASS_MAPPING,
    RAW_PATH,
    build_preprocessor,
    get_feature_names,
    read_xapi,
    to_dense,
)
from src.evaluation.metrics import (  # noqa: E402
    CLASS_LABELS,
    classification_metrics,
    classification_probability_metrics,
    save_confusion_matrix_plot,
)
from src.features.feature_selection import (  # noqa: E402
    resolve_columns_by_aliases,
    select_features_supervised,
)
from src.features.imbalance import (  # noqa: E402
    RESAMPLING_STRATEGIES,
    class_distribution,
    resample_raw_mixed_train_data,
    resample_train_data,
)
from src.models.deep_learning import (  # noqa: E402
    CNNBiLSTMTabularClassifier,
    CNNBiLSTMXAPI,
    MLPClassifier,
)
from src.train.deep_utils import compute_class_weights, get_device, save_training_curve, set_seed  # noqa: E402
from src.train.train_deep_classification import upsert_csv  # noqa: E402


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "xapi" / "xapi_behavior"
RESULTS_PATH = PROJECT_ROOT / "reports" / "results" / "xapi_deep_results.csv"
SKIPPED_PATH = PROJECT_ROOT / "reports" / "results" / "xapi_deep_skipped_records.csv"
PRED_DIR = PROJECT_ROOT / "reports" / "results" / "xapi_deep_predictions"
MODEL_DIR = PROJECT_ROOT / "models" / "saved" / "xapi_deep"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "xapi_deep"

CANONICAL_MODEL_NAMES = ("cnn_bilstm_xapi", "mlp_static", "cnn_bilstm_tabular")
MODEL_ALIASES = {
    "cls_xapi": "cnn_bilstm_xapi",
    "cls-xapi": "cnn_bilstm_xapi",
    "CLS-XAPI": "cnn_bilstm_xapi",
    "cnn-bilstm-xapi": "cnn_bilstm_xapi",
    "CNN-BiLSTM-XAPI": "cnn_bilstm_xapi",
    "xapi_behavior_cnn_bilstm": "cnn_bilstm_xapi",
}
MODEL_CHOICES = ("all", *CANONICAL_MODEL_NAMES, *MODEL_ALIASES.keys())
XAPI_RESAMPLING_STRATEGIES = (*RESAMPLING_STRATEGIES, "smotenc")
LOSS_WEIGHTS = ("none", "balanced")
FEATURE_SELECTIONS = ("none", "pearson_chi2")
RESAMPLING_FALLBACKS = ("error", "none", "smote", "random_over")
XAPI_BEHAVIOR_ALIASES = (
    "raisedhands",
    "visitedresources",
    "announcementsview",
    "discussion",
)
XAPI_STATIC_PRIORITY_ALIASES = (
    "studentabsencedays",
    "parentansweringsurvey",
    "parentschoolsatisfaction",
    "relation",
)
CLASS_NAMES_BY_ID = {value: key for key, value in CLASS_MAPPING.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train xAPI deep learning models.")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="cnn_bilstm_xapi")
    parser.add_argument("--loss-weight", choices=["all", *LOSS_WEIGHTS], default="none")
    parser.add_argument("--imbalance-strategy", choices=["all", *XAPI_RESAMPLING_STRATEGIES], default="adasyn")
    parser.add_argument("--feature-selection", choices=FEATURE_SELECTIONS, default="pearson_chi2")
    parser.add_argument(
        "--max-features",
        type=int,
        default=56,
        help="Maximum processed xAPI input features after preprocessing. Use <=0 to keep all.",
    )
    parser.add_argument("--resampling-fallback", choices=RESAMPLING_FALLBACKS, default="smote")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--fusion", choices=["concat", "gated"], default="concat")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="none")
    parser.add_argument("--split-mode", choices=["processed", "holdout80"], default="processed")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--internal-val-size", type=float, default=0.10)
    parser.add_argument("--early-stopping", choices=["val_f1", "none"], default="val_f1")
    parser.add_argument("--model-preset", choices=["default", "simple", "paper", "paper_deep"], default="default")
    parser.add_argument("--swa", action="store_true", help="Use Stochastic Weight Averaging; disables early stopping.")
    parser.add_argument("--conv-channels", type=int, default=None)
    parser.add_argument("--n-conv-blocks", type=int, choices=[1, 2], default=None)
    parser.add_argument("--lstm-hidden", type=int, default=None)
    parser.add_argument("--n-bilstm-layers", type=int, default=None)
    parser.add_argument("--dense-hidden", "--fusion-hidden", dest="dense_hidden", type=int, default=None)
    parser.add_argument("--static-hidden", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--static-layers", type=int, choices=[1, 2], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--sequence-hidden", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=1.0,
        help="Max norm for gradient clipping. 0 = disabled.",
    )
    return parser.parse_args()


def selected(selection: str, values: tuple[str, ...]) -> list[str]:
    return list(values) if selection == "all" else [selection]


def apply_model_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Fill model-capacity defaults for the single-input xAPI CNN-BiLSTM."""
    if args.model_preset == "simple":
        defaults = {
            "conv_channels": 32,
            "n_conv_blocks": 1,
            "lstm_hidden": 32,
            "n_bilstm_layers": 1,
            "static_hidden": 0,
            "static_layers": 1,
            "sequence_hidden": 0,
            "dense_hidden": 64,
            "dropout": 0.40,
            "weight_decay": 1e-3,
        }
    elif args.model_preset == "paper":
        defaults = {
            "conv_channels": 64,
            "n_conv_blocks": 1,
            "lstm_hidden": 64,
            "n_bilstm_layers": 2,
            "static_hidden": 0,
            "static_layers": 1,
            "sequence_hidden": 0,
            "dense_hidden": 128,
            "dropout": 0.30,
            "weight_decay": 1e-4,
        }
    elif args.model_preset == "paper_deep":
        defaults = {
            "conv_channels": 64,
            "n_conv_blocks": 2,
            "lstm_hidden": 64,
            "n_bilstm_layers": 2,
            "static_hidden": 0,
            "static_layers": 1,
            "sequence_hidden": 0,
            "dense_hidden": 128,
            "dropout": 0.30,
            "weight_decay": 1e-4,
        }
    else:
        defaults = {
            "conv_channels": 16,
            "n_conv_blocks": 1,
            "lstm_hidden": 16,
            "n_bilstm_layers": 1,
            "static_hidden": 0,
            "static_layers": 1,
            "sequence_hidden": 0,
            "dense_hidden": 64,
            "dropout": 0.10,
            "weight_decay": 1e-4,
        }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.model_preset in {"paper", "paper_deep"} and args.epochs == 80:
        args.epochs = 150
        args.patience = 20
    return args


def selected_models(selection: str) -> list[str]:
    if selection == "all":
        return list(CANONICAL_MODEL_NAMES)
    return [normalize_model_name(selection)]


def normalize_model_name(model_name: str) -> str:
    normalized_key = model_name.strip().lower()
    return MODEL_ALIASES.get(normalized_key, normalized_key)


def is_hybrid_model(model_name: str) -> bool:
    return False


def load_data() -> dict:
    required = (
        "X_train.npy",
        "X_val.npy",
        "X_test.npy",
        "y_train_class.npy",
        "y_val_class.npy",
        "y_test_class.npy",
        "train_raw.csv",
        "val_raw.csv",
        "test_raw.csv",
        "feature_names.json",
        "metadata.json",
    )
    missing = [filename for filename in required if not (PROCESSED_DIR / filename).exists()]
    if missing:
        raise FileNotFoundError(f"Missing xAPI processed files: {missing}")
    metadata = json.loads((PROCESSED_DIR / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("leakage_checks", {}).get("passed") is not True:
        raise ValueError(f"xAPI leakage check failed: {metadata.get('leakage_checks')}")

    data = {
        "X_train": np.load(PROCESSED_DIR / "X_train.npy", allow_pickle=False),
        "X_val": np.load(PROCESSED_DIR / "X_val.npy", allow_pickle=False),
        "X_test": np.load(PROCESSED_DIR / "X_test.npy", allow_pickle=False),
        "y_train": np.load(PROCESSED_DIR / "y_train_class.npy", allow_pickle=False),
        "y_val": np.load(PROCESSED_DIR / "y_val_class.npy", allow_pickle=False),
        "y_test": np.load(PROCESSED_DIR / "y_test_class.npy", allow_pickle=False),
        "train_raw": pd.read_csv(PROCESSED_DIR / "train_raw.csv"),
        "val_raw": pd.read_csv(PROCESSED_DIR / "val_raw.csv"),
        "test_raw": pd.read_csv(PROCESSED_DIR / "test_raw.csv"),
        "feature_names": json.loads((PROCESSED_DIR / "feature_names.json").read_text(encoding="utf-8")),
        "metadata": metadata,
    }
    validate_base_data(data)
    return data


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    if "Class" not in df.columns:
        raise ValueError("xAPI target column 'Class' was not found.")
    unknown_labels = sorted(set(df["Class"].dropna().unique()) - set(CLASS_MAPPING))
    if unknown_labels:
        raise ValueError(f"xAPI Class contains unsupported labels: {unknown_labels}")
    output = df.copy()
    output["target_class_name"] = output["Class"]
    output["target_class"] = output["Class"].map(CLASS_MAPPING).astype("int64")
    return output


def _distribution(y: pd.Series | np.ndarray) -> dict[str, int]:
    counts = pd.Series(y).value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def build_data_from_raw_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    split_mode: str,
    seed: int,
    split_ratio: dict,
) -> dict:
    feature_columns = [
        column
        for column in df_train.columns
        if column not in {"Class", "target_class_name", "target_class"}
    ]
    preprocessor, numeric_columns, categorical_columns = build_preprocessor(df_train[feature_columns])
    X_train = to_dense(preprocessor.fit_transform(df_train[feature_columns])).astype("float32")
    X_val = to_dense(preprocessor.transform(df_val[feature_columns])).astype("float32")
    X_test = to_dense(preprocessor.transform(df_test[feature_columns])).astype("float32")
    feature_names = get_feature_names(preprocessor, feature_columns)
    contains_target = any(name.split("__")[-1] == "Class" or name == "Class" for name in feature_names)
    leakage_checks = {"contains_target_in_features": bool(contains_target), "passed": not contains_target}
    if not leakage_checks["passed"]:
        raise ValueError(f"xAPI leakage check failed: {leakage_checks}")
    data = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": df_train["target_class"].to_numpy(dtype="int64"),
        "y_val": df_val["target_class"].to_numpy(dtype="int64"),
        "y_test": df_test["target_class"].to_numpy(dtype="int64"),
        "train_raw": df_train.reset_index(drop=True),
        "val_raw": df_val.reset_index(drop=True),
        "test_raw": df_test.reset_index(drop=True),
        "feature_names": feature_names,
        "metadata": {
            "dataset_name": "xapi",
            "scenario": "xapi_behavior",
            "source_file": str(RAW_PATH),
            "target_column": "Class",
            "class_mapping": CLASS_MAPPING,
            "random_seed": seed,
            "split_mode": split_mode,
            "split_ratio": split_ratio,
            "n_rows_total": int(len(df_train) + len(df_val) + len(df_test)),
            "n_train": int(X_train.shape[0]),
            "n_val": int(X_val.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_features_raw": int(len(feature_columns)),
            "n_features_processed": int(X_train.shape[1]),
            "raw_feature_columns": feature_columns,
            "class_distribution_total": _distribution(
                pd.concat([df_train["target_class"], df_val["target_class"], df_test["target_class"]])
            ),
            "class_distribution_train": _distribution(df_train["target_class"]),
            "class_distribution_val": _distribution(df_val["target_class"]),
            "class_distribution_test": _distribution(df_test["target_class"]),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "leakage_checks": leakage_checks,
        },
    }
    validate_base_data(data)
    return data


def _xapi_raw_column_types(df_train: pd.DataFrame, feature_columns: list[str]) -> tuple[list[str], list[str]]:
    X_train = df_train[feature_columns]
    numeric_columns = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]
    return numeric_columns, categorical_columns


def resample_xapi_raw_train_split(
    df_train: pd.DataFrame,
    *,
    seed: int,
) -> tuple[pd.DataFrame, dict[int, int], dict[int, int]]:
    """Apply SMOTENC on the raw xAPI training split before one-hot encoding."""
    feature_columns = [
        column
        for column in df_train.columns
        if column not in {"Class", "target_class_name", "target_class"}
    ]
    _, categorical_columns = _xapi_raw_column_types(df_train, feature_columns)
    X_resampled, y_resampled, before, after = resample_raw_mixed_train_data(
        df_train[feature_columns],
        df_train["target_class"].to_numpy(dtype="int64"),
        categorical_columns,
        seed=seed,
    )
    resampled = X_resampled.copy()
    resampled["target_class"] = y_resampled
    resampled["target_class_name"] = [CLASS_NAMES_BY_ID[int(label)] for label in y_resampled]
    resampled["Class"] = resampled["target_class_name"]
    ordered_columns = list(df_train.columns)
    return resampled[ordered_columns].reset_index(drop=True), before, after


def apply_raw_smotenc_if_requested(
    data: dict,
    *,
    imbalance_strategy: str,
    seed: int,
) -> tuple[dict, str, dict[int, int] | None, dict[int, int] | None, str]:
    if imbalance_strategy != "smotenc":
        return data, "not_applied", None, None, ""

    try:
        resampled_train, before, after = resample_xapi_raw_train_split(
            data["train_raw"],
            seed=seed,
        )
    except Exception as exc:
        raise RuntimeError(
            "SMOTENC raw oversampling failed before one-hot encoding. "
            f"Validation/test were not resampled. Reason: {exc}"
        ) from exc

    rebuilt = build_data_from_raw_splits(
        resampled_train,
        data["val_raw"],
        data["test_raw"],
        split_mode=data["metadata"].get("split_mode", "processed"),
        seed=seed,
        split_ratio=data["metadata"].get("split_ratio", {}),
    )
    rebuilt["metadata"]["raw_resampling"] = {
        "strategy": "smotenc",
        "before_distribution": before,
        "after_distribution": after,
        "note": "SMOTENC was fit only on the raw training split before one-hot encoding.",
    }
    warning = "SMOTENC applied before one-hot encoding on training split only."
    return rebuilt, "smotenc", before, after, warning


def load_holdout80_data(args: argparse.Namespace) -> dict:
    if not 0.05 <= args.test_size <= 0.5:
        raise ValueError("--test-size must be between 0.05 and 0.5.")
    if not 0.0 < args.internal_val_size < 0.5:
        raise ValueError("--internal-val-size must be between 0 and 0.5.")
    raw = add_targets(read_xapi(RAW_PATH))
    train_val, test = train_test_split(
        raw,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=raw["target_class"],
    )
    train, val = train_test_split(
        train_val,
        test_size=args.internal_val_size,
        random_state=args.seed,
        stratify=train_val["target_class"],
    )
    return build_data_from_raw_splits(
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
        split_mode="holdout80",
        seed=args.seed,
        split_ratio={
            "train_pool": round(1.0 - args.test_size, 6),
            "test": args.test_size,
            "internal_train_fraction_of_train_pool": round(1.0 - args.internal_val_size, 6),
            "internal_val_fraction_of_train_pool": args.internal_val_size,
        },
    )


def load_data_for_args(args: argparse.Namespace) -> dict:
    if args.split_mode == "processed":
        data = load_data()
        data["metadata"]["split_mode"] = "processed"
        return data
    if args.split_mode == "holdout80":
        return load_holdout80_data(args)
    raise ValueError(f"Unsupported split mode: {args.split_mode}")


def validate_base_data(data: dict) -> None:
    for split in ("train", "val", "test"):
        X = data[f"X_{split}"]
        y = data[f"y_{split}"]
        if X.ndim != 2:
            raise ValueError(f"xAPI {split}: X must be 2D, got {X.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"xAPI {split}: X/y row mismatch.")
        if X.shape[1] != len(data["feature_names"]):
            raise ValueError(f"xAPI {split}: X width does not match feature_names.")
        if np.isnan(X).any():
            raise ValueError(f"xAPI {split}: X contains NaN.")
        if not set(np.unique(y).tolist()).issubset({0, 1, 2}):
            raise ValueError(f"xAPI {split}: invalid labels.")


def resolve_priority_columns(raw_df: pd.DataFrame, include_behavior: bool = True) -> list[str]:
    aliases = list(XAPI_STATIC_PRIORITY_ALIASES)
    if include_behavior:
        aliases = [*XAPI_BEHAVIOR_ALIASES, *aliases]
    return resolve_columns_by_aliases(raw_df.columns, aliases, required=False)


def base_arrays_for_model(data: dict, model_name: str) -> dict:
    return {
        "arrays": {
            "X_train": data["X_train"].astype("float32"),
            "X_val": data["X_val"].astype("float32"),
            "X_test": data["X_test"].astype("float32"),
            "y_train": data["y_train"].astype("int64"),
            "y_val": data["y_val"].astype("int64"),
            "y_test": data["y_test"].astype("int64"),
        },
        "feature_names": list(data["feature_names"]),
        "sequence_len": None,
        "sequence_columns": [],
        "sequence_scaler": None,
        "warning": "",
    }


def apply_feature_selection(
    *,
    arrays: dict,
    feature_names: list[str],
    data: dict,
    model_name: str,
    feature_selection: str,
    max_features: int,
) -> tuple[dict, list[str], dict]:
    if feature_selection == "none":
        metadata = {
            "method": "none",
            "n_features_before": len(feature_names),
            "n_features_after": len(feature_names),
            "selected_feature_names": feature_names,
        }
        return arrays, feature_names, metadata

    if feature_selection != "pearson_chi2":
        raise ValueError(f"Unsupported feature_selection: {feature_selection}")

    forced_raw_columns = resolve_priority_columns(data["train_raw"], include_behavior=not is_hybrid_model(model_name))
    result = select_features_supervised(
        X_train=arrays["X_train"],
        y_train=arrays["y_train"],
        X_val=arrays["X_val"],
        X_test=arrays["X_test"],
        feature_names=feature_names,
        max_features=max_features if max_features > 0 else None,
        min_features=min(8, len(feature_names)),
        numeric_raw_columns=data["metadata"].get("numeric_columns", []),
        force_raw_columns=forced_raw_columns,
    )
    selected_arrays = dict(arrays)
    selected_arrays["X_train"] = result.X_train
    selected_arrays["X_val"] = result.X_val
    selected_arrays["X_test"] = result.X_test
    return selected_arrays, result.selected_feature_names, result.metadata


def prepare_training_data(
    data: dict,
    model_name: str,
    feature_selection: str = "none",
    max_features: int = 0,
) -> dict:
    model_name = normalize_model_name(model_name)
    prepared = base_arrays_for_model(data, model_name)
    arrays, feature_names, feature_selection_metadata = apply_feature_selection(
        arrays=prepared["arrays"],
        feature_names=prepared["feature_names"],
        data=data,
        model_name=model_name,
        feature_selection=feature_selection,
        max_features=max_features,
    )
    prepared["arrays"] = arrays
    prepared["feature_names"] = feature_names
    prepared["input_dim"] = int(arrays["X_train"].shape[1])
    prepared["feature_selection"] = feature_selection_metadata
    return prepared


def validate_training_arrays(arrays: dict, *, is_hybrid: bool) -> None:
    for split in ("train", "val", "test"):
        X = arrays[f"X_{split}"]
        y = arrays[f"y_{split}"]
        if X.ndim != 2:
            raise ValueError(f"xAPI {split}: input tensor must be 2D, got {X.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"xAPI {split}: input X/y row mismatch.")
        if np.isnan(X).any():
            raise ValueError(f"xAPI {split}: input X contains NaN.")


def resample_training_arrays(
    arrays: dict,
    *,
    is_hybrid: bool,
    strategy: str,
    seed: int,
    fallback: str,
) -> tuple[dict, str, dict[int, int], dict[int, int], str]:
    before_distribution = class_distribution(arrays["y_train"])
    if strategy == "none":
        return arrays, "none", before_distribution, before_distribution, ""

    def attempt_resample(candidate_strategy: str) -> tuple[dict, dict[int, int], dict[int, int]]:
        X_resampled, y_resampled, before, after = resample_train_data(
            arrays["X_train"],
            arrays["y_train"],
            candidate_strategy,
            seed=seed,
        )
        resampled_arrays = dict(arrays)
        resampled_arrays["X_train"] = X_resampled.astype("float32")
        resampled_arrays["y_train"] = y_resampled.astype("int64")
        return resampled_arrays, before, after

    try:
        resampled_arrays, before, after = attempt_resample(strategy)
        return resampled_arrays, strategy, before, after, ""
    except Exception as exc:
        if fallback == "error":
            raise RuntimeError(
                f"Training oversampling '{strategy}' failed after split/feature selection. "
                "Validation and test data were not resampled. "
                f"Reason: {exc}"
            ) from exc
        if fallback == "none":
            warning = f"oversampling {strategy} failed; controlled fallback used: none; reason: {exc}"
            return arrays, "none", before_distribution, before_distribution, warning
        if fallback == "smote":
            try:
                resampled_arrays, before, after = attempt_resample("smote")
                warning = f"oversampling {strategy} failed; controlled fallback used: smote; reason: {exc}"
                return resampled_arrays, "smote", before, after, warning
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"Training oversampling '{strategy}' failed and fallback smote also failed: {fallback_exc}"
                ) from fallback_exc
        if fallback == "random_over":
            try:
                resampled_arrays, before, after = attempt_resample("random_over")
                warning = f"oversampling {strategy} failed; controlled fallback used: random_over; reason: {exc}"
                return resampled_arrays, "random_over", before, after, warning
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"Training oversampling '{strategy}' failed and fallback random_over also failed: {fallback_exc}"
                ) from fallback_exc
        raise ValueError(f"Unsupported resampling fallback: {fallback}")


def make_static_loaders(arrays: dict, batch_size: int) -> dict[str, DataLoader]:
    def dataset_for(split: str) -> TensorDataset:
        return TensorDataset(
            torch.tensor(arrays[f"X_{split}"], dtype=torch.float32),
            torch.tensor(arrays[f"y_{split}"], dtype=torch.long),
        )

    train_dataset = dataset_for("train")
    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "train_eval": DataLoader(train_dataset, batch_size=batch_size, shuffle=False),
        "val": DataLoader(dataset_for("val"), batch_size=batch_size, shuffle=False),
        "test": DataLoader(dataset_for("test"), batch_size=batch_size, shuffle=False),
    }


def build_model(
    model_name: str,
    input_dim: int,
    sequence_len: int | None = None,
    fusion: str = "concat",
    *,
    conv_channels: int = 16,
    n_conv_blocks: int = 1,
    lstm_hidden: int = 16,
    n_bilstm_layers: int = 1,
    static_hidden: int = 64,
    static_layers: int = 2,
    sequence_hidden: int = 16,
    dense_hidden: int = 64,
    dropout: float = 0.10,
):
    model_name = normalize_model_name(model_name)
    if model_name == "mlp_static":
        return MLPClassifier(input_dim=input_dim, hidden_dim=128, dropout=0.3, num_classes=3)
    if model_name == "cnn_bilstm_tabular":
        return CNNBiLSTMTabularClassifier(
            input_dim=input_dim,
            conv_channels=64,
            lstm_hidden=64,
            dropout=0.3,
            num_classes=3,
        )
    if model_name == "cnn_bilstm_xapi":
        return CNNBiLSTMXAPI(
            input_dim=input_dim,
            conv_channels=conv_channels,
            n_conv_blocks=n_conv_blocks,
            lstm_hidden=lstm_hidden,
            n_lstm_layers=n_bilstm_layers,
            dense_hidden=dense_hidden,
            dropout=dropout,
            num_classes=3,
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def unpack_batch(batch, device: torch.device, is_hybrid: bool):
    x, y = batch
    return x.to(device), y.to(device)


def forward_model(model, inputs, is_hybrid: bool):
    return model(inputs)


def train_epoch(model, loader, criterion, optimizer, device, is_hybrid: bool, clip_grad: float = 1.0) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0
    for batch in loader:
        inputs, y = unpack_batch(batch, device, is_hybrid)
        optimizer.zero_grad()
        logits = forward_model(model, inputs, is_hybrid)
        loss = criterion(logits, y)
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()
        rows = int(y.shape[0])
        total_loss += float(loss.item()) * rows
        total_rows += rows
    return total_loss / max(total_rows, 1)


def update_swa_batch_norm(loader, model, device, is_hybrid: bool) -> None:
    update_bn(loader, model, device=device)


@torch.no_grad()
def average_loss(model, loader, criterion, device, is_hybrid: bool) -> float:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    for batch in loader:
        inputs, y = unpack_batch(batch, device, is_hybrid)
        logits = forward_model(model, inputs, is_hybrid)
        loss = criterion(logits, y)
        rows = int(y.shape[0])
        total_loss += float(loss.item()) * rows
        total_rows += rows
    return total_loss / max(total_rows, 1)


@torch.no_grad()
def predict(model, loader, device, is_hybrid: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    y_prob_parts: list[np.ndarray] = []
    for batch in loader:
        inputs, y = unpack_batch(batch, device, is_hybrid)
        logits = forward_model(model, inputs, is_hybrid)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        y_true_parts.append(y.detach().cpu().numpy())
        y_pred_parts.append(predictions.detach().cpu().numpy())
        y_prob_parts.append(probabilities.detach().cpu().numpy())
    return (
        np.concatenate(y_true_parts),
        np.concatenate(y_pred_parts),
        np.concatenate(y_prob_parts),
    )


def save_predictions(path: Path, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "prob_low": y_prob[:, 0],
            "prob_middle": y_prob[:, 1],
            "prob_high": y_prob[:, 2],
        }
    ).to_csv(path, index=False)


def save_confusion_matrix_csv(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    pd.DataFrame(matrix, index=CLASS_LABELS, columns=CLASS_LABELS).to_csv(path)


def train_configuration(
    model_name: str,
    loss_weight: str,
    imbalance_strategy: str,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    model_name = normalize_model_name(model_name)
    set_seed(args.seed)
    device = get_device()
    data = load_data_for_args(args)
    raw_effective_strategy = "not_applied"
    raw_before_distribution = None
    raw_after_distribution = None
    raw_resampling_warning = ""
    if imbalance_strategy == "smotenc":
        (
            data,
            raw_effective_strategy,
            raw_before_distribution,
            raw_after_distribution,
            raw_resampling_warning,
        ) = apply_raw_smotenc_if_requested(
            data,
            imbalance_strategy=imbalance_strategy,
            seed=args.seed,
        )
    prepared = prepare_training_data(
        data,
        model_name,
        feature_selection=args.feature_selection,
        max_features=args.max_features,
    )
    arrays = prepared["arrays"]
    is_hybrid = is_hybrid_model(model_name)
    validate_training_arrays(arrays, is_hybrid=is_hybrid)
    one_hot_resampling_warning = ""
    post_encode_strategy = "none" if imbalance_strategy == "smotenc" else imbalance_strategy
    if post_encode_strategy != "none" and any(
        str(feature_name).startswith("categorical__") for feature_name in prepared["feature_names"]
    ):
        one_hot_resampling_warning = (
            "Oversampling is applied after one-hot encoding on the training set only; "
            "synthetic categorical indicator columns may contain fractional values."
        )
    arrays, effective_strategy, before_distribution, after_distribution, resampling_warning = resample_training_arrays(
        arrays,
        is_hybrid=is_hybrid,
        strategy=post_encode_strategy,
        seed=args.seed,
        fallback=args.resampling_fallback,
    )
    if imbalance_strategy == "smotenc":
        effective_strategy = raw_effective_strategy
        before_distribution = raw_before_distribution or before_distribution
        after_distribution = raw_after_distribution or after_distribution
        resampling_warning = raw_resampling_warning
    validate_training_arrays(arrays, is_hybrid=is_hybrid)
    set_seed(args.seed)

    input_dim = int(arrays["X_train"].shape[1])
    sequence_len = prepared["sequence_len"]
    loaders = make_static_loaders(arrays, args.batch_size)
    model = build_model(
        model_name,
        input_dim,
        sequence_len,
        fusion=args.fusion,
        conv_channels=args.conv_channels,
        n_conv_blocks=args.n_conv_blocks,
        lstm_hidden=args.lstm_hidden,
        n_bilstm_layers=args.n_bilstm_layers,
        static_hidden=args.static_hidden,
        static_layers=args.static_layers,
        sequence_hidden=args.sequence_hidden,
        dense_hidden=args.dense_hidden,
        dropout=args.dropout,
    ).to(device)
    class_weight = compute_class_weights(arrays["y_train"]).to(device) if loss_weight == "balanced" else None
    criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "cosine" and not args.swa:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-5,
        )
    if args.scheduler == "plateau" and not args.swa:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )
    swa_model = AveragedModel(model) if args.swa else None
    swa_scheduler = SWALR(optimizer, swa_lr=3e-4, anneal_epochs=10) if args.swa else None
    swa_start_epoch = max(1, args.epochs // 2) if args.swa else None

    print(
        "Training xAPI deep: model={} loss_weight={} imbalance={} effective={} "
        "feature_selection={} split_mode={} early_stopping={} swa={} seed={} device={}".format(
            model_name,
            loss_weight,
            imbalance_strategy,
            effective_strategy,
            args.feature_selection,
            args.split_mode,
            "disabled" if args.swa else args.early_stopping,
            args.swa,
            args.seed,
            device,
        )
    )
    print(
        "  pipeline=Conv1D->MaxPool1D->Conv1D->MaxPool1D->BiLSTM->Dropout->BiLSTM/Dense "
        "label_smoothing={} scheduler={} preset={} conv={} conv_blocks={} lstm={} layers={} "
        "dense_hidden={} dropout={} weight_decay={} clip_grad={}".format(
            args.label_smoothing,
            args.scheduler,
            args.model_preset,
            args.conv_channels,
            args.n_conv_blocks,
            args.lstm_hidden,
            args.n_bilstm_layers,
            args.dense_hidden,
            args.dropout,
            args.weight_decay,
            args.clip_grad,
        )
    )
    print(
        "  raw split rows: train={} internal_val={} test={}".format(
            data["X_train"].shape[0],
            data["X_val"].shape[0],
            data["X_test"].shape[0],
        )
    )
    print(f"  input_features={input_dim} sequence_split=disabled class_labels=Low/Middle/High")
    print(f"  train class distribution before/after oversampling: {before_distribution} -> {after_distribution}")
    print(f"  tensor shapes: X_train={arrays['X_train'].shape}")
    if prepared["warning"]:
        print(f"  warning: {prepared['warning']}")
    if one_hot_resampling_warning:
        print(f"  warning: {one_hot_resampling_warning}")
    if resampling_warning:
        print(f"  warning: {resampling_warning}")

    best_state: dict[str, torch.Tensor] | None = None
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            is_hybrid,
            clip_grad=args.clip_grad,
        )
        val_loss = float("nan")
        y_true_val, y_pred_val, _ = predict(model, loaders["val"], device, is_hybrid)
        val_metrics = classification_metrics(y_true_val, y_pred_val)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
                "val_recall_weak": val_metrics["recall_weak"],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            if args.early_stopping == "val_f1" and not args.swa:
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if args.swa and epoch >= int(swa_start_epoch):
            swa_model.update_parameters(model)
            swa_scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or patience_counter >= args.patience:
            print(
                "  epoch={:03d} val_f1={:.4f} val_recall_weak={:.4f}{}".format(
                    epoch,
                    val_metrics["f1_macro"],
                    val_metrics["recall_weak"],
                    " swa" if args.swa and epoch >= int(swa_start_epoch) else "",
                )
            )
        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_metrics["f1_macro"])
            else:
                scheduler.step()
        if not args.swa and args.early_stopping == "val_f1" and patience_counter >= args.patience:
            break

    if args.swa:
        update_swa_batch_norm(loaders["train"], swa_model, device, is_hybrid=is_hybrid)
        model_for_eval = swa_model
        y_true_val, y_pred_val, _ = predict(model_for_eval, loaders["val"], device, is_hybrid)
        swa_val_metrics = classification_metrics(y_true_val, y_pred_val)
        best_val_f1 = float(swa_val_metrics["f1_macro"])
        best_epoch = len(history)
    else:
        model_for_eval = model

    if not args.swa and args.early_stopping == "none":
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        best_epoch = len(history)
    if not args.swa and best_state is not None:
        model.load_state_dict(copy.deepcopy(best_state))
        model_for_eval = model

    model_dir = MODEL_DIR / args.split_mode / args.model_preset / model_name / imbalance_strategy / loss_weight
    model_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset": "xapi",
        "scenario": "xapi_behavior",
        "split_mode": args.split_mode,
        "model_preset": args.model_preset,
        "model_name": model_name,
        "loss_weight": loss_weight,
        "imbalance_strategy": imbalance_strategy,
        "imbalance_effective_strategy": effective_strategy,
        "resampling_fallback": args.resampling_fallback,
        "class_distribution_train_before": before_distribution,
        "class_distribution_train_after": after_distribution,
        "seed": args.seed,
        "early_stopping": args.early_stopping,
        "swa": bool(args.swa),
        "swa_start_epoch": int(swa_start_epoch or 0),
        "epochs": args.epochs,
        "epochs_ran": len(history),
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "conv_channels": args.conv_channels,
        "n_conv_blocks": args.n_conv_blocks,
        "lstm_hidden": args.lstm_hidden,
        "n_bilstm_layers": args.n_bilstm_layers,
        "dense_hidden": args.dense_hidden,
        "static_hidden": args.static_hidden,
        "static_layers": args.static_layers,
        "sequence_hidden": args.sequence_hidden,
        "fusion_hidden": args.dense_hidden,
        "dropout": args.dropout,
        "clip_grad": args.clip_grad,
        "fusion": "single_input",
        "label_smoothing": args.label_smoothing,
        "scheduler": args.scheduler,
        "input_dim": input_dim,
        "n_input_features": input_dim,
        "n_static_features": 0,
        "sequence_len": 0,
        "n_sequence_features": 0,
        "sequence_columns": [],
        "input_feature_names": prepared["feature_names"],
        "static_feature_names": [],
        "class_labels": {"0": "Low", "1": "Middle", "2": "High"},
        "model_description": (
            "CNN-BiLSTM-XAPI single-input pipeline over all processed xAPI features; "
            "no separate behavior/static branches."
        ),
        "feature_selection": prepared["feature_selection"],
        "warning": "; ".join(filter(None, [prepared["warning"], one_hot_resampling_warning, resampling_warning])),
        "best_epoch": best_epoch,
        "best_val_f1_macro": best_val_f1,
        "device": str(device),
    }
    torch.save({"model_state_dict": model_for_eval.state_dict(), "config": config}, model_dir / "best_model.pt")
    (model_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=True), encoding="utf-8")
    history_dir = (
        PROJECT_ROOT
        / "reports"
        / "results"
        / "xapi_deep_training_history"
        / args.split_mode
        / args.model_preset
        / model_name
        / imbalance_strategy
    )
    history_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_dir / f"{loss_weight}_history.csv", index=False)
    save_training_curve(
        history,
        FIGURE_DIR / args.split_mode / args.model_preset / model_name / imbalance_strategy / f"{loss_weight}_training_curve.png",
        f"xAPI {model_name} {imbalance_strategy} {loss_weight}",
    )

    rows: list[dict] = []
    for split, loader_key in (("train", "train_eval"), ("val", "val"), ("test", "test")):
        y_true, y_pred, y_prob = predict(model_for_eval, loaders[loader_key], device, is_hybrid)
        metrics = classification_metrics(y_true, y_pred)
        probability_metrics = classification_probability_metrics(y_true, y_prob)
        rows.append(
            {
                "dataset": "xapi",
                "scenario": "xapi_behavior",
                "split_mode": args.split_mode,
                "model_preset": args.model_preset,
                "model_name": model_name,
                "loss_weight": loss_weight,
                "imbalance_strategy": imbalance_strategy,
                "imbalance_effective_strategy": effective_strategy,
                "feature_selection": args.feature_selection,
                "fusion": "single_input",
                "label_smoothing": args.label_smoothing,
                "scheduler": args.scheduler,
                "early_stopping": args.early_stopping,
                "swa": bool(args.swa),
                "split": split,
                "n_rows": int(len(y_true)),
                "n_input_features": input_dim,
                "n_static_features": 0,
                "n_sequence_features": 0,
                "best_epoch": best_epoch,
                "best_val_f1_macro": best_val_f1,
                "seed": args.seed,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "conv_channels": args.conv_channels,
                "n_conv_blocks": args.n_conv_blocks,
                "lstm_hidden": args.lstm_hidden,
                "n_bilstm_layers": args.n_bilstm_layers,
                "dense_hidden": args.dense_hidden,
                "static_hidden": args.static_hidden,
                "static_layers": args.static_layers,
                "sequence_hidden": args.sequence_hidden,
                "fusion_hidden": args.dense_hidden,
                "dropout": args.dropout,
                "clip_grad": args.clip_grad,
                **metrics,
                **probability_metrics,
            }
        )
        prediction_path = (
            PRED_DIR
            / args.split_mode
            / args.model_preset
            / model_name
            / imbalance_strategy
            / loss_weight
            / f"{split}_predictions.csv"
        )
        save_predictions(prediction_path, y_true, y_pred, y_prob)
        if split == "test":
            save_confusion_matrix_plot(
                y_true,
                y_pred,
                FIGURE_DIR / args.split_mode / args.model_preset / model_name / imbalance_strategy / f"{loss_weight}_test_confusion_matrix.png",
                f"xAPI {model_name} {imbalance_strategy} {loss_weight} test confusion matrix",
            )
            save_confusion_matrix_csv(
                PRED_DIR / args.split_mode / args.model_preset / model_name / imbalance_strategy / loss_weight / "test_confusion_matrix.csv",
                y_true,
                y_pred,
            )
    test_row = next(row for row in rows if row["split"] == "test")
    print(
        "  best_epoch={} val_f1={:.4f} test_f1={:.4f} test_acc={:.4f} test_recall_macro={:.4f}".format(
            best_epoch,
            best_val_f1,
            test_row["f1_macro"],
            test_row["accuracy"],
            test_row["recall_macro"],
        )
    )
    return rows, config


def main() -> int:
    args = apply_model_preset(parse_args())
    models = selected_models(args.model)
    loss_weights = selected(args.loss_weight, LOSS_WEIGHTS)
    imbalance_strategies = selected(args.imbalance_strategy, XAPI_RESAMPLING_STRATEGIES)
    all_rows: list[dict] = []
    skipped: list[dict] = []
    for model_name in models:
        for imbalance_strategy in imbalance_strategies:
            for loss_weight in loss_weights:
                try:
                    rows, _ = train_configuration(
                        model_name=model_name,
                        loss_weight=loss_weight,
                        imbalance_strategy=imbalance_strategy,
                        args=args,
                    )
                    all_rows.extend(rows)
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    print(f"SKIP xAPI deep {model_name}/{imbalance_strategy}/{loss_weight}: {reason}")
                    skipped.append(
                        {
                            "dataset": "xapi",
                            "scenario": "xapi_behavior",
                            "model_name": normalize_model_name(model_name),
                            "loss_weight": loss_weight,
                            "imbalance_strategy": imbalance_strategy,
                            "feature_selection": args.feature_selection,
                            "split_mode": args.split_mode,
                            "model_preset": args.model_preset,
                            "fusion": args.fusion,
                            "label_smoothing": args.label_smoothing,
                            "scheduler": args.scheduler,
                            "reason": reason,
                            "traceback": traceback.format_exc(),
                        }
                    )

    upsert_csv(
        RESULTS_PATH,
        all_rows,
        [
            "dataset",
            "scenario",
            "split_mode",
            "model_preset",
            "model_name",
            "loss_weight",
            "imbalance_strategy",
            "feature_selection",
            "fusion",
            "label_smoothing",
            "scheduler",
            "early_stopping",
            "swa",
            "split",
            "seed",
        ],
    )
    upsert_csv(
        SKIPPED_PATH,
        skipped,
        [
            "dataset",
            "scenario",
            "model_name",
            "loss_weight",
            "imbalance_strategy",
            "feature_selection",
            "split_mode",
            "model_preset",
        ],
    )
    print(f"xAPI deep result rows: {len(all_rows)}")
    print(f"xAPI deep skipped rows: {len(skipped)}")
    print(f"Saved to: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
