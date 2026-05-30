from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.deep_dataset import (  # noqa: E402
    build_grade_sequence_from_raw,
    load_processed_classification_data,
    remove_grade_features_from_X,
)
from src.evaluation.metrics import (  # noqa: E402
    classification_metrics,
    classification_probability_metrics,
    save_confusion_matrix_plot,
)
from src.features.feature_selection import feature_refers_to_raw_column, select_features_supervised  # noqa: E402
from src.features.imbalance import RESAMPLING_STRATEGIES, class_distribution, resample_train_data  # noqa: E402
from src.models.deep_learning import (  # noqa: E402
    CNNBiLSTMTabularClassifier,
    HybridCNNBiLSTMClassifier,
    MLPClassifier,
    StudentCNNBiLSTMV2,
)
from src.train.deep_utils import (  # noqa: E402
    EarlyStopping,
    compute_average_loss,
    compute_class_weights,
    evaluate_model,
    get_device,
    make_tensor_dataset_for_hybrid,
    make_tensor_dataset_for_static,
    predict_model,
    save_training_curve,
    set_seed,
    train_one_epoch,
)


DATASETS = ("student-mat", "student-por", "student-combined")
SCENARIOS = ("mid", "late")
MODEL_NAMES = ("mlp_static", "cnn_bilstm_tabular", "hybrid_cnn_bilstm", "clsv2")
MODEL_SELECTIONS = ("auto", "all", *MODEL_NAMES)
LOSS_WEIGHTS = ("none", "balanced")
FEATURE_SELECTIONS = ("none", "pearson_chi2")
RESAMPLING_FALLBACKS = ("error", "none", "smote", "random_over")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PyTorch deep classification models.")
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument("--scenario", choices=["all", *SCENARIOS], default="all")
    parser.add_argument("--model", choices=MODEL_SELECTIONS, default="auto")
    parser.add_argument("--loss-weight", choices=["all", *LOSS_WEIGHTS], default="all")
    parser.add_argument("--feature-selection", choices=FEATURE_SELECTIONS, default="none")
    parser.add_argument("--max-features", type=int, default=0)
    parser.add_argument("--imbalance-strategy", "--oversampling", dest="imbalance_strategy", choices=RESAMPLING_STRATEGIES, default="smote")
    parser.add_argument("--resampling-fallback", choices=RESAMPLING_FALLBACKS, default="smote")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--n-bilstm-layers", type=int, default=1)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    return parser.parse_args()


def selected(selection: str, values: tuple[str, ...]) -> list[str]:
    return list(values) if selection == "all" else [selection]


def selected_models(selection: str) -> list[str]:
    if selection == "all":
        return list(MODEL_NAMES)
    return [selection]


def resolve_models_for_scenario(model_selections: list[str], scenario: str) -> list[str]:
    resolved: list[str] = []
    for model_name in model_selections:
        if model_name == "auto":
            resolved.append("clsv2")
        else:
            resolved.append(model_name)
    return list(dict.fromkeys(resolved))


def upsert_csv(path: Path, rows: list[dict], key_columns: list[str]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if "scenario" in new_df.columns:
        new_df = new_df[new_df["scenario"].ne("early")].copy()
    if path.exists():
        existing = pd.read_csv(path)
        if "scenario" in existing.columns:
            existing = existing[existing["scenario"].ne("early")].copy()
        for column in key_columns:
            if column not in existing.columns:
                existing[column] = ""
            if column not in new_df.columns:
                new_df[column] = ""
        if not existing.empty:
            new_keys = set(map(tuple, new_df[key_columns].astype(str).to_numpy()))
            keep_mask = [
                tuple(row) not in new_keys
                for row in existing[key_columns].astype(str).to_numpy()
            ]
            existing = existing.loc[keep_mask]
        output = pd.concat([existing, new_df], ignore_index=True)
    else:
        output = new_df
    output.to_csv(path, index=False)


def append_skip(skipped_rows: list[dict], dataset: str, scenario: str, model_name: str, loss_weight: str, reason: str, trace: str = "") -> None:
    skipped_rows.append(
        {
            "dataset": dataset,
            "scenario": scenario,
            "model_name": model_name,
            "loss_weight": loss_weight,
            "reason": reason,
            "traceback": trace,
        }
    )


def build_static_loaders(data: dict, batch_size: int):
    train_dataset = make_tensor_dataset_for_static(data["X_train"], data["y_train"])
    val_dataset = make_tensor_dataset_for_static(data["X_val"], data["y_val"])
    test_dataset = make_tensor_dataset_for_static(data["X_test"], data["y_test"])
    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "train_eval": DataLoader(train_dataset, batch_size=batch_size, shuffle=False),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    }


def apply_static_feature_selection(data: dict, args: argparse.Namespace) -> tuple[dict, list[str], dict]:
    feature_names = list(data["feature_names"])
    if args.feature_selection == "none":
        return data, feature_names, {
            "method": "none",
            "n_features_before": len(feature_names),
            "n_features_after": len(feature_names),
        }
    if args.feature_selection != "pearson_chi2":
        raise ValueError(f"Unsupported feature_selection: {args.feature_selection}")

    forced_raw_columns = [
        column
        for column in ("G1", "G2", "grade_trend", "grade_velocity")
        if column in data["metadata"].get("raw_feature_columns", [])
    ]
    result = select_features_supervised(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_val=data["X_val"],
        X_test=data["X_test"],
        feature_names=feature_names,
        max_features=args.max_features if args.max_features > 0 else None,
        min_features=min(8, len(feature_names)),
        numeric_raw_columns=data["metadata"].get("numeric_columns", []),
        force_raw_columns=forced_raw_columns,
    )
    selected_data = dict(data)
    selected_data["X_train"] = result.X_train
    selected_data["X_val"] = result.X_val
    selected_data["X_test"] = result.X_test
    selected_data["feature_names"] = result.selected_feature_names
    return selected_data, result.selected_feature_names, result.metadata


def _resample_arrays(X_train, y_train, strategy: str, seed: int, fallback: str):
    before_distribution = class_distribution(y_train)
    if strategy == "none":
        return X_train, y_train, "none", before_distribution, before_distribution, ""
    try:
        X_resampled, y_resampled, before, after = resample_train_data(
            X_train,
            y_train,
            strategy,
            seed=seed,
        )
        return X_resampled.astype("float32"), y_resampled.astype("int64"), strategy, before, after, ""
    except Exception as exc:
        if fallback == "error":
            raise RuntimeError(
                f"Training oversampling '{strategy}' failed after split/feature selection. Reason: {exc}"
            ) from exc
        if fallback == "none":
            return X_train, y_train, "none", before_distribution, before_distribution, (
                f"oversampling {strategy} failed; controlled fallback used: none; reason: {exc}"
            )
        if fallback in {"smote", "random_over"}:
            try:
                X_resampled, y_resampled, before, after = resample_train_data(
                    X_train,
                    y_train,
                    fallback,
                    seed=seed,
                )
                return X_resampled.astype("float32"), y_resampled.astype("int64"), fallback, before, after, (
                    f"oversampling {strategy} failed; controlled fallback used: {fallback}; reason: {exc}"
                )
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"Training oversampling '{strategy}' failed and fallback {fallback} also failed: {fallback_exc}"
                ) from fallback_exc
        raise ValueError(f"Unsupported resampling fallback: {fallback}")


def resample_static_training_data(data: dict, args: argparse.Namespace) -> tuple[dict, str, dict, dict, str]:
    X_train, y_train, effective, before, after, warning = _resample_arrays(
        data["X_train"],
        data["y_train"],
        args.imbalance_strategy,
        args.seed,
        args.resampling_fallback,
    )
    resampled = dict(data)
    resampled["X_train"] = X_train
    resampled["y_train"] = y_train
    return resampled, effective, before, after, warning


def find_grade_representation_indices(feature_names: list[str]) -> tuple[list[int], list[str]]:
    raw_columns = ("G1", "G2", "grade_trend", "grade_velocity")
    indices: list[int] = []
    names: list[str] = []
    for index, feature_name in enumerate(feature_names):
        if any(feature_refers_to_raw_column(feature_name, column) for column in raw_columns):
            indices.append(index)
            names.append(feature_name)
    return indices, names


def add_grade_representation_arrays(data: dict) -> tuple[dict, int, list[str]]:
    indices, names = find_grade_representation_indices(list(data["feature_names"]))
    output = dict(data)
    for split in ("train", "val", "test"):
        X = output[f"X_{split}"]
        if indices:
            output[f"grade_{split}"] = X[:, indices].astype("float32")
        else:
            output[f"grade_{split}"] = np.zeros((X.shape[0], 0), dtype=np.float32)
    return output, len(indices), names


def build_rich_grade_sequence_from_raw(raw_df: pd.DataFrame, scenario: str) -> np.ndarray:
    """Build a compact grade trajectory sequence for CLSV2 without using G3."""
    n_rows = int(len(raw_df))
    if scenario == "mid":
        required_columns = ["G1"]
    elif scenario == "late":
        required_columns = ["G1", "G2"]
    else:
        raise ValueError(f"Invalid scenario '{scenario}'.")

    missing = [column for column in required_columns if column not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing grade columns for CLSV2 grade sequence in {scenario}: {missing}")

    g1 = raw_df["G1"].astype("float32").to_numpy()
    if scenario == "mid":
        sequence = np.zeros((n_rows, 1, 4), dtype=np.float32)
        sequence[:, 0, 0] = g1
        sequence[:, 0, 2] = g1 / 20.0
        return sequence

    g2 = raw_df["G2"].astype("float32").to_numpy()
    trend = g2 - g1
    velocity = trend / (g1 + 1.0)
    sequence = np.zeros((n_rows, 2, 4), dtype=np.float32)
    sequence[:, 0, 0] = g1
    sequence[:, 0, 2] = g1 / 20.0
    sequence[:, 1, 0] = g2
    sequence[:, 1, 1] = trend
    sequence[:, 1, 2] = g2 / 20.0
    sequence[:, 1, 3] = velocity
    if np.isnan(sequence).any():
        raise ValueError(f"CLSV2 grade sequence for {scenario} contains NaN values.")
    return sequence.astype("float32")


def add_rich_grade_sequence_arrays(data: dict, scenario: str) -> tuple[dict, tuple[int, int], list[str]]:
    output = dict(data)
    for split in ("train", "val", "test"):
        output[f"grade_{split}"] = build_rich_grade_sequence_from_raw(output[f"{split}_raw"], scenario)
    grade_shape = tuple(output["grade_train"].shape[1:])
    names = [
        "t1_G1",
        "t1_zero_trend",
        "t1_G1_over_20",
        "t1_zero_velocity",
    ]
    if grade_shape[0] >= 2:
        names.extend(["t2_G2", "t2_G2_minus_G1", "t2_G2_over_20", "t2_grade_velocity"])
    return output, (int(grade_shape[0]), int(grade_shape[1]) if grade_shape else 0), names


def resample_hybrid_training_data(hybrid_data: dict, args: argparse.Namespace) -> tuple[dict, str, dict, dict, str]:
    grade_shape = hybrid_data["grade_train"].shape[1:]
    grade_flat = hybrid_data["grade_train"].reshape(hybrid_data["grade_train"].shape[0], -1)
    static_dim = int(hybrid_data["X_train"].shape[1])
    combined = np.concatenate([hybrid_data["X_train"], grade_flat], axis=1)
    X_resampled, y_resampled, effective, before, after, warning = _resample_arrays(
        combined,
        hybrid_data["y_train"],
        args.imbalance_strategy,
        args.seed,
        args.resampling_fallback,
    )
    resampled = dict(hybrid_data)
    resampled["X_train"] = X_resampled[:, :static_dim].astype("float32")
    if grade_flat.shape[1] == 0:
        resampled["grade_train"] = np.zeros((len(y_resampled), *grade_shape), dtype=np.float32)
    else:
        resampled["grade_train"] = X_resampled[:, static_dim:].reshape((-1, *grade_shape)).astype("float32")
    resampled["y_train"] = y_resampled.astype("int64")
    return resampled, effective, before, after, warning


def build_hybrid_data(data: dict, scenario: str) -> tuple[dict, int, int]:
    X_train_static, _, train_removed = remove_grade_features_from_X(
        data["X_train"],
        data["feature_names"],
    )
    X_val_static, _, val_removed = remove_grade_features_from_X(
        data["X_val"],
        data["feature_names"],
    )
    X_test_static, _, test_removed = remove_grade_features_from_X(
        data["X_test"],
        data["feature_names"],
    )
    if train_removed != val_removed or train_removed != test_removed:
        raise ValueError("Hybrid grade feature removal produced inconsistent indices.")

    grade_train = build_grade_sequence_from_raw(data["train_raw"], scenario)
    grade_val = build_grade_sequence_from_raw(data["val_raw"], scenario)
    grade_test = build_grade_sequence_from_raw(data["test_raw"], scenario)

    hybrid_data = {
        "X_train": X_train_static,
        "X_val": X_val_static,
        "X_test": X_test_static,
        "grade_train": grade_train,
        "grade_val": grade_val,
        "grade_test": grade_test,
        "y_train": data["y_train"],
        "y_val": data["y_val"],
        "y_test": data["y_test"],
    }
    return hybrid_data, int(X_train_static.shape[1]), int(grade_train.shape[1])


def build_hybrid_loaders(hybrid_data: dict, batch_size: int):
    train_dataset = make_tensor_dataset_for_hybrid(
        hybrid_data["X_train"],
        hybrid_data["grade_train"],
        hybrid_data["y_train"],
    )
    val_dataset = make_tensor_dataset_for_hybrid(
        hybrid_data["X_val"],
        hybrid_data["grade_val"],
        hybrid_data["y_val"],
    )
    test_dataset = make_tensor_dataset_for_hybrid(
        hybrid_data["X_test"],
        hybrid_data["grade_test"],
        hybrid_data["y_test"],
    )
    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "train_eval": DataLoader(train_dataset, batch_size=batch_size, shuffle=False),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    }


def build_model(
    model_name: str,
    input_dim: int,
    args: argparse.Namespace,
    grade_seq_len: int | None = None,
    grade_input_dim: int | None = None,
):
    if model_name == "mlp_static":
        return MLPClassifier(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            num_classes=3,
        )
    if model_name == "cnn_bilstm_tabular":
        return CNNBiLSTMTabularClassifier(
            input_dim=input_dim,
            conv_channels=64,
            lstm_hidden=max(args.hidden_dim // 2, 16),
            dropout=args.dropout,
            num_classes=3,
        )
    if model_name == "hybrid_cnn_bilstm":
        if grade_seq_len is None:
            raise ValueError("grade_seq_len is required for hybrid_cnn_bilstm.")
        return HybridCNNBiLSTMClassifier(
            static_input_dim=input_dim,
            grade_seq_len=grade_seq_len,
            conv_channels=32,
            lstm_hidden=max(args.hidden_dim // 2, 16),
            static_hidden=max(args.hidden_dim // 2, 32),
            dropout=args.dropout,
            num_classes=3,
        )
    if model_name == "clsv2":
        return StudentCNNBiLSTMV2(
            input_dim=input_dim,
            grade_seq_len=int(grade_seq_len or 0),
            grade_input_dim=int(grade_input_dim or 0),
            conv_channels=64,
            lstm_hidden=max(args.hidden_dim // 4, 16),
            n_lstm_layers=int(getattr(args, "n_bilstm_layers", 1)),
            grade_hidden=max(args.hidden_dim // 4, 16),
            fusion_hidden=max(args.hidden_dim // 2, 32),
            dropout=args.dropout,
            num_classes=3,
        )
    raise ValueError(f"Unknown model_name '{model_name}'.")


def save_predictions(path: Path, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "prob_weak": y_prob[:, 0],
            "prob_average": y_prob[:, 1],
            "prob_good": y_prob[:, 2],
        }
    ).to_csv(path, index=False)


def create_config(
    dataset: str,
    scenario: str,
    model_name: str,
    loss_weight: str,
    args: argparse.Namespace,
    device: torch.device,
    n_features: int,
    grade_seq_len: int | None,
) -> dict:
    return {
        "dataset": dataset,
        "scenario": scenario,
        "model_name": model_name,
        "loss_weight": loss_weight,
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "n_bilstm_layers": int(getattr(args, "n_bilstm_layers", 1)),
        "mixup_alpha": float(getattr(args, "mixup_alpha", 0.0)),
        "device": str(device),
        "n_features": n_features,
        "grade_seq_len": grade_seq_len,
        "selection_metric": "validation f1_macro",
    }


def train_configuration(
    *,
    dataset: str,
    scenario: str,
    model_name: str,
    loss_weight: str,
    data: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict], dict]:
    set_seed(args.seed)
    is_legacy_hybrid = model_name == "hybrid_cnn_bilstm"
    is_clsv2 = model_name == "clsv2"
    is_hybrid = is_legacy_hybrid or is_clsv2
    feature_selection_metadata = {
        "method": "not_applied_to_hybrid" if is_legacy_hybrid else args.feature_selection,
        "reason": "hybrid_cnn_bilstm keeps G1/G2 in a separate pseudo-sequence for explicit comparison"
        if is_legacy_hybrid
        else "",
    }

    if is_legacy_hybrid:
        hybrid_data, input_dim, grade_seq_len = build_hybrid_data(data, scenario)
        hybrid_data, effective_strategy, before_distribution, after_distribution, resampling_warning = resample_hybrid_training_data(
            hybrid_data,
            args,
        )
        set_seed(args.seed)
        loaders = build_hybrid_loaders(hybrid_data, args.batch_size)
        model = build_model(model_name, input_dim, args, grade_seq_len=grade_seq_len)
        n_features = input_dim + grade_seq_len
        y_for_class_weight = hybrid_data["y_train"]
        grade_feature_names: list[str] = ["G1/G2 raw pseudo-sequence"]
    elif is_clsv2:
        selected_data, _, feature_selection_metadata = apply_static_feature_selection(data, args)
        selected_data, grade_shape, grade_feature_names = add_rich_grade_sequence_arrays(selected_data, scenario)
        selected_data, effective_strategy, before_distribution, after_distribution, resampling_warning = resample_hybrid_training_data(
            selected_data,
            args,
        )
        set_seed(args.seed)
        loaders = build_hybrid_loaders(selected_data, args.batch_size)
        input_dim = int(selected_data["X_train"].shape[1])
        grade_seq_len, grade_input_dim = grade_shape
        model = build_model(
            model_name,
            input_dim,
            args,
            grade_seq_len=grade_seq_len,
            grade_input_dim=grade_input_dim,
        )
        n_features = input_dim
        y_for_class_weight = selected_data["y_train"]
    else:
        selected_data, _, feature_selection_metadata = apply_static_feature_selection(data, args)
        selected_data, effective_strategy, before_distribution, after_distribution, resampling_warning = resample_static_training_data(
            selected_data,
            args,
        )
        set_seed(args.seed)
        loaders = build_static_loaders(selected_data, args.batch_size)
        input_dim = int(selected_data["X_train"].shape[1])
        grade_seq_len = None
        model = build_model(model_name, input_dim, args)
        n_features = input_dim
        y_for_class_weight = selected_data["y_train"]
        grade_feature_names = []

    model = model.to(device)
    base_variant_name = loss_weight if float(args.mixup_alpha) <= 0.0 else f"{loss_weight}_mixup_{float(args.mixup_alpha):g}"
    variant_name = base_variant_name if args.imbalance_strategy == "none" else f"{base_variant_name}_{effective_strategy}"
    if loss_weight == "balanced":
        class_weights = compute_class_weights(y_for_class_weight).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    early_stopping = EarlyStopping(patience=args.patience, mode="max")

    history: list[dict] = []
    print(f"Training deep: {dataset}/{scenario}/{model_name}/{loss_weight}")
    print(
        "  classes=3 imbalance={} effective={} seed={}".format(
            args.imbalance_strategy,
            effective_strategy,
            args.seed,
        )
    )
    print(f"  train class distribution before/after oversampling: {before_distribution} -> {after_distribution}")
    if resampling_warning:
        print(f"  warning: {resampling_warning}")
    if is_legacy_hybrid:
        print("  note: hybrid_cnn_bilstm uses G1/G2 as a short pseudo-sequence for comparison, not a true temporal series.")
        print(
            "  tensor shapes: X_static_train={} X_sequence_train={} y_train={}".format(
                hybrid_data["X_train"].shape,
                hybrid_data["grade_train"].shape,
                hybrid_data["y_train"].shape,
            )
        )
    elif is_clsv2:
        print("  note: clsv2 uses the full feature vector plus a rich grade sequence branch when grades are available.")
        print(f"  grade_sequence_shape={(grade_seq_len, grade_input_dim)} grade_features={grade_feature_names}")
        print(
            "  tensor shapes: X_static_train={} X_grade_train={} y_train={}".format(
                selected_data["X_train"].shape,
                selected_data["grade_train"].shape,
                selected_data["y_train"].shape,
            )
        )
    else:
        print(
            "  static_features={} feature_selection={}".format(
                input_dim,
                feature_selection_metadata.get("method", args.feature_selection),
            )
        )
        print(
            "  tensor shapes: X_static_train={} X_sequence_train=None y_train={}".format(
                selected_data["X_train"].shape,
                selected_data["y_train"].shape,
            )
        )
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            is_hybrid=is_hybrid,
            mixup_alpha=float(args.mixup_alpha),
            num_classes=3,
        )
        val_loss = compute_average_loss(
            model,
            loaders["val"],
            criterion,
            device,
            is_hybrid=is_hybrid,
        )
        val_metrics = evaluate_model(
            model,
            loaders["val"],
            device,
            is_hybrid=is_hybrid,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
                "val_recall_weak": val_metrics["recall_weak"],
            }
        )

        should_stop = early_stopping.step(val_metrics["f1_macro"], model, epoch)
        if epoch == 1 or epoch % 10 == 0 or should_stop:
            print(
                "  epoch={:03d} train_loss={:.4f} val_loss={:.4f} "
                "val_f1_macro={:.4f} val_recall_weak={:.4f}".format(
                    epoch,
                    train_loss,
                    val_loss,
                    val_metrics["f1_macro"],
                    val_metrics["recall_weak"],
                )
            )
        if should_stop:
            break

    early_stopping.restore_best_weights(model)
    best_epoch = int(early_stopping.best_epoch or len(history))
    best_val_f1 = float(early_stopping.best_score or 0.0)
    epochs_ran = len(history)

    checkpoint_dir = (
        PROJECT_ROOT
        / "models"
        / "saved"
        / "deep_classification"
        / dataset
        / scenario
        / model_name
        / variant_name
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = create_config(
        dataset,
        scenario,
        model_name,
        loss_weight,
        args,
        device,
        n_features,
        grade_seq_len,
    )
    config["epochs_ran"] = epochs_ran
    config["best_epoch"] = best_epoch
    config["best_val_f1_macro"] = best_val_f1
    config["feature_selection"] = feature_selection_metadata
    config["grade_feature_names"] = grade_feature_names
    if is_clsv2:
        config["grade_input_dim"] = int(grade_input_dim)
        config["grade_sequence_shape"] = [int(grade_seq_len or 0), int(grade_input_dim)]
    config["imbalance_strategy"] = args.imbalance_strategy
    config["imbalance_effective_strategy"] = effective_strategy
    config["class_distribution_train_before"] = before_distribution
    config["class_distribution_train_after"] = after_distribution
    config["resampling_warning"] = resampling_warning
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
        },
        checkpoint_dir / "best_model.pt",
    )
    (checkpoint_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    history_dir = PROJECT_ROOT / "reports" / "results" / "deep_training_history" / dataset / scenario
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / f"{model_name}_{variant_name}_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    curve_path = (
        PROJECT_ROOT
        / "reports"
        / "figures"
        / "deep_classification"
        / dataset
        / scenario
        / f"{model_name}_{variant_name}_training_curve.png"
    )
    save_training_curve(
        history,
        curve_path,
        f"{dataset} {scenario} {model_name} {loss_weight}",
    )

    rows: list[dict] = []
    for split, loader_key in (("train", "train_eval"), ("val", "val"), ("test", "test")):
        y_true, y_pred, y_prob = predict_model(
            model,
            loaders[loader_key],
            device,
            is_hybrid=is_hybrid,
        )
        metrics = classification_metrics(y_true, y_pred)
        probability_metrics = classification_probability_metrics(y_true, y_prob)
        rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "task": "classification",
                "model_name": model_name,
                "loss_weight": loss_weight,
                "variant_name": variant_name,
                "split": split,
                "n_rows": int(len(y_true)),
                "n_features": int(n_features),
                "epochs_ran": epochs_ran,
                "best_epoch": best_epoch,
                "best_val_f1_macro": best_val_f1,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "dropout": args.dropout,
                "mixup_alpha": float(args.mixup_alpha),
                "seed": args.seed,
                "feature_selection": feature_selection_metadata.get("method", args.feature_selection),
                "imbalance_strategy": args.imbalance_strategy,
                "imbalance_effective_strategy": effective_strategy,
                "grade_feature_dim": int((grade_seq_len or 0) * (grade_input_dim if is_clsv2 else 1)),
                "grade_seq_len": int(grade_seq_len or 0),
                "grade_input_dim": int(grade_input_dim if is_clsv2 else 1),
                **metrics,
                **probability_metrics,
            }
        )

        prediction_path = (
            PROJECT_ROOT
            / "reports"
            / "results"
            / "deep_predictions"
            / dataset
            / scenario
            / model_name
            / variant_name
            / f"{split}_predictions.csv"
        )
        save_predictions(prediction_path, y_true, y_pred, y_prob)

        if split == "test":
            figure_path = (
                PROJECT_ROOT
                / "reports"
                / "figures"
                / "deep_classification"
                / dataset
                / scenario
                / f"{model_name}_{variant_name}_test_confusion_matrix.png"
            )
            save_confusion_matrix_plot(
                y_true,
                y_pred,
                figure_path,
                f"{dataset} {scenario} {model_name} {loss_weight} test confusion matrix",
            )

    val_row = next(row for row in rows if row["split"] == "val")
    test_row = next(row for row in rows if row["split"] == "test")
    print(
        "  best_epoch={} best_val_f1_macro={:.4f} "
        "test_f1_macro={:.4f} test_accuracy={:.4f} test_recall_weak={:.4f}".format(
            best_epoch,
            best_val_f1,
            test_row["f1_macro"],
            test_row["accuracy"],
            test_row["recall_weak"],
        )
    )
    return rows, config


def process_dataset_scenario(
    dataset: str,
    scenario: str,
    models: list[str],
    loss_weights: list[str],
    args: argparse.Namespace,
    device: torch.device,
    result_rows: list[dict],
    skipped_rows: list[dict],
) -> None:
    print(f"\nLoading processed data: dataset={dataset}, scenario={scenario}")
    data = load_processed_classification_data(dataset, scenario, PROJECT_ROOT)
    print(
        "Processed data OK: train={}, val={}, test={}, n_features={}".format(
            data["X_train"].shape[0],
            data["X_val"].shape[0],
            data["X_test"].shape[0],
            data["X_train"].shape[1],
        )
    )

    scenario_models = resolve_models_for_scenario(models, scenario)
    if models == ["auto"]:
        print(f"Auto model for {dataset}/{scenario}: clsv2 (G1/G2 stay in the full vector plus rich grade branch when available)")

    for model_name in scenario_models:
        for loss_weight in loss_weights:
            try:
                rows, _ = train_configuration(
                    dataset=dataset,
                    scenario=scenario,
                    model_name=model_name,
                    loss_weight=loss_weight,
                    data=data,
                    args=args,
                    device=device,
                )
                result_rows.extend(rows)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                print(f"  SKIPPED {dataset}/{scenario}/{model_name}/{loss_weight}: {reason}")
                append_skip(
                    skipped_rows,
                    dataset,
                    scenario,
                    model_name,
                    loss_weight,
                    reason,
                    traceback.format_exc(),
                )


def write_training_report(device: torch.device, skipped_rows: list[dict]) -> None:
    report_path = PROJECT_ROOT / "reports" / "tables" / "deep_training_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Deep Classification Training Report",
        f"Project root: {PROJECT_ROOT}",
        f"Device: {device}",
        "",
    ]
    if skipped_rows:
        lines.append("Skipped records:")
        for row in skipped_rows:
            lines.append(
                "- {dataset}/{scenario}/{model_name}/{loss_weight}: {reason}".format(**row)
            )
    else:
        lines.append("Skipped records: none")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    datasets = selected(args.dataset, DATASETS)
    scenarios = selected(args.scenario, SCENARIOS)
    models = selected_models(args.model)
    loss_weights = selected(args.loss_weight, LOSS_WEIGHTS)
    device = get_device()
    print(f"Using device: {device}")

    result_rows: list[dict] = []
    skipped_rows: list[dict] = []

    for dataset in datasets:
        for scenario in scenarios:
            process_dataset_scenario(
                dataset,
                scenario,
                models,
                loss_weights,
                args,
                device,
                result_rows,
                skipped_rows,
            )

    upsert_csv(
        PROJECT_ROOT / "reports" / "results" / "deep_classification_results.csv",
        result_rows,
        [
            "dataset",
            "scenario",
            "model_name",
            "loss_weight",
            "variant_name",
            "imbalance_strategy",
            "split",
            "seed",
        ],
    )
    upsert_csv(
        PROJECT_ROOT / "reports" / "results" / "deep_skipped_records.csv",
        skipped_rows,
        ["dataset", "scenario", "model_name", "loss_weight"],
    )
    write_training_report(device, skipped_rows)

    print("\nDeep classification training completed.")
    print(f"Result rows written/updated: {len(result_rows)}")
    print(f"Skipped records: {len(skipped_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
