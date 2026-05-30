from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn


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
from src.evaluation.metrics import classification_metrics, save_confusion_matrix_plot  # noqa: E402
from src.train.deep_utils import compute_class_weights, get_device, set_seed  # noqa: E402
from src.train.train_xapi_deep import (  # noqa: E402
    build_model,
    is_hybrid_model,
    make_static_loaders,
    normalize_model_name,
    predict,
    prepare_training_data,
    resample_xapi_raw_train_split,
    resample_training_arrays,
    train_epoch,
    update_swa_batch_norm,
    validate_training_arrays,
)


RESULTS_PATH = PROJECT_ROOT / "reports" / "results" / "xapi_cv_results.csv"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "tables" / "xapi_cv_summary.csv"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "xapi_cv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run xAPI deep model with stratified K-fold CV.")
    parser.add_argument("--folds", "--cv", dest="folds", type=int, default=5)
    parser.add_argument(
        "--model",
        choices=[
            "cnn_bilstm_xapi",
            "cls_xapi",
            "cls-xapi",
            "mlp_static",
            "cnn_bilstm_tabular",
        ],
        default="cnn_bilstm_xapi",
    )
    parser.add_argument(
        "--imbalance-strategy",
        "--oversampling",
        dest="imbalance_strategy",
        choices=["none", "random_over", "smote", "borderline_smote", "adasyn", "smotenc"],
        default="adasyn",
    )
    parser.add_argument("--feature-selection", choices=["none", "pearson_chi2"], default="pearson_chi2")
    parser.add_argument("--max-features", type=int, default=56)
    parser.add_argument("--resampling-fallback", choices=["error", "none", "smote", "random_over"], default="smote")
    parser.add_argument("--loss-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--fusion", choices=["concat", "gated"], default="concat")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n-conv-blocks", type=int, choices=[1, 2], default=1)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--swa", action="store_true")
    return parser.parse_args()


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


def build_fold_data(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    feature_columns = [column for column in df_train.columns if column not in {"Class", "target_class_name", "target_class"}]
    preprocessor, numeric_columns, categorical_columns = build_preprocessor(df_train[feature_columns])
    X_train = to_dense(preprocessor.fit_transform(df_train[feature_columns])).astype("float32")
    X_val = to_dense(preprocessor.transform(df_val[feature_columns])).astype("float32")
    X_test = to_dense(preprocessor.transform(df_test[feature_columns])).astype("float32")
    feature_names = get_feature_names(preprocessor, feature_columns)
    return {
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
            "scenario": "xapi_cv",
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "raw_feature_columns": feature_columns,
            "leakage_checks": {"contains_target_in_features": False, "passed": True},
        },
    }


def train_fold(fold: int, data: dict, args: argparse.Namespace, device: torch.device) -> dict:
    set_seed(args.seed + fold)
    model_name = normalize_model_name(args.model)
    prepared = prepare_training_data(
        data,
        model_name,
        feature_selection=args.feature_selection,
        max_features=args.max_features,
    )
    arrays = prepared["arrays"]
    is_hybrid = is_hybrid_model(model_name)
    validate_training_arrays(arrays, is_hybrid=is_hybrid)
    post_encode_strategy = "none" if args.imbalance_strategy == "smotenc" else args.imbalance_strategy
    arrays, effective_strategy, before_distribution, after_distribution, warning = resample_training_arrays(
        arrays,
        is_hybrid=is_hybrid,
        strategy=post_encode_strategy,
        seed=args.seed + fold,
        fallback=args.resampling_fallback,
    )
    if args.imbalance_strategy == "smotenc":
        raw_resampling = data["metadata"].get("raw_resampling", {})
        effective_strategy = "smotenc"
        before_distribution = raw_resampling.get("before_distribution", before_distribution)
        after_distribution = raw_resampling.get("after_distribution", after_distribution)
        warning = raw_resampling.get("note", "SMOTENC applied before one-hot encoding on training fold.")
    validate_training_arrays(arrays, is_hybrid=is_hybrid)
    set_seed(args.seed + fold)

    loaders = make_static_loaders(arrays, args.batch_size)
    model = build_model(
        model_name,
        int(arrays["X_train"].shape[1]),
        prepared["sequence_len"],
        fusion=args.fusion,
        conv_channels=16,
        n_conv_blocks=args.n_conv_blocks,
        lstm_hidden=16,
        n_bilstm_layers=1,
        dense_hidden=64,
        dropout=0.10,
    ).to(device)
    class_weight = compute_class_weights(arrays["y_train"]).to(device) if args.loss_weight == "balanced" else None
    criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=args.label_smoothing)
    effective_weight_decay = args.weight_decay
    if effective_weight_decay is None:
        effective_weight_decay = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=effective_weight_decay)
    scheduler = None
    if args.scheduler == "cosine" and not args.swa:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
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

    best_state = None
    best_epoch = 0
    best_val_f1 = -1.0
    patience_counter = 0
    print(
        f"Fold {fold}: model={model_name} imbalance={args.imbalance_strategy} effective={effective_strategy} "
        f"input_features={arrays['X_train'].shape[1]} sequence_split=disabled"
    )
    print(f"  class distribution train before/after: {before_distribution} -> {after_distribution}")
    if warning:
        print(f"  warning: {warning}")

    for epoch in range(1, args.epochs + 1):
        train_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            is_hybrid,
            clip_grad=args.clip_grad,
        )
        y_true_val, y_pred_val, _ = predict(model, loaders["val"], device, is_hybrid)
        val_metrics = classification_metrics(y_true_val, y_pred_val)
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            if not args.swa:
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if args.swa and epoch >= int(swa_start_epoch):
            swa_model.update_parameters(model)
            swa_scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or patience_counter >= args.patience:
            suffix = " swa" if args.swa and epoch >= int(swa_start_epoch) else ""
            print(f"  epoch={epoch:03d} val_f1={val_metrics['f1_macro']:.4f}{suffix}")
        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_metrics["f1_macro"])
            else:
                scheduler.step()
        if not args.swa and patience_counter >= args.patience:
            break

    if args.swa:
        update_swa_batch_norm(loaders["train"], swa_model, device, is_hybrid=is_hybrid)
        model_for_eval = swa_model
        y_true_val, y_pred_val, _ = predict(model_for_eval, loaders["val"], device, is_hybrid)
        best_val_f1 = classification_metrics(y_true_val, y_pred_val)["f1_macro"]
        best_epoch = args.epochs
    else:
        model_for_eval = model
    if not args.swa and best_state is not None:
        model.load_state_dict(copy.deepcopy(best_state))
        model_for_eval = model
    y_true, y_pred, _ = predict(model_for_eval, loaders["test"], device, is_hybrid)
    metrics = classification_metrics(y_true, y_pred)
    save_confusion_matrix_plot(
        y_true,
        y_pred,
            FIGURE_DIR / model_name / f"fold_{fold}_confusion_matrix.png",
        f"xAPI CV fold {fold} {model_name}",
    )
    return {
        "dataset": "xapi",
        "scenario": "xapi_cv",
        "fold": fold,
        "model_name": model_name,
        "loss_weight": args.loss_weight,
        "imbalance_strategy": args.imbalance_strategy,
        "imbalance_effective_strategy": effective_strategy,
        "feature_selection": args.feature_selection,
        "fusion": "single_input",
        "label_smoothing": args.label_smoothing,
        "scheduler": args.scheduler,
        "swa": bool(args.swa),
        "weight_decay": effective_weight_decay,
        "n_conv_blocks": args.n_conv_blocks,
        "clip_grad": args.clip_grad,
        "best_epoch": best_epoch,
        "best_val_f1_macro": best_val_f1,
        "n_input_features": int(arrays["X_train"].shape[1]),
        "n_static_features": 0,
        "n_sequence_features": 0,
        **metrics,
    }


def summarize(rows: list[dict]) -> pd.DataFrame:
    data = pd.DataFrame(rows)
    metric_columns = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "balanced_accuracy"]
    summary = {
        "dataset": "xapi",
        "scenario": "xapi_cv",
        "model_name": data.iloc[0]["model_name"],
        "folds": int(data["fold"].nunique()),
    }
    for column in metric_columns:
        summary[f"{column}_mean"] = float(data[column].mean())
        summary[f"{column}_std"] = float(data[column].std(ddof=1) if len(data) > 1 else 0.0)
    return pd.DataFrame([summary])


def main() -> int:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2 for cross-validation.")
    set_seed(args.seed)
    device = get_device()
    raw = add_targets(read_xapi(RAW_PATH))
    y = raw["target_class"].to_numpy()
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    rows: list[dict] = []

    for fold, (train_val_idx, test_idx) in enumerate(splitter.split(raw, y), start=1):
        train_val = raw.iloc[train_val_idx].reset_index(drop=True)
        test = raw.iloc[test_idx].reset_index(drop=True)
        train, val = train_test_split(
            train_val,
            test_size=0.15,
            random_state=args.seed + fold,
            stratify=train_val["target_class"],
        )
        train_fold_raw = train.reset_index(drop=True)
        raw_resampling = {}
        if args.imbalance_strategy == "smotenc":
            train_fold_raw, before, after = resample_xapi_raw_train_split(
                train_fold_raw,
                seed=args.seed + fold,
            )
            raw_resampling = {
                "strategy": "smotenc",
                "before_distribution": before,
                "after_distribution": after,
                "note": "SMOTENC was fit only on the raw training fold before one-hot encoding.",
            }
        data = build_fold_data(train_fold_raw, val.reset_index(drop=True), test)
        data["metadata"]["raw_resampling"] = raw_resampling
        rows.append(train_fold(fold, data, args, device))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)
    summary = summarize(rows)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"Saved fold results: {RESULTS_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
