from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.paper_replication.pipeline import PROJECT_ROOT, build_preprocessor, dense_array
from src.paper_replication.v18_strict_validation import features_for_case
from src.paper_replication.v18_strict_validation import prepare_strict_split
from src.paper_replication.v6_case_sweep import (
    ALL_DATASETS,
    RESULTS_DIR,
    REPORTS_DIR,
    SweepCase,
    fmt,
    parse_csv_strings,
    read_dataset,
)
from src.paper_replication.advanced_experiments import ExactPaperCNNBiLSTM, model_dataset_type, set_seed


BASELINE_CV_RESULTS_PATH = RESULTS_DIR / "v19_baseline_cv_results.json"
BASELINE_CV_REPORT_PATH = REPORTS_DIR / "v19_baseline_cv_report.md"
PERMUTATION_RESULTS_PATH = RESULTS_DIR / "v19_permutation_importance_results.json"
PERMUTATION_REPORT_PATH = REPORTS_DIR / "v19_permutation_importance_report.md"
ABLATION_RESULTS_PATH = RESULTS_DIR / "v19_ablation_results.json"
ABLATION_REPORT_PATH = REPORTS_DIR / "v19_ablation_report.md"


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                values.append(fmt(value))
            else:
                values.append(str(value) if value is not None else "-")
        lines.append("| " + " | ".join(values) + " |")
    return lines


def paper_case(dataset: str) -> SweepCase:
    return SweepCase(
        name=f"v19_{dataset}_paper_feature_cv",
        feature_set="paper",
        oversampling="none",
        conv_filters=64,
        kernel_size=3,
        bilstm_hidden=64,
        dense_hidden=64,
        dropout=0.2,
        lr=1e-3,
        batch_size=32,
        datasets=(dataset,),
    )


def estimators(seed: int) -> dict[str, Any]:
    return {
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
    }


def run_dataset_cv(dataset: str, *, folds: int, seed: int) -> list[dict[str, Any]]:
    raw = read_dataset(dataset)
    frame, y, feature_columns, _ = features_for_case(raw, dataset, paper_case(dataset))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows: list[dict[str, Any]] = []
    for model_name, estimator in estimators(seed).items():
        fold_rows = []
        for fold, (train_idx, test_idx) in enumerate(splitter.split(frame, y), start=1):
            preprocessor, _, _ = build_preprocessor(frame.iloc[train_idx])
            X_train = dense_array(preprocessor.fit_transform(frame.iloc[train_idx]))
            X_test = dense_array(preprocessor.transform(frame.iloc[test_idx]))
            model = estimator.__class__(**estimator.get_params())
            model.fit(X_train, y[train_idx])
            pred = model.predict(X_test)
            fold_rows.append(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "fold": int(fold),
                    "accuracy": float(accuracy_score(y[test_idx], pred)),
                    "f1_macro": float(f1_score(y[test_idx], pred, average="macro", zero_division=0)),
                    "train_size": int(len(train_idx)),
                    "test_size": int(len(test_idx)),
                    "feature_columns": feature_columns,
                }
            )
        acc = np.asarray([row["accuracy"] for row in fold_rows], dtype=np.float64)
        f1 = np.asarray([row["f1_macro"] for row in fold_rows], dtype=np.float64)
        rows.append(
            {
                "dataset": dataset,
                "model": model_name,
                "folds": int(folds),
                "accuracy_mean": float(acc.mean()),
                "accuracy_std": float(acc.std(ddof=0)),
                "f1_macro_mean": float(f1.mean()),
                "f1_macro_std": float(f1.std(ddof=0)),
                "feature_set": "paper",
                "feature_columns": feature_columns,
                "fold_rows": fold_rows,
            }
        )
    return rows


def run_baseline_cv(args: argparse.Namespace) -> dict[str, Any]:
    datasets = ALL_DATASETS if args.datasets == "all" else parse_csv_strings(args.datasets)
    summary: list[dict[str, Any]] = []
    for dataset in datasets:
        summary.extend(run_dataset_cv(dataset, folds=int(args.folds), seed=int(args.seed)))
    payload = {
        "config": {
            "protocol": "stratified k-fold cross-validation",
            "folds": int(args.folds),
            "seed": int(args.seed),
            "datasets": datasets,
            "feature_set": "paper",
            "note": "Preprocessor is fit inside each fold to avoid leakage.",
        },
        "summary": summary,
        "artifacts": {
            "results": str(BASELINE_CV_RESULTS_PATH),
            "report": str(BASELINE_CV_REPORT_PATH),
        },
    }
    save_json(BASELINE_CV_RESULTS_PATH, payload)
    write_baseline_cv_report(payload)
    return payload


def write_baseline_cv_report(payload: dict[str, Any]) -> None:
    compact = [
        {key: row[key] for key in ("dataset", "model", "folds", "accuracy_mean", "accuracy_std", "f1_macro_mean", "f1_macro_std")}
        for row in payload["summary"]
    ]
    lines = [
        "# V19 Baseline Cross-Validation Report",
        "",
        "Protocol: stratified 5-fold cross-validation. The preprocessor is fit inside each fold, then the model is evaluated on that fold's holdout split.",
        "",
        *markdown_table(compact, ["dataset", "model", "folds", "accuracy_mean", "accuracy_std", "f1_macro_mean", "f1_macro_std"]),
        "",
        "## Notes",
        "",
        "- This satisfies the baseline CV requirement for Decision Tree and Random Forest.",
        "- These are baseline ML results, not CNN-BiLSTM results.",
        "- Feature set is the paper feature set to keep the comparison simple and reproducible.",
    ]
    BASELINE_CV_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_CV_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_permutation_importance(args: argparse.Namespace) -> dict[str, Any]:
    datasets = ALL_DATASETS if args.datasets == "all" else parse_csv_strings(args.datasets)
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        raw = read_dataset(dataset)
        case = paper_case(dataset)
        split = prepare_strict_split(dataset, raw, case, int(args.seed))
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=int(args.seed),
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        model.fit(split.X_train, split.y_train)
        pred = model.predict(split.X_test)
        baseline = {
            "accuracy": float(accuracy_score(split.y_test, pred)),
            "f1_macro": float(f1_score(split.y_test, pred, average="macro", zero_division=0)),
        }
        result = permutation_importance(
            model,
            split.X_test,
            split.y_test,
            scoring="f1_macro",
            n_repeats=int(args.repeats),
            random_state=int(args.seed),
            n_jobs=-1,
        )
        feature_rows = []
        for name, mean, std in zip(split.processed_feature_names, result.importances_mean, result.importances_std):
            feature_rows.append({"feature": name, "importance_mean": float(mean), "importance_std": float(std)})
        feature_rows = sorted(feature_rows, key=lambda item: item["importance_mean"], reverse=True)
        rows.append(
            {
                "dataset": dataset,
                "model": "RandomForest",
                "seed": int(args.seed),
                "repeats": int(args.repeats),
                "baseline": baseline,
                "top_features": feature_rows[: int(args.top_k)],
                "all_features": feature_rows,
            }
        )
    payload = {
        "config": {
            "protocol": "strict split permutation importance",
            "seed": int(args.seed),
            "repeats": int(args.repeats),
            "top_k": int(args.top_k),
            "feature_set": "paper",
            "note": "RandomForest proxy explainability on strict test split.",
        },
        "rows": rows,
        "artifacts": {
            "results": str(PERMUTATION_RESULTS_PATH),
            "report": str(PERMUTATION_REPORT_PATH),
        },
    }
    save_json(PERMUTATION_RESULTS_PATH, payload)
    write_permutation_report(payload)
    return payload


def write_permutation_report(payload: dict[str, Any]) -> None:
    lines = [
        "# V19 Permutation Importance Report",
        "",
        "Protocol: RandomForest proxy model on the strict split. Importance is measured by the drop in Macro-F1 after permuting each processed feature.",
        "",
    ]
    for row in payload["rows"]:
        lines.extend(
            [
                f"## {row['dataset']}",
                "",
                f"Baseline strict test Accuracy: {row['baseline']['accuracy']:.4f}; Macro-F1: {row['baseline']['f1_macro']:.4f}.",
                "",
                *markdown_table(row["top_features"], ["feature", "importance_mean", "importance_std"]),
                "",
            ]
        )
    lines.extend(
        [
            "## Notes",
            "",
            "- This is not SHAP and not a direct neural-network explanation; it is a reproducible permutation-importance proxy.",
            "- It supports recommendation design by identifying which paper features most affect a baseline classifier.",
        ]
    )
    PERMUTATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PERMUTATION_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


class CNNOnly(nn.Module):
    def __init__(self, n_features: int, n_classes: int, conv_filters: int = 64, kernel_size: int = 3, dense_hidden: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for index in range(4):
            layers.append(nn.Conv1d(1 if index == 0 else conv_filters, conv_filters, kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU())
        self.cnn = nn.Sequential(*layers, nn.AdaptiveMaxPool1d(4))
        self.classifier = nn.Sequential(
            nn.Linear(conv_filters * 4, dense_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cnn(x.unsqueeze(1)).flatten(1)
        return self.classifier(out)


class BiLSTMOnly(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden: int = 64, dense_hidden: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.bilstm = nn.LSTM(1, hidden, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, dense_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.bilstm(x.unsqueeze(-1))
        return self.classifier(out[:, -1, :])


def predict_model(model: nn.Module, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (batch_x,) in loader:
            logits = model(batch_x.to(device))
            preds.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def train_ablation_model(dataset: str, split: Any, model_name: str, seed: int, epochs: int) -> dict[str, Any]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "CNN-only":
        model = CNNOnly(split.n_features, split.n_classes).to(device)
    elif model_name == "BiLSTM-only":
        model = BiLSTMOnly(split.n_features, split.n_classes).to(device)
    elif model_name == "CNN+BiLSTM":
        model = ExactPaperCNNBiLSTM(model_dataset_type(dataset), split.n_features, split.n_classes).to(device)
    else:
        raise ValueError(f"Unsupported ablation model: {model_name}")
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.tensor(split.X_train, dtype=torch.float32), torch.tensor(split.y_train, dtype=torch.long)),
        batch_size=32,
        shuffle=True,
        generator=generator,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_state: dict[str, torch.Tensor] | None = None
    best_val_f1 = -1.0
    best_val_acc = -1.0
    best_epoch = 0
    for epoch in range(1, int(epochs) + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        val_pred = predict_model(model, split.X_val, 32, device).argmax(axis=1)
        val_acc = float(accuracy_score(split.y_val, val_pred))
        val_f1 = float(f1_score(split.y_val, val_pred, average="macro", zero_division=0))
        if val_f1 > best_val_f1 or (abs(val_f1 - best_val_f1) <= 1e-12 and val_acc > best_val_acc):
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    test_pred = predict_model(model, split.X_test, 32, device).argmax(axis=1)
    return {
        "dataset": dataset,
        "model": model_name,
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "val_accuracy": best_val_acc,
        "val_f1_macro": best_val_f1,
        "test_accuracy": float(accuracy_score(split.y_test, test_pred)),
        "test_f1_macro": float(f1_score(split.y_test, test_pred, average="macro", zero_division=0)),
    }


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    datasets = ALL_DATASETS if args.datasets == "all" else parse_csv_strings(args.datasets)
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        raw = read_dataset(dataset)
        split = prepare_strict_split(dataset, raw, paper_case(dataset), int(args.seed))
        for model_name in ("CNN-only", "BiLSTM-only", "CNN+BiLSTM"):
            rows.append(train_ablation_model(dataset, split, model_name, int(args.seed), int(args.epochs)))
    payload = {
        "config": {
            "protocol": "strict validation ablation",
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "feature_set": "paper",
            "models": ["CNN-only", "BiLSTM-only", "CNN+BiLSTM"],
        },
        "rows": rows,
        "artifacts": {
            "results": str(ABLATION_RESULTS_PATH),
            "report": str(ABLATION_REPORT_PATH),
        },
    }
    save_json(ABLATION_RESULTS_PATH, payload)
    write_ablation_report(payload)
    return payload


def write_ablation_report(payload: dict[str, Any]) -> None:
    lines = [
        "# V19 Ablation Study Report",
        "",
        "Protocol: strict train/validation/test split with paper features. Validation selects best epoch; test is evaluated after epoch selection.",
        "",
        *markdown_table(payload["rows"], ["dataset", "model", "seed", "best_epoch", "val_f1_macro", "test_accuracy", "test_f1_macro"]),
        "",
        "## Notes",
        "",
        "- This ablation tests whether CNN+BiLSTM is consistently better than CNN-only or BiLSTM-only under the same strict split.",
        "- Results must be read honestly; if CNN+BiLSTM does not win on a dataset, the report should say so.",
    ]
    ABLATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ABLATION_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V19 support experiments: baseline cross-validation.")
    sub = parser.add_subparsers(dest="command", required=True)
    cv = sub.add_parser("baseline-cv")
    cv.add_argument("--datasets", default="all")
    cv.add_argument("--folds", type=int, default=5)
    cv.add_argument("--seed", type=int, default=42)
    perm = sub.add_parser("permutation-importance")
    perm.add_argument("--datasets", default="all")
    perm.add_argument("--seed", type=int, default=42)
    perm.add_argument("--repeats", type=int, default=10)
    perm.add_argument("--top-k", type=int, default=10)
    ablation = sub.add_parser("ablation")
    ablation.add_argument("--datasets", default="all")
    ablation.add_argument("--seed", type=int, default=42)
    ablation.add_argument("--epochs", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "baseline-cv":
        run_baseline_cv(args)
    elif args.command == "permutation-importance":
        run_permutation_importance(args)
    elif args.command == "ablation":
        run_ablation(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
