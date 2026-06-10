from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.paper_replication.advanced_experiments import PAPER_BENCHMARKS, apply_binning, set_seed
from src.paper_replication.pipeline import PROJECT_ROOT, build_preprocessor, dense_array
from src.paper_replication.v6_case_sweep import (
    ALL_DATASETS,
    BINNING_KEY,
    CASES,
    CLASS_NAMES,
    DEFAULT_OVERSAMPLING,
    MODELS_DIR,
    REPORTS_DIR,
    RESULTS_DIR,
    STUDENT_DATASETS,
    SweepCase,
    criterion_for,
    effective_oversampling,
    fmt,
    get_feature_names,
    make_loader,
    make_model,
    metric_dict,
    oversample_train,
    parse_csv_ints,
    parse_csv_strings,
    predict_proba,
    read_dataset,
    save_json,
    student_features,
    xapi_features,
)


STRICT_RESULTS_PATH = RESULTS_DIR / "v18_strict_validation_results.json"
STRICT_REPORT_PATH = REPORTS_DIR / "v18_strict_validation_report.md"
STRICT_MODEL_DIR = MODELS_DIR / "strict_validation"
STRICT_OPTUNA_TRIALS_PATH = RESULTS_DIR / "v18_strict_optuna_trials.csv"
STRICT_OPTUNA_BEST_PATH = RESULTS_DIR / "v18_strict_optuna_best_params.json"

DEFAULT_STRICT_SEEDS = [42, 123, 314]


@dataclass
class StrictSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    feature_columns: list[str]
    processed_feature_names: list[str]
    n_features: int
    n_classes: int


@dataclass
class StrictTrainResult:
    row: dict[str, Any]
    state_dict: dict[str, torch.Tensor]
    split: StrictSplit


def ensure_dirs() -> None:
    for path in (RESULTS_DIR, REPORTS_DIR, MODELS_DIR, STRICT_MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)


def deep_engineered_student_features(raw: pd.DataFrame, dataset: str, binning_key: str = BINNING_KEY) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    g1 = pd.to_numeric(raw["G1"], errors="raise").astype(float)
    g2 = pd.to_numeric(raw["G2"], errors="raise").astype(float)
    ratio = np.divide(g2, np.maximum(g1, 1.0))
    delta = g2 - g1
    trend = np.where(delta > 1.0, "up", np.where(delta < -1.0, "down", "stable"))
    data = pd.DataFrame(
        {
            "G1": g1,
            "G2": g2,
            "G2_minus_G1": delta,
            "G2_div_G1_safe": ratio,
            "G_mean": (g1 + g2) / 2.0,
            "G_min": np.minimum(g1, g2),
            "G_max": np.maximum(g1, g2),
            "G1_score_percent": g1 / 20.0,
            "G2_score_percent": g2 / 20.0,
            "G1_bin": apply_binning(g1, binning_key, dataset).astype(float),
            "G2_bin": apply_binning(g2, binning_key, dataset).astype(float),
            "G_trend_label": trend,
        },
        index=raw.index,
    )
    y = apply_binning(raw["G3"], binning_key, dataset).to_numpy(dtype=np.int64)
    return data, y, data.columns.tolist()


def features_for_case(raw: pd.DataFrame, dataset: str, case: SweepCase) -> tuple[pd.DataFrame, np.ndarray, list[str], int]:
    if dataset in STUDENT_DATASETS:
        if case.feature_set == "deep_engineered":
            frame, y, feature_columns = deep_engineered_student_features(raw, dataset, case.binning_key)
        else:
            frame, y, feature_columns = student_features(raw, dataset, case.feature_set, case.binning_key)
        return frame, y, feature_columns, 5
    frame, y, feature_columns = xapi_features(raw, case.feature_set)
    return frame, y, feature_columns, 3


def prepare_strict_split(dataset: str, raw: pd.DataFrame, case: SweepCase, seed: int) -> StrictSplit:
    frame, y, feature_columns, n_classes = features_for_case(raw, dataset, case)
    indices = np.arange(len(frame))
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices,
        y,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )
    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx,
        y_temp,
        test_size=0.50,
        random_state=seed,
        stratify=y_temp,
    )
    preprocessor, _, _ = build_preprocessor(frame.iloc[train_idx])
    X_train = dense_array(preprocessor.fit_transform(frame.iloc[train_idx]))
    X_val = dense_array(preprocessor.transform(frame.iloc[val_idx]))
    X_test = dense_array(preprocessor.transform(frame.iloc[test_idx]))
    return StrictSplit(
        X_train=X_train,
        y_train=np.asarray(y_train, dtype=np.int64),
        X_val=X_val,
        y_val=np.asarray(y_val, dtype=np.int64),
        X_test=X_test,
        y_test=np.asarray(y_test, dtype=np.int64),
        train_indices=np.asarray(train_idx, dtype=np.int64),
        val_indices=np.asarray(val_idx, dtype=np.int64),
        test_indices=np.asarray(test_idx, dtype=np.int64),
        feature_columns=feature_columns,
        processed_feature_names=get_feature_names(preprocessor, feature_columns),
        n_features=int(X_train.shape[1]),
        n_classes=int(n_classes),
    )


def train_strict_case_seed(dataset: str, raw: pd.DataFrame, case: SweepCase, seed: int, epochs: int) -> StrictTrainResult:
    set_seed(seed)
    random.seed(seed)
    split = prepare_strict_split(dataset, raw, case, seed)
    sampling = effective_oversampling(dataset, case)
    if sampling == "none":
        X_train = np.asarray(split.X_train, dtype=np.float32)
        y_train = np.asarray(split.y_train, dtype=np.int64)
        effective = "none"
    else:
        X_train, y_train, effective = oversample_train(split.X_train, split.y_train, sampling, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(dataset, split, case).to(device)
    loader = make_loader(X_train, y_train, case.batch_size, seed)
    criterion = criterion_for(case, split, device)
    if case.weight_decay > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(case.lr), weight_decay=float(case.weight_decay))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(case.lr))

    best_state: dict[str, torch.Tensor] | None = None
    best_val_metrics: dict[str, float] | None = None
    best_epoch = 0
    wait = 0
    history: list[dict[str, float]] = []
    started = time.time()
    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses: list[float] = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_probs = predict_proba(model, split.X_val, case.batch_size, device)
        val_pred = val_probs.argmax(axis=1)
        val_metrics = metric_dict(split.y_val, val_pred)
        history.append({"epoch": int(epoch), "train_loss": float(np.mean(losses)) if losses else 0.0, **val_metrics})
        is_better = best_val_metrics is None or val_metrics["f1_macro"] > best_val_metrics["f1_macro"] + 1e-12
        if best_val_metrics is not None and abs(val_metrics["f1_macro"] - best_val_metrics["f1_macro"]) <= 1e-12:
            is_better = val_metrics["accuracy"] > best_val_metrics["accuracy"] + 1e-12
        if is_better:
            best_val_metrics = val_metrics
            best_epoch = int(epoch)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if case.patience > 0 and wait >= int(case.patience):
                break
    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_val_metrics is None:
        best_val_metrics = {"accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0}
    row = {
        "dataset": dataset,
        "case": case.name,
        "seed": int(seed),
        "protocol": "strict validation",
        "selection_split": "validation",
        "test_usage": "not_used_for_epoch_or_case_selection",
        "best_epoch": int(best_epoch),
        "elapsed_seconds": float(time.time() - started),
        "model_variant": case.model_variant,
        "num_conv_layers": int(case.num_conv_layers),
        "num_bilstm_layers": int(case.num_bilstm_layers),
        "use_batchnorm": bool(case.use_batchnorm),
        "use_residual": bool(case.use_residual),
        "use_attention": bool(case.use_attention),
        "weight_decay": float(case.weight_decay),
        "patience": int(case.patience),
        "feature_set": case.feature_set,
        "binning": case.binning_key,
        "split_policy": "strict_train_val_test_70_15_15",
        "selection_metric": "val_f1_macro",
        "feature_columns": split.feature_columns,
        "processed_feature_names": split.processed_feature_names,
        "oversampling_requested": case.oversampling,
        "oversampling_effective": effective,
        "loss": case.loss,
        "label_smoothing": float(case.label_smoothing),
        "train_size": int(len(split.y_train)),
        "val_size": int(len(split.y_val)),
        "test_size": int(len(split.y_test)),
        "n_features": int(split.n_features),
        "n_classes": int(split.n_classes),
        "history": history,
        "val_accuracy": float(best_val_metrics["accuracy"]),
        "val_precision_macro": float(best_val_metrics["precision_macro"]),
        "val_recall_macro": float(best_val_metrics["recall_macro"]),
        "val_f1_macro": float(best_val_metrics["f1_macro"]),
    }
    return StrictTrainResult(row=row, state_dict=best_state, split=split)


def evaluate_selected_on_test(dataset: str, case: SweepCase, result: StrictTrainResult) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(dataset, result.split, case).to(device)
    model.load_state_dict(result.state_dict)
    probabilities = predict_proba(model, result.split.X_test, case.batch_size, device)
    predictions = probabilities.argmax(axis=1)
    metrics = metric_dict(result.split.y_test, predictions)
    labels = list(range(result.split.n_classes))
    report = classification_report(
        result.split.y_test,
        predictions,
        labels=labels,
        target_names=CLASS_NAMES[dataset],
        output_dict=True,
        zero_division=0,
    )
    return {
        **result.row,
        "test_evaluated": True,
        "test_accuracy": metrics["accuracy"],
        "test_precision_macro": metrics["precision_macro"],
        "test_recall_macro": metrics["recall_macro"],
        "test_f1_macro": metrics["f1_macro"],
        "test_confusion_matrix": confusion_matrix(result.split.y_test, predictions, labels=labels).tolist(),
        "test_classification_report": report,
    }


def strict_student_cases() -> list[SweepCase]:
    return [
        replace(CASES[0], name="strict_paper_exact", datasets=("student-mat", "student-por"), oversampling="none", patience=15),
        SweepCase(
            "strict_deep_engineered_exact_classw_smooth",
            "deep_engineered",
            "none",
            64,
            5,
            128,
            128,
            0.25,
            5e-4,
            32,
            0.05,
            "class_weight",
            ("student-mat", "student-por"),
            selection_metric="f1_macro",
            weight_decay=1e-4,
            patience=20,
        ),
        SweepCase(
            "strict_deep_engineered_improved_classw_smooth",
            "deep_engineered",
            "none",
            96,
            5,
            128,
            128,
            0.30,
            5e-4,
            32,
            0.05,
            "class_weight",
            ("student-mat", "student-por"),
            selection_metric="f1_macro",
            model_variant="improved",
            use_batchnorm=True,
            use_residual=True,
            use_attention=True,
            weight_decay=1e-4,
            patience=20,
        ),
    ]


def strict_xapi_cases() -> list[SweepCase]:
    return [
        SweepCase("strict_xapi_paper_exact_adasyn", "paper", "adasyn", 64, 3, 64, 64, 0.20, 1e-3, 32, datasets=("xapi",), patience=15),
        SweepCase(
            "strict_xapi_full_exact_classw_smooth",
            "full",
            "none",
            128,
            5,
            128,
            128,
            0.25,
            3e-4,
            16,
            0.05,
            "class_weight",
            ("xapi",),
            weight_decay=1e-4,
            patience=20,
        ),
        SweepCase(
            "strict_xapi_behavior8_improved_classw",
            "behavior8",
            "none",
            96,
            5,
            128,
            128,
            0.30,
            5e-4,
            32,
            0.05,
            "class_weight",
            ("xapi",),
            model_variant="improved",
            use_batchnorm=True,
            use_residual=True,
            weight_decay=1e-4,
            patience=20,
        ),
    ]


def default_strict_cases(datasets: list[str]) -> list[SweepCase]:
    cases: list[SweepCase] = []
    if any(dataset in STUDENT_DATASETS for dataset in datasets):
        cases.extend(strict_student_cases())
    if "xapi" in datasets:
        cases.extend(strict_xapi_cases())
    return [case for case in cases if any(dataset in case.datasets for dataset in datasets)]


def select_case(results: list[StrictTrainResult]) -> StrictTrainResult:
    return max(results, key=lambda item: (item.row["val_f1_macro"], item.row["val_accuracy"]))


def save_strict_model(dataset: str, selected: dict[str, Any], state_dict: dict[str, torch.Tensor], case: SweepCase) -> str:
    safe_case = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in selected["case"]).strip("_")
    path = STRICT_MODEL_DIR / f"v18_strict_{dataset}_{safe_case}_seed{int(selected['seed'])}.pt"
    torch.save(
        {
            "state_dict": state_dict,
            "dataset": dataset,
            "case": selected["case"],
            "seed": int(selected["seed"]),
            "protocol": "strict validation",
            "selection_metric": "val_f1_macro",
            "metrics": {
                "val_f1_macro": selected["val_f1_macro"],
                "test_f1_macro": selected["test_f1_macro"],
                "test_accuracy": selected["test_accuracy"],
            },
            "feature_columns": selected["feature_columns"],
            "processed_feature_names": selected["processed_feature_names"],
            "case_params": asdict(case),
        },
        path,
    )
    return str(path)


def summarize_selected(selected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in selected_rows}):
        rows = [row for row in selected_rows if row["dataset"] == dataset]
        test_f1 = np.asarray([row["test_f1_macro"] for row in rows], dtype=np.float64)
        test_acc = np.asarray([row["test_accuracy"] for row in rows], dtype=np.float64)
        val_f1 = np.asarray([row["val_f1_macro"] for row in rows], dtype=np.float64)
        paper = PAPER_BENCHMARKS.get(dataset, {})
        summaries.append(
            {
                "dataset": dataset,
                "n_seeds": int(len(rows)),
                "selected_cases": sorted({row["case"] for row in rows}),
                "val_f1_macro_mean": float(val_f1.mean()),
                "val_f1_macro_std": float(val_f1.std(ddof=0)),
                "test_accuracy_mean": float(test_acc.mean()),
                "test_accuracy_std": float(test_acc.std(ddof=0)),
                "test_f1_macro_mean": float(test_f1.mean()),
                "test_f1_macro_std": float(test_f1.std(ddof=0)),
                "paper_f1_macro": paper.get("f1_macro"),
                "gap_to_paper_f1": float(test_f1.mean() - paper["f1_macro"]) if paper.get("f1_macro") is not None else None,
            }
        )
    return summaries


def load_optimistic_manifest() -> dict[str, Any]:
    path = PROJECT_ROOT / "models" / "final" / "final_model_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def optimistic_comparison_rows(strict_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest = load_optimistic_manifest()
    selected = manifest.get("selected_models", {})
    rows: list[dict[str, Any]] = []
    for summary in strict_summary:
        dataset = summary["dataset"]
        optimistic = selected.get(dataset, {}).get("metrics", {})
        rows.append(
            {
                "dataset": dataset,
                "paper_like_optimistic_f1": optimistic.get("f1_macro"),
                "strict_test_f1_mean": summary["test_f1_macro_mean"],
                "strict_test_f1_std": summary["test_f1_macro_std"],
                "paper_f1": summary.get("paper_f1_macro"),
                "strict_gap_to_paper": summary.get("gap_to_paper_f1"),
            }
        )
    return rows


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                values.append(fmt(value))
            elif isinstance(value, list):
                values.append(", ".join(str(item) for item in value))
            else:
                values.append(str(value) if value is not None else "-")
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_strict_report(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# V18 Strict Validation Report",
        "",
        "Protocol: train/validation/test = 70/15/15. Validation selects epoch and case. Test is evaluated only after validation selection for each dataset/seed.",
        "",
        "## Paper-like Optimistic vs Strict",
        "",
        *markdown_table(
            payload["optimistic_vs_strict"],
            ["dataset", "paper_like_optimistic_f1", "strict_test_f1_mean", "strict_test_f1_std", "paper_f1", "strict_gap_to_paper"],
        ),
        "",
        "## Strict Test Summary",
        "",
        *markdown_table(
            payload["strict_summary"],
            [
                "dataset",
                "n_seeds",
                "selected_cases",
                "val_f1_macro_mean",
                "val_f1_macro_std",
                "test_accuracy_mean",
                "test_accuracy_std",
                "test_f1_macro_mean",
                "test_f1_macro_std",
                "gap_to_paper_f1",
            ],
        ),
        "",
        "## Selected Rows",
        "",
        *markdown_table(
            payload["selected_by_validation"],
            ["dataset", "seed", "case", "best_epoch", "val_f1_macro", "test_accuracy", "test_f1_macro", "model_path"],
        ),
        "",
        "## Notes",
        "",
        "- Rows in `validation_rows` do not contain test metrics unless they are selected by validation.",
        "- This report intentionally separates strict validation metrics from the previous paper-like optimistic manifest.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_strict_sweep(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dirs()
    datasets = parse_csv_strings(args.datasets)
    seeds = parse_csv_ints(args.seeds)
    cases = default_strict_cases(datasets)
    if args.case_names:
        wanted = set(parse_csv_strings(args.case_names))
        cases = [case for case in cases if case.name in wanted]
        missing = wanted - {case.name for case in cases}
        if missing:
            raise ValueError(f"Unknown strict cases: {sorted(missing)}")
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    raws = {dataset: read_dataset(dataset) for dataset in datasets}
    validation_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    model_paths: dict[str, str] = {}
    print(f"=== V18 strict validation sweep datasets={datasets} seeds={seeds} cases={len(cases)} epochs={args.epochs}", flush=True)
    for dataset in datasets:
        dataset_cases = [case for case in cases if dataset in case.datasets]
        for seed in seeds:
            seed_results: list[StrictTrainResult] = []
            for case in dataset_cases:
                result = train_strict_case_seed(dataset, raws[dataset], case, int(seed), int(args.epochs))
                validation_rows.append(result.row)
                seed_results.append(result)
                print(
                    f"{dataset:12s} {case.name:42s} seed={seed:<4d} "
                    f"val_f1={result.row['val_f1_macro']:.4f} val_acc={result.row['val_accuracy']:.4f} epoch={result.row['best_epoch']}",
                    flush=True,
                )
            selected_result = select_case(seed_results)
            selected_case = next(case for case in dataset_cases if case.name == selected_result.row["case"])
            selected = evaluate_selected_on_test(dataset, selected_case, selected_result)
            model_path = save_strict_model(dataset, selected, selected_result.state_dict, selected_case)
            selected["model_path"] = model_path
            selected_rows.append(selected)
            model_paths[f"{dataset}_seed{seed}"] = model_path
            print(
                f"SELECT {dataset:12s} seed={seed:<4d} case={selected['case']} "
                f"val_f1={selected['val_f1_macro']:.4f} test_f1={selected['test_f1_macro']:.4f}",
                flush=True,
            )
    strict_summary = summarize_selected(selected_rows)
    payload = {
        "config": {
            "protocol": "strict validation",
            "split": "70/15/15 stratified train/validation/test",
            "datasets": datasets,
            "seeds": seeds,
            "epochs": int(args.epochs),
            "case_names": [case.name for case in cases],
            "test_policy": "test evaluated only after validation selects case for each dataset/seed",
        },
        "benchmarks": PAPER_BENCHMARKS,
        "strict_summary": strict_summary,
        "optimistic_vs_strict": optimistic_comparison_rows(strict_summary),
        "validation_rows": validation_rows,
        "selected_by_validation": selected_rows,
        "artifacts": {
            "results": str(STRICT_RESULTS_PATH),
            "report": str(STRICT_REPORT_PATH),
            "models": model_paths,
        },
    }
    save_json(STRICT_RESULTS_PATH, payload)
    write_strict_report(payload, STRICT_REPORT_PATH)
    return payload


def make_trial_case(dataset: str, trial: Any) -> SweepCase:
    if dataset in STUDENT_DATASETS:
        feature_set = trial.suggest_categorical("feature_set", ["paper", "paper_engineered", "deep_engineered"])
        model_variant = trial.suggest_categorical("model_variant", ["exact", "improved"])
        loss = trial.suggest_categorical("loss", ["ce", "class_weight"])
        oversampling = trial.suggest_categorical("oversampling", ["none", "smote"])
    else:
        feature_set = trial.suggest_categorical("feature_set", ["paper", "top5", "behavior8", "full"])
        model_variant = trial.suggest_categorical("model_variant", ["exact", "improved"])
        loss = trial.suggest_categorical("loss", ["ce", "class_weight"])
        oversampling = trial.suggest_categorical("oversampling", ["none", "adasyn", "smote"])
    if loss == "class_weight":
        oversampling = "none"
    return SweepCase(
        name=f"strict_optuna_trial_{trial.number}",
        feature_set=feature_set,
        oversampling=oversampling,
        conv_filters=trial.suggest_categorical("conv_filters", [32, 64, 96, 128]),
        kernel_size=trial.suggest_categorical("kernel_size", [3, 5, 7]),
        bilstm_hidden=trial.suggest_categorical("bilstm_hidden", [64, 96, 128]),
        dense_hidden=trial.suggest_categorical("dense_hidden", [64, 128, 192]),
        dropout=trial.suggest_float("dropout", 0.1, 0.5, step=0.05),
        lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
        label_smoothing=trial.suggest_float("label_smoothing", 0.0, 0.1, step=0.05),
        loss=loss,
        datasets=(dataset,),
        model_variant=model_variant,
        num_conv_layers=trial.suggest_categorical("num_conv_layers", [3, 4, 5]),
        num_bilstm_layers=trial.suggest_categorical("num_bilstm_layers", [1, 2]),
        use_batchnorm=model_variant == "improved" and trial.suggest_categorical("use_batchnorm", [True, False]),
        use_residual=model_variant == "improved" and trial.suggest_categorical("use_residual", [True, False]),
        use_attention=dataset in STUDENT_DATASETS and model_variant == "improved" and trial.suggest_categorical("use_attention", [True, False]),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        patience=trial.suggest_categorical("patience", [5, 10, 15, 20]),
    )


def case_from_optuna_params(dataset: str, params: dict[str, Any], name: str) -> SweepCase:
    loss = str(params.get("loss", "ce"))
    oversampling = "none" if loss == "class_weight" else str(params.get("oversampling", "none"))
    model_variant = str(params.get("model_variant", "exact"))
    return SweepCase(
        name=name,
        feature_set=str(params.get("feature_set", "paper")),
        oversampling=oversampling,
        conv_filters=int(params.get("conv_filters", 64)),
        kernel_size=int(params.get("kernel_size", 3)),
        bilstm_hidden=int(params.get("bilstm_hidden", 64)),
        dense_hidden=int(params.get("dense_hidden", 128)),
        dropout=float(params.get("dropout", 0.2)),
        lr=float(params.get("lr", 1e-3)),
        batch_size=int(params.get("batch_size", 32)),
        label_smoothing=float(params.get("label_smoothing", 0.0)),
        loss=loss,
        datasets=(dataset,),
        model_variant=model_variant,
        num_conv_layers=int(params.get("num_conv_layers", 4)),
        num_bilstm_layers=int(params.get("num_bilstm_layers", 1)),
        use_batchnorm=bool(params.get("use_batchnorm", False)) if model_variant == "improved" else False,
        use_residual=bool(params.get("use_residual", False)) if model_variant == "improved" else False,
        use_attention=bool(params.get("use_attention", False)) if dataset in STUDENT_DATASETS and model_variant == "improved" else False,
        weight_decay=float(params.get("weight_decay", 1e-4)),
        patience=int(params.get("patience", 10)),
    )


def append_optuna_trials(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    data = pd.DataFrame(rows)
    if STRICT_OPTUNA_TRIALS_PATH.exists():
        data = pd.concat([pd.read_csv(STRICT_OPTUNA_TRIALS_PATH), data], ignore_index=True)
    data.to_csv(STRICT_OPTUNA_TRIALS_PATH, index=False)


def run_strict_optuna(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import optuna
    except Exception as exc:
        raise RuntimeError("Optuna is required for strict Optuna runs.") from exc
    ensure_dirs()
    datasets = ALL_DATASETS if args.datasets == "all" else parse_csv_strings(args.datasets)
    all_payload: dict[str, Any] = {}
    trial_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        raw = read_dataset(dataset)

        def objective(trial: Any) -> float:
            case = make_trial_case(dataset, trial)
            result = train_strict_case_seed(dataset, raw, case, int(args.seed), int(args.epochs))
            row = {
                "dataset": dataset,
                "trial": int(trial.number),
                "seed": int(args.seed),
                "epochs": int(args.epochs),
                "val_accuracy": result.row["val_accuracy"],
                "val_f1_macro": result.row["val_f1_macro"],
                "best_epoch": result.row["best_epoch"],
                "case_params": json.dumps(asdict(case), ensure_ascii=True, sort_keys=True),
            }
            trial_rows.append(row)
            return float(result.row["val_f1_macro"])

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=int(args.seed)))
        study.optimize(objective, n_trials=int(args.trials))
        best_case = case_from_optuna_params(dataset, study.best_params, f"strict_optuna_best_{dataset}")
        best_result = train_strict_case_seed(dataset, raw, best_case, int(args.seed), int(args.train_epochs or args.epochs))
        selected = evaluate_selected_on_test(dataset, best_case, best_result)
        model_path = save_strict_model(dataset, selected, best_result.state_dict, best_case)
        selected["model_path"] = model_path
        all_payload[dataset] = {
            "best_value": float(study.best_value),
            "best_trial": int(study.best_trial.number),
            "best_params": study.best_params,
            "objective": "validation_macro_f1",
            "final_selected": selected,
        }
        print(
            f"OPTUNA {dataset} trials={args.trials} best_val_f1={study.best_value:.4f} "
            f"test_f1={selected['test_f1_macro']:.4f}",
            flush=True,
        )
    append_optuna_trials(trial_rows)
    payload = {"config": vars(args), "datasets": all_payload, "trials_csv": str(STRICT_OPTUNA_TRIALS_PATH)}
    save_json(STRICT_OPTUNA_BEST_PATH, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V18 strict validation CNN-BiLSTM experiments.")
    sub = parser.add_subparsers(dest="command", required=True)
    sweep = sub.add_parser("sweep")
    sweep.add_argument("--datasets", default="student-mat,student-por,xapi")
    sweep.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_STRICT_SEEDS))
    sweep.add_argument("--epochs", type=int, default=80)
    sweep.add_argument("--case-names", default="")
    sweep.add_argument("--max-cases", type=int, default=0)
    optuna_parser = sub.add_parser("optuna")
    optuna_parser.add_argument("--datasets", default="student-mat")
    optuna_parser.add_argument("--trials", type=int, default=50)
    optuna_parser.add_argument("--epochs", type=int, default=50)
    optuna_parser.add_argument("--train-epochs", type=int, default=0)
    optuna_parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "sweep":
        run_strict_sweep(args)
    elif args.command == "optuna":
        run_strict_optuna(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
