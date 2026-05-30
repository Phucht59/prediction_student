from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCENARIO_ORDER = ["mid", "late"]


def read_results(path: Path, task_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {task_name} results file: {path}")
    data = pd.read_csv(path)
    if "scenario" in data.columns:
        data = data[data["scenario"].ne("early")].copy()
    if data.empty:
        raise ValueError(f"{task_name} results file is empty: {path}")
    return data


def build_classification_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (dataset, scenario), group in results.groupby(["dataset", "scenario"], sort=False):
        validation_rows = group[group["split"] == "val"]
        if validation_rows.empty:
            continue

        best_validation = validation_rows.loc[validation_rows["f1_macro"].idxmax()]
        model_name = best_validation["model_name"]
        test_row = group[(group["model_name"] == model_name) & (group["split"] == "test")].iloc[0]
        train_row = group[(group["model_name"] == model_name) & (group["split"] == "train")].iloc[0]

        rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "best_model": model_name,
                "val_macro_f1": best_validation["f1_macro"],
                "test_macro_f1": test_row["f1_macro"],
                "test_accuracy": test_row["accuracy"],
                "test_recall_weak": test_row["recall_weak"],
                "train_macro_f1": train_row["f1_macro"],
                "val_accuracy": best_validation["accuracy"],
                "seed": best_validation["seed"],
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "scenario"], key=scenario_sort_key)


def build_regression_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (dataset, scenario), group in results.groupby(["dataset", "scenario"], sort=False):
        validation_rows = group[group["split"] == "val"]
        if validation_rows.empty:
            continue

        best_validation = validation_rows.loc[validation_rows["rmse"].idxmin()]
        model_name = best_validation["model_name"]
        test_row = group[(group["model_name"] == model_name) & (group["split"] == "test")].iloc[0]
        train_row = group[(group["model_name"] == model_name) & (group["split"] == "train")].iloc[0]

        rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "best_model": model_name,
                "val_rmse": best_validation["rmse"],
                "test_rmse": test_row["rmse"],
                "test_mae": test_row["mae"],
                "test_r2": test_row["r2"],
                "train_rmse": train_row["rmse"],
                "val_mae": best_validation["mae"],
                "seed": best_validation["seed"],
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "scenario"], key=scenario_sort_key)


def scenario_sort_key(series: pd.Series) -> pd.Series:
    if series.name == "scenario":
        return series.map({scenario: index for index, scenario in enumerate(SCENARIO_ORDER)})
    return series


def format_float(value: float) -> str:
    return f"{value:.4f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    lines.extend(" | ".join(row) for row in rows)
    return lines


def analyze_scenario_difficulty(class_summary: pd.DataFrame, reg_summary: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if not class_summary.empty:
        class_by_scenario = class_summary.groupby("scenario")["test_macro_f1"].mean()
        easiest = class_by_scenario.idxmax()
        hardest = class_by_scenario.idxmin()
        lines.append(
            "- Classification: scenario with highest average test Macro-F1 is "
            f"{easiest} ({class_by_scenario[easiest]:.4f}); lowest is "
            f"{hardest} ({class_by_scenario[hardest]:.4f})."
        )

    if not reg_summary.empty:
        reg_by_scenario = reg_summary.groupby("scenario")["test_rmse"].mean()
        easiest = reg_by_scenario.idxmin()
        hardest = reg_by_scenario.idxmax()
        lines.append(
            "- Regression: scenario with lowest average test RMSE is "
            f"{easiest} ({reg_by_scenario[easiest]:.4f}); highest is "
            f"{hardest} ({reg_by_scenario[hardest]:.4f})."
        )
    return lines


def analyze_grade_features(class_summary: pd.DataFrame, reg_summary: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    for dataset in sorted(set(class_summary.get("dataset", [])) | set(reg_summary.get("dataset", []))):
        class_part = class_summary[class_summary["dataset"] == dataset].set_index("scenario")
        reg_part = reg_summary[reg_summary["dataset"] == dataset].set_index("scenario")

        if {"mid", "late"}.issubset(class_part.index):
            values = class_part.loc[SCENARIO_ORDER, "test_macro_f1"]
            lines.append(
                f"- {dataset} classification test Macro-F1 mid/late: "
                f"{values.iloc[0]:.4f} -> {values.iloc[1]:.4f}."
            )

        if {"mid", "late"}.issubset(reg_part.index):
            values = reg_part.loc[SCENARIO_ORDER, "test_rmse"]
            lines.append(
                f"- {dataset} regression test RMSE mid/late: "
                f"{values.iloc[0]:.4f} -> {values.iloc[1]:.4f}."
            )
    return lines


def analyze_best_models(class_summary: pd.DataFrame, reg_summary: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if not class_summary.empty:
        counts = class_summary["best_model"].value_counts()
        leaders = ", ".join(f"{model} ({count})" for model, count in counts.items())
        lines.append(f"- Classification best-model counts: {leaders}.")
    if not reg_summary.empty:
        counts = reg_summary["best_model"].value_counts()
        leaders = ", ".join(f"{model} ({count})" for model, count in counts.items())
        lines.append(f"- Regression best-model counts: {leaders}.")
    return lines


def analyze_overfitting(class_summary: pd.DataFrame, reg_summary: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    class_overfit = class_summary[
        (class_summary["train_macro_f1"] - class_summary["val_macro_f1"]) > 0.20
    ]
    if class_overfit.empty:
        lines.append("- Classification: no large train-vs-validation Macro-F1 gap above 0.20 among selected models.")
    else:
        items = [
            f"{row.dataset}/{row.scenario}/{row.best_model} "
            f"(train {row.train_macro_f1:.4f}, val {row.val_macro_f1:.4f})"
            for row in class_overfit.itertuples()
        ]
        lines.append("- Classification possible overfitting: " + "; ".join(items) + ".")

    reg_overfit = reg_summary[(reg_summary["val_rmse"] - reg_summary["train_rmse"]) > 1.0]
    if reg_overfit.empty:
        lines.append("- Regression: no large train-vs-validation RMSE gap above 1.0 among selected models.")
    else:
        items = [
            f"{row.dataset}/{row.scenario}/{row.best_model} "
            f"(train {row.train_rmse:.4f}, val {row.val_rmse:.4f})"
            for row in reg_overfit.itertuples()
        ]
        lines.append("- Regression possible overfitting: " + "; ".join(items) + ".")
    return lines


def analyze_weak_recall(class_summary: pd.DataFrame) -> list[str]:
    if class_summary.empty:
        return []

    low = class_summary[class_summary["test_recall_weak"] < 0.60]
    if low.empty:
        return ["- Weak-class recall is at least 0.60 for all selected classification baselines."]

    items = [
        f"{row.dataset}/{row.scenario}={row.test_recall_weak:.4f}"
        for row in low.itertuples()
    ]
    return ["- Weak-class recall below 0.60 appears in: " + "; ".join(items) + "."]


def write_report(class_summary: pd.DataFrame, reg_summary: pd.DataFrame, report_path: Path) -> None:
    lines = ["## Baseline Summary Report", ""]

    lines.append("### Classification best models by validation Macro-F1")
    class_rows = [
        [
            row.dataset,
            row.scenario,
            row.best_model,
            format_float(row.val_macro_f1),
            format_float(row.test_macro_f1),
            format_float(row.test_accuracy),
            format_float(row.test_recall_weak),
        ]
        for row in class_summary.itertuples()
    ]
    lines.extend(
        markdown_table(
            [
                "Dataset",
                "Scenario",
                "Best model",
                "Val Macro-F1",
                "Test Macro-F1",
                "Test Accuracy",
                "Test Recall weak",
            ],
            class_rows,
        )
    )
    lines.append("")

    lines.append("### Regression best models by validation RMSE")
    reg_rows = [
        [
            row.dataset,
            row.scenario,
            row.best_model,
            format_float(row.val_rmse),
            format_float(row.test_rmse),
            format_float(row.test_mae),
            format_float(row.test_r2),
        ]
        for row in reg_summary.itertuples()
    ]
    lines.extend(
        markdown_table(
            ["Dataset", "Scenario", "Best model", "Val RMSE", "Test RMSE", "Test MAE", "Test R2"],
            reg_rows,
        )
    )
    lines.append("")

    lines.append("### Automatic observations")
    observations: list[str] = []
    observations.extend(analyze_scenario_difficulty(class_summary, reg_summary))
    observations.extend(analyze_grade_features(class_summary, reg_summary))
    observations.extend(analyze_best_models(class_summary, reg_summary))
    observations.extend(analyze_overfitting(class_summary, reg_summary))
    observations.extend(analyze_weak_recall(class_summary))
    lines.extend(observations)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    results_dir = PROJECT_ROOT / "reports" / "results"
    tables_dir = PROJECT_ROOT / "reports" / "tables"

    class_results = read_results(
        results_dir / "baseline_classification_results.csv",
        "classification",
    )
    reg_results = read_results(results_dir / "baseline_regression_results.csv", "regression")

    class_summary = build_classification_summary(class_results)
    reg_summary = build_regression_summary(reg_results)

    tables_dir.mkdir(parents=True, exist_ok=True)
    class_summary.to_csv(tables_dir / "baseline_classification_summary.csv", index=False)
    reg_summary.to_csv(tables_dir / "baseline_regression_summary.csv", index=False)
    write_report(class_summary, reg_summary, tables_dir / "baseline_summary_report.txt")

    print(f"Classification summary saved to: {tables_dir / 'baseline_classification_summary.csv'}")
    print(f"Regression summary saved to: {tables_dir / 'baseline_regression_summary.csv'}")
    print(f"Text report saved to: {tables_dir / 'baseline_summary_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
