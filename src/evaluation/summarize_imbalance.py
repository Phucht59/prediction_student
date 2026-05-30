from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCENARIO_ORDER = ["mid", "late"]


def read_required_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name} file: {path}")
    data = pd.read_csv(path)
    if "scenario" in data.columns:
        data = data[data["scenario"].ne("early")].copy()
    if data.empty:
        raise ValueError(f"{name} file is empty: {path}")
    return data


def scenario_sort_key(series: pd.Series) -> pd.Series:
    if series.name == "scenario":
        return series.map({scenario: index for index, scenario in enumerate(SCENARIO_ORDER)})
    return series


def select_test_row(group: pd.DataFrame, selected_row: pd.Series) -> pd.Series:
    test_rows = group[
        (group["model_name"] == selected_row["model_name"])
        & (group["imbalance_strategy"] == selected_row["imbalance_strategy"])
        & (group["split"] == "test")
    ]
    if test_rows.empty:
        raise ValueError(
            "Missing test row for "
            f"{selected_row['dataset']}/{selected_row['scenario']}/"
            f"{selected_row['imbalance_strategy']}/{selected_row['model_name']}"
        )
    return test_rows.iloc[0]


def select_train_row(group: pd.DataFrame, selected_row: pd.Series) -> pd.Series:
    train_rows = group[
        (group["model_name"] == selected_row["model_name"])
        & (group["imbalance_strategy"] == selected_row["imbalance_strategy"])
        & (group["split"] == "train")
    ]
    if train_rows.empty:
        raise ValueError(
            "Missing train row for "
            f"{selected_row['dataset']}/{selected_row['scenario']}/"
            f"{selected_row['imbalance_strategy']}/{selected_row['model_name']}"
        )
    return train_rows.iloc[0]


def build_summary_by_macro_f1(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (dataset, scenario), group in results.groupby(["dataset", "scenario"], sort=False):
        validation_rows = group[group["split"] == "val"]
        selected = validation_rows.loc[validation_rows["f1_macro"].idxmax()]
        test = select_test_row(group, selected)
        train = select_train_row(group, selected)
        rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "best_model": selected["model_name"],
                "best_imbalance_strategy": selected["imbalance_strategy"],
                "val_f1_macro": selected["f1_macro"],
                "test_f1_macro": test["f1_macro"],
                "test_accuracy": test["accuracy"],
                "test_recall_weak": test["recall_weak"],
                "test_f1_weak": test["f1_weak"],
                "train_f1_macro": train["f1_macro"],
                "val_accuracy": selected["accuracy"],
                "seed": selected["seed"],
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "scenario"], key=scenario_sort_key)


def build_summary_by_weak_recall(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (dataset, scenario), group in results.groupby(["dataset", "scenario"], sort=False):
        validation_rows = group[group["split"] == "val"].sort_values(
            ["recall_weak", "f1_macro"],
            ascending=[False, False],
        )
        selected = validation_rows.iloc[0]
        test = select_test_row(group, selected)
        train = select_train_row(group, selected)
        rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "best_model": selected["model_name"],
                "best_imbalance_strategy": selected["imbalance_strategy"],
                "val_recall_weak": selected["recall_weak"],
                "val_f1_macro": selected["f1_macro"],
                "test_recall_weak": test["recall_weak"],
                "test_f1_weak": test["f1_weak"],
                "test_f1_macro": test["f1_macro"],
                "test_accuracy": test["accuracy"],
                "train_recall_weak": train["recall_weak"],
                "train_f1_macro": train["f1_macro"],
                "seed": selected["seed"],
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "scenario"], key=scenario_sort_key)


def select_baseline_best(baseline_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (dataset, scenario), group in baseline_results.groupby(["dataset", "scenario"], sort=False):
        validation_rows = group[group["split"] == "val"]
        selected = validation_rows.loc[validation_rows["f1_macro"].idxmax()]
        test = group[(group["model_name"] == selected["model_name"]) & (group["split"] == "test")].iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "baseline_model": selected["model_name"],
                "baseline_val_f1_macro": selected["f1_macro"],
                "baseline_test_f1_macro": test["f1_macro"],
                "baseline_test_accuracy": test["accuracy"],
                "baseline_test_recall_weak": test["recall_weak"],
            }
        )
    return pd.DataFrame(rows)


def build_baseline_comparison(
    baseline_results: pd.DataFrame,
    imbalance_summary: pd.DataFrame,
) -> pd.DataFrame:
    baseline_best = select_baseline_best(baseline_results)
    comparison = baseline_best.merge(imbalance_summary, on=["dataset", "scenario"], how="inner")
    comparison = comparison.rename(
        columns={
            "best_model": "imbalance_model",
            "best_imbalance_strategy": "imbalance_strategy",
            "val_f1_macro": "imbalance_val_f1_macro",
            "test_f1_macro": "imbalance_test_f1_macro",
            "test_accuracy": "imbalance_test_accuracy",
            "test_recall_weak": "imbalance_test_recall_weak",
        }
    )
    comparison["delta_test_f1_macro"] = (
        comparison["imbalance_test_f1_macro"] - comparison["baseline_test_f1_macro"]
    )
    comparison["delta_test_recall_weak"] = (
        comparison["imbalance_test_recall_weak"] - comparison["baseline_test_recall_weak"]
    )
    comparison["delta_test_accuracy"] = (
        comparison["imbalance_test_accuracy"] - comparison["baseline_test_accuracy"]
    )
    desired_columns = [
        "dataset",
        "scenario",
        "baseline_model",
        "baseline_val_f1_macro",
        "baseline_test_f1_macro",
        "baseline_test_accuracy",
        "baseline_test_recall_weak",
        "imbalance_model",
        "imbalance_strategy",
        "imbalance_val_f1_macro",
        "imbalance_test_f1_macro",
        "imbalance_test_accuracy",
        "imbalance_test_recall_weak",
        "delta_test_f1_macro",
        "delta_test_recall_weak",
        "delta_test_accuracy",
    ]
    return comparison[desired_columns].sort_values(["dataset", "scenario"], key=scenario_sort_key)


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    lines.extend(" | ".join(row) for row in rows)
    return lines


def fmt(value: float) -> str:
    return f"{value:.4f}"


def analyze_strategy_counts(macro_summary: pd.DataFrame, weak_summary: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    macro_counts = macro_summary["best_imbalance_strategy"].value_counts()
    weak_counts = weak_summary["best_imbalance_strategy"].value_counts()
    lines.append(
        "- Best-by-Macro-F1 strategy counts: "
        + ", ".join(f"{strategy} ({count})" for strategy, count in macro_counts.items())
        + "."
    )
    lines.append(
        "- Best-by-weak-recall strategy counts: "
        + ", ".join(f"{strategy} ({count})" for strategy, count in weak_counts.items())
        + "."
    )
    return lines


def analyze_baseline_deltas(comparison: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    improved_recall = comparison[comparison["delta_test_recall_weak"] > 0]
    worse_recall = comparison[comparison["delta_test_recall_weak"] < 0]
    improved_macro = comparison[comparison["delta_test_f1_macro"] > 0]
    worse_macro = comparison[comparison["delta_test_f1_macro"] < 0]

    lines.append(
        f"- Weak recall improved in {len(improved_recall)}/{len(comparison)} dataset-scenario pairs "
        f"when selecting by validation Macro-F1."
    )
    if not worse_recall.empty:
        items = [
            f"{row.dataset}/{row.scenario} ({row.delta_test_recall_weak:+.4f})"
            for row in worse_recall.itertuples()
        ]
        lines.append("- Weak recall got worse in: " + "; ".join(items) + ".")
    lines.append(
        f"- Test Macro-F1 improved in {len(improved_macro)}/{len(comparison)} pairs; "
        f"got worse in {len(worse_macro)}/{len(comparison)} pairs."
    )

    tradeoffs = comparison[
        (comparison["delta_test_recall_weak"] > 0)
        & (
            (comparison["delta_test_f1_macro"] < 0)
            | (comparison["delta_test_accuracy"] < 0)
        )
    ]
    if tradeoffs.empty:
        lines.append("- No selected Macro-F1 configuration showed higher weak recall with lower test Macro-F1 or accuracy.")
    else:
        items = [
            f"{row.dataset}/{row.scenario} "
            f"(delta recall {row.delta_test_recall_weak:+.4f}, "
            f"delta Macro-F1 {row.delta_test_f1_macro:+.4f}, "
            f"delta accuracy {row.delta_test_accuracy:+.4f})"
            for row in tradeoffs.itertuples()
        ]
        lines.append("- Trade-offs observed: " + "; ".join(items) + ".")
    return lines


def analyze_scenario_gain(comparison: pd.DataFrame) -> list[str]:
    scenario_delta = comparison.groupby("scenario")[["delta_test_f1_macro", "delta_test_recall_weak"]].mean()
    ordered = scenario_delta.reindex(SCENARIO_ORDER)
    best_recall_scenario = ordered["delta_test_recall_weak"].idxmax()
    best_macro_scenario = ordered["delta_test_f1_macro"].idxmax()
    return [
        "- Average delta by scenario "
        + "; ".join(
            f"{scenario}: Macro-F1 {row.delta_test_f1_macro:+.4f}, weak recall {row.delta_test_recall_weak:+.4f}"
            for scenario, row in ordered.iterrows()
        )
        + ".",
        f"- Largest average weak-recall gain is in {best_recall_scenario}; largest average Macro-F1 gain is in {best_macro_scenario}.",
    ]


def analyze_imbalance_strategies(results: pd.DataFrame, baseline_results: pd.DataFrame) -> list[str]:
    baseline_best = select_baseline_best(baseline_results)
    lines: list[str] = []
    for strategy in ("smote", "adasyn", "class_weight_balanced"):
        better = 0
        total = 0
        for row in baseline_best.itertuples():
            subset = results[
                (results["dataset"] == row.dataset)
                & (results["scenario"] == row.scenario)
                & (results["imbalance_strategy"] == strategy)
                & (results["split"] == "test")
            ]
            if subset.empty:
                continue
            total += 1
            if subset["f1_macro"].max() > row.baseline_test_f1_macro:
                better += 1
        if total > 0:
            lines.append(
                f"- {strategy} beat the Prompt 3 best baseline test Macro-F1 in {better}/{total} comparable pairs."
            )
    return lines


def analyze_overfitting(macro_summary: pd.DataFrame, weak_summary: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    macro_overfit = macro_summary[(macro_summary["train_f1_macro"] - macro_summary["val_f1_macro"]) > 0.20]
    weak_overfit = weak_summary[(weak_summary["train_f1_macro"] - weak_summary["val_f1_macro"]) > 0.20]
    if macro_overfit.empty:
        lines.append("- Best-by-Macro-F1 configs do not show train-vs-val Macro-F1 gaps above 0.20.")
    else:
        items = [
            f"{row.dataset}/{row.scenario}/{row.best_imbalance_strategy}/{row.best_model} "
            f"(train {row.train_f1_macro:.4f}, val {row.val_f1_macro:.4f})"
            for row in macro_overfit.itertuples()
        ]
        lines.append("- Possible overfitting by Macro-F1 summary: " + "; ".join(items) + ".")

    if weak_overfit.empty:
        lines.append("- Best-by-weak-recall configs do not show train-vs-val Macro-F1 gaps above 0.20.")
    else:
        items = [
            f"{row.dataset}/{row.scenario}/{row.best_imbalance_strategy}/{row.best_model} "
            f"(train {row.train_f1_macro:.4f}, val {row.val_f1_macro:.4f})"
            for row in weak_overfit.itertuples()
        ]
        lines.append("- Possible overfitting by weak-recall summary: " + "; ".join(items) + ".")
    return lines


def write_report(
    macro_summary: pd.DataFrame,
    weak_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    results: pd.DataFrame,
    baseline_results: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = ["## Imbalance Handling Summary Report", ""]

    lines.append("### 1. Best configurations by validation Macro-F1")
    lines.extend(
        markdown_table(
            [
                "Dataset",
                "Scenario",
                "Best model",
                "Strategy",
                "Val Macro-F1",
                "Test Macro-F1",
                "Test Accuracy",
                "Test Recall weak",
                "Test F1 weak",
            ],
            [
                [
                    row.dataset,
                    row.scenario,
                    row.best_model,
                    row.best_imbalance_strategy,
                    fmt(row.val_f1_macro),
                    fmt(row.test_f1_macro),
                    fmt(row.test_accuracy),
                    fmt(row.test_recall_weak),
                    fmt(row.test_f1_weak),
                ]
                for row in macro_summary.itertuples()
            ],
        )
    )
    lines.append("")

    lines.append("### 2. Best configurations by validation weak recall")
    lines.extend(
        markdown_table(
            [
                "Dataset",
                "Scenario",
                "Best model",
                "Strategy",
                "Val Recall weak",
                "Test Recall weak",
                "Test F1 weak",
                "Test Macro-F1",
                "Test Accuracy",
            ],
            [
                [
                    row.dataset,
                    row.scenario,
                    row.best_model,
                    row.best_imbalance_strategy,
                    fmt(row.val_recall_weak),
                    fmt(row.test_recall_weak),
                    fmt(row.test_f1_weak),
                    fmt(row.test_f1_macro),
                    fmt(row.test_accuracy),
                ]
                for row in weak_summary.itertuples()
            ],
        )
    )
    lines.append("")

    lines.append("### 3. Comparison with Prompt 3 baseline")
    lines.extend(
        markdown_table(
            [
                "Dataset",
                "Scenario",
                "Baseline test Macro-F1",
                "Imbalance test Macro-F1",
                "Delta Macro-F1",
                "Baseline test recall weak",
                "Imbalance test recall weak",
                "Delta recall weak",
            ],
            [
                [
                    row.dataset,
                    row.scenario,
                    fmt(row.baseline_test_f1_macro),
                    fmt(row.imbalance_test_f1_macro),
                    fmt(row.delta_test_f1_macro),
                    fmt(row.baseline_test_recall_weak),
                    fmt(row.imbalance_test_recall_weak),
                    fmt(row.delta_test_recall_weak),
                ]
                for row in comparison.itertuples()
            ],
        )
    )
    lines.append("")

    lines.append("### 4. Automatic observations")
    observations: list[str] = []
    observations.extend(analyze_strategy_counts(macro_summary, weak_summary))
    observations.extend(analyze_baseline_deltas(comparison))
    observations.extend(analyze_scenario_gain(comparison))
    observations.extend(analyze_imbalance_strategies(results, baseline_results))
    observations.extend(analyze_overfitting(macro_summary, weak_summary))
    lines.extend(observations)
    lines.append("")

    lines.append("### 5. Technical limitations")
    lines.append("- Resampling is applied only to the training split.")
    lines.append("- Validation and test splits keep the original class distribution.")
    lines.append(
        "- The processed data is already one-hot encoded, so SMOTE/ADASYN synthetic samples "
        "can contain fractional values in one-hot feature positions."
    )
    lines.append(
        "- These results should be treated as comparative experiments, not final conclusions."
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    results_dir = PROJECT_ROOT / "reports" / "results"
    tables_dir = PROJECT_ROOT / "reports" / "tables"

    imbalance_path = results_dir / "imbalance_classification_results.csv"
    baseline_path = results_dir / "baseline_classification_results.csv"
    if not imbalance_path.exists():
        print(f"Skipping imbalance summary; missing results file: {imbalance_path}")
        return 0
    if not baseline_path.exists():
        print(f"Skipping imbalance summary; missing baseline file: {baseline_path}")
        return 0

    imbalance_results = read_required_csv(
        imbalance_path,
        "imbalance classification results",
    )
    baseline_results = read_required_csv(
        baseline_path,
        "baseline classification results",
    )

    macro_summary = build_summary_by_macro_f1(imbalance_results)
    weak_summary = build_summary_by_weak_recall(imbalance_results)
    comparison = build_baseline_comparison(baseline_results, macro_summary)

    tables_dir.mkdir(parents=True, exist_ok=True)
    macro_summary.to_csv(
        tables_dir / "imbalance_classification_summary_by_macro_f1.csv",
        index=False,
    )
    weak_summary.to_csv(
        tables_dir / "imbalance_classification_summary_by_weak_recall.csv",
        index=False,
    )
    comparison.to_csv(tables_dir / "imbalance_vs_baseline_comparison.csv", index=False)
    write_report(
        macro_summary,
        weak_summary,
        comparison,
        imbalance_results,
        baseline_results,
        tables_dir / "imbalance_summary_report.txt",
    )

    print(
        "Macro-F1 summary saved to: "
        f"{tables_dir / 'imbalance_classification_summary_by_macro_f1.csv'}"
    )
    print(
        "Weak-recall summary saved to: "
        f"{tables_dir / 'imbalance_classification_summary_by_weak_recall.csv'}"
    )
    print(f"Baseline comparison saved to: {tables_dir / 'imbalance_vs_baseline_comparison.csv'}")
    print(f"Text report saved to: {tables_dir / 'imbalance_summary_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
