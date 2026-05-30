from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCENARIO_ORDER = ["mid", "late"]


def scenario_sort_key(series: pd.Series) -> pd.Series:
    if series.name == "scenario":
        return series.map({scenario: index for index, scenario in enumerate(SCENARIO_ORDER)})
    return series


def read_required_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name} file: {path}")
    data = pd.read_csv(path)
    if "scenario" in data.columns:
        data = data[data["scenario"].ne("early")].copy()
    if data.empty:
        raise ValueError(f"{name} file is empty: {path}")
    return data


def read_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    data = pd.read_csv(path)
    if "scenario" in data.columns:
        data = data[data["scenario"].ne("early")].copy()
    return None if data.empty else data


def select_matching_row(group: pd.DataFrame, selected: pd.Series, split: str) -> pd.Series:
    rows = group[
        (group["model_name"] == selected["model_name"])
        & (group["loss_weight"] == selected["loss_weight"])
        & (group["split"] == split)
    ]
    if rows.empty:
        raise ValueError(
            f"Missing {split} row for {selected['dataset']}/{selected['scenario']}/"
            f"{selected['model_name']}/{selected['loss_weight']}"
        )
    return rows.iloc[0]


def build_deep_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (dataset, scenario), group in results.groupby(["dataset", "scenario"], sort=False):
        validation_rows = group[group["split"] == "val"]
        selected = validation_rows.loc[validation_rows["f1_macro"].idxmax()]
        test = select_matching_row(group, selected, "test")
        train = select_matching_row(group, selected, "train")
        rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "best_deep_model": selected["model_name"],
                "loss_weight": selected["loss_weight"],
                "val_f1_macro": selected["f1_macro"],
                "test_f1_macro": test["f1_macro"],
                "test_accuracy": test["accuracy"],
                "test_recall_weak": test["recall_weak"],
                "test_f1_weak": test["f1_weak"],
                "best_epoch": selected["best_epoch"],
                "train_f1_macro": train["f1_macro"],
                "train_recall_weak": train["recall_weak"],
                "epochs_ran": selected["epochs_ran"],
                "seed": selected["seed"],
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "scenario"], key=scenario_sort_key)


def baseline_summary_from_results(path: Path) -> pd.DataFrame | None:
    results = read_optional_csv(path)
    if results is None:
        return None
    rows: list[dict] = []
    for (dataset, scenario), group in results.groupby(["dataset", "scenario"], sort=False):
        validation_rows = group[group["split"] == "val"]
        selected = validation_rows.loc[validation_rows["f1_macro"].idxmax()]
        test = group[(group["model_name"] == selected["model_name"]) & (group["split"] == "test")].iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "baseline_test_f1_macro": test["f1_macro"],
                "baseline_test_recall_weak": test["recall_weak"],
            }
        )
    return pd.DataFrame(rows)


def load_baseline_summary() -> pd.DataFrame | None:
    summary_path = PROJECT_ROOT / "reports" / "tables" / "baseline_classification_summary.csv"
    summary = read_optional_csv(summary_path)
    if summary is not None:
        return summary.rename(
            columns={
                "test_macro_f1": "baseline_test_f1_macro",
                "test_recall_weak": "baseline_test_recall_weak",
            }
        )[["dataset", "scenario", "baseline_test_f1_macro", "baseline_test_recall_weak"]]
    return baseline_summary_from_results(
        PROJECT_ROOT / "reports" / "results" / "baseline_classification_results.csv"
    )


def load_imbalance_summary() -> pd.DataFrame | None:
    path = PROJECT_ROOT / "reports" / "tables" / "imbalance_classification_summary_by_macro_f1.csv"
    data = read_optional_csv(path)
    if data is None:
        return None
    return data.rename(
        columns={
            "test_f1_macro": "imbalance_test_f1_macro",
            "test_recall_weak": "imbalance_test_recall_weak",
        }
    )[["dataset", "scenario", "imbalance_test_f1_macro", "imbalance_test_recall_weak"]]


def build_comparison(deep_summary: pd.DataFrame) -> pd.DataFrame:
    comparison = deep_summary[
        ["dataset", "scenario", "test_f1_macro", "test_recall_weak"]
    ].rename(
        columns={
            "test_f1_macro": "deep_test_f1_macro",
            "test_recall_weak": "deep_test_recall_weak",
        }
    )

    baseline = load_baseline_summary()
    if baseline is not None:
        comparison = comparison.merge(baseline, on=["dataset", "scenario"], how="left")
    else:
        comparison["baseline_test_f1_macro"] = pd.NA
        comparison["baseline_test_recall_weak"] = pd.NA

    imbalance = load_imbalance_summary()
    if imbalance is not None:
        comparison = comparison.merge(imbalance, on=["dataset", "scenario"], how="left")
    else:
        comparison["imbalance_test_f1_macro"] = pd.NA
        comparison["imbalance_test_recall_weak"] = pd.NA

    comparison["delta_deep_vs_baseline"] = (
        comparison["deep_test_f1_macro"] - comparison["baseline_test_f1_macro"]
    )
    comparison["delta_deep_vs_imbalance"] = (
        comparison["deep_test_f1_macro"] - comparison["imbalance_test_f1_macro"]
    )
    comparison["delta_deep_recall_weak_vs_baseline"] = (
        comparison["deep_test_recall_weak"] - comparison["baseline_test_recall_weak"]
    )
    comparison["delta_deep_recall_weak_vs_imbalance"] = (
        comparison["deep_test_recall_weak"] - comparison["imbalance_test_recall_weak"]
    )

    desired_columns = [
        "dataset",
        "scenario",
        "baseline_test_f1_macro",
        "imbalance_test_f1_macro",
        "deep_test_f1_macro",
        "delta_deep_vs_baseline",
        "delta_deep_vs_imbalance",
        "baseline_test_recall_weak",
        "imbalance_test_recall_weak",
        "deep_test_recall_weak",
        "delta_deep_recall_weak_vs_baseline",
        "delta_deep_recall_weak_vs_imbalance",
    ]
    return comparison[desired_columns].sort_values(["dataset", "scenario"], key=scenario_sort_key)


def fmt(value) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    lines.extend(" | ".join(row) for row in rows)
    return lines


def analyze_deep_vs_baselines(comparison: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if comparison["baseline_test_f1_macro"].notna().any():
        valid = comparison.dropna(subset=["delta_deep_vs_baseline"])
        wins = int((valid["delta_deep_vs_baseline"] > 0).sum())
        lines.append(f"- Deep learning beat the Prompt 3 baseline in {wins}/{len(valid)} pairs by test Macro-F1.")
    if comparison["imbalance_test_f1_macro"].notna().any():
        valid = comparison.dropna(subset=["delta_deep_vs_imbalance"])
        wins = int((valid["delta_deep_vs_imbalance"] > 0).sum())
        lines.append(f"- Deep learning beat the Prompt 4 imbalance best config in {wins}/{len(valid)} pairs by test Macro-F1.")
    return lines


def best_model_per_family(results: pd.DataFrame) -> pd.DataFrame:
    validation = results[results["split"] == "val"].copy()
    rows: list[pd.Series] = []
    for (_, _, model_name), group in validation.groupby(["dataset", "scenario", "model_name"]):
        rows.append(group.loc[group["f1_macro"].idxmax()])
    return pd.DataFrame(rows)


def analyze_architectures(results: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    family_best = best_model_per_family(results)

    cnn_better = 0
    mlp_pairs = 0
    for (dataset, scenario), group in family_best.groupby(["dataset", "scenario"]):
        model_scores = group.set_index("model_name")["f1_macro"]
        if {"mlp_static", "cnn_bilstm_tabular"}.issubset(model_scores.index):
            mlp_pairs += 1
            if model_scores["cnn_bilstm_tabular"] > model_scores["mlp_static"]:
                cnn_better += 1
    lines.append(
        f"- CNN-BiLSTM tabular had higher validation Macro-F1 than MLP in {cnn_better}/{mlp_pairs} comparable pairs."
    )

    hybrid_better = 0
    hybrid_pairs = 0
    for (dataset, scenario), group in family_best.groupby(["dataset", "scenario"]):
        model_scores = group.set_index("model_name")["f1_macro"]
        if {"cnn_bilstm_tabular", "hybrid_cnn_bilstm"}.issubset(model_scores.index):
            hybrid_pairs += 1
            if model_scores["hybrid_cnn_bilstm"] > model_scores["cnn_bilstm_tabular"]:
                hybrid_better += 1
    lines.append(
        f"- Hybrid CNN-BiLSTM had higher validation Macro-F1 than tabular CNN-BiLSTM in {hybrid_better}/{hybrid_pairs} mid/late pairs."
    )
    return lines


def analyze_loss_weight(results: pd.DataFrame) -> list[str]:
    test_rows = results[results["split"] == "test"].copy()
    deltas: list[float] = []
    positive = 0
    total = 0
    group_columns = ["dataset", "scenario", "model_name", "imbalance_strategy"]
    if "imbalance_effective_strategy" in test_rows.columns:
        group_columns.append("imbalance_effective_strategy")
    for _, group in test_rows.groupby(group_columns):
        by_weight = group.set_index("loss_weight")
        if {"none", "balanced"}.issubset(by_weight.index):
            balanced_recall = float(by_weight.loc["balanced", "recall_weak"])
            none_recall = float(by_weight.loc["none", "recall_weak"])
            delta = balanced_recall - none_recall
            deltas.append(delta)
            positive += int(delta > 0)
            total += 1
    if not deltas:
        return ["- Could not compare loss_weight strategies because paired none/balanced rows were not available."]
    return [
        f"- Balanced loss improved test weak recall in {positive}/{total} paired model runs; average delta was {sum(deltas) / len(deltas):+.4f}."
    ]


def analyze_overfitting(deep_summary: pd.DataFrame) -> list[str]:
    overfit = deep_summary[(deep_summary["train_f1_macro"] - deep_summary["val_f1_macro"]) > 0.20]
    if overfit.empty:
        return ["- Selected deep models do not show train-vs-validation Macro-F1 gaps above 0.20."]
    items = [
        f"{row.dataset}/{row.scenario}/{row.best_deep_model}/{row.loss_weight} "
        f"(train {row.train_f1_macro:.4f}, val {row.val_f1_macro:.4f})"
        for row in overfit.itertuples()
    ]
    return ["- Possible overfitting in selected deep models: " + "; ".join(items) + "."]


def analyze_scenarios(deep_summary: pd.DataFrame) -> list[str]:
    scenario_scores = deep_summary.groupby("scenario")["test_f1_macro"].mean().reindex(SCENARIO_ORDER)
    best = scenario_scores.idxmax()
    worst = scenario_scores.idxmin()
    return [
        "- Average deep test Macro-F1 by scenario: "
        + " -> ".join(f"{scenario}={score:.4f}" for scenario, score in scenario_scores.items())
        + ".",
        f"- Best scenario for deep learning is {best}; hardest is {worst}.",
    ]


def write_report(
    deep_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    results: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = ["## Deep Learning Classification Summary Report", ""]

    lines.append("### 1. Best deep models by validation Macro-F1")
    lines.extend(
        markdown_table(
            [
                "Dataset",
                "Scenario",
                "Best deep model",
                "Loss weight",
                "Val Macro-F1",
                "Test Macro-F1",
                "Test Accuracy",
                "Test Recall weak",
                "Test F1 weak",
                "Best epoch",
            ],
            [
                [
                    row.dataset,
                    row.scenario,
                    row.best_deep_model,
                    row.loss_weight,
                    fmt(row.val_f1_macro),
                    fmt(row.test_f1_macro),
                    fmt(row.test_accuracy),
                    fmt(row.test_recall_weak),
                    fmt(row.test_f1_weak),
                    str(int(row.best_epoch)),
                ]
                for row in deep_summary.itertuples()
            ],
        )
    )
    lines.append("")

    lines.append("### 2. Comparison with tabular baselines")
    lines.extend(
        markdown_table(
            [
                "Dataset",
                "Scenario",
                "Baseline Test Macro-F1",
                "Imbalance Test Macro-F1",
                "Deep Test Macro-F1",
                "Deep vs Baseline",
                "Deep vs Imbalance",
                "Baseline Recall weak",
                "Imbalance Recall weak",
                "Deep Recall weak",
            ],
            [
                [
                    row.dataset,
                    row.scenario,
                    fmt(row.baseline_test_f1_macro),
                    fmt(row.imbalance_test_f1_macro),
                    fmt(row.deep_test_f1_macro),
                    fmt(row.delta_deep_vs_baseline),
                    fmt(row.delta_deep_vs_imbalance),
                    fmt(row.baseline_test_recall_weak),
                    fmt(row.imbalance_test_recall_weak),
                    fmt(row.deep_test_recall_weak),
                ]
                for row in comparison.itertuples()
            ],
        )
    )
    lines.append("")

    lines.append("### 3. Automatic observations")
    observations: list[str] = []
    observations.extend(analyze_deep_vs_baselines(comparison))
    observations.extend(analyze_architectures(results))
    observations.extend(analyze_loss_weight(results))
    observations.extend(analyze_overfitting(deep_summary))
    observations.extend(analyze_scenarios(deep_summary))
    lines.extend(observations)
    lines.append("")

    lines.append("### 4. Technical limitations")
    lines.append("- CNN-BiLSTM tabular uses one-hot encoded feature vectors as pseudo-sequences, not true semester time series.")
    lines.append("- Hybrid CNN-BiLSTM only uses G1/G2 as the grade sequence, so the sequence is very short; mid has seq_len=1.")
    lines.append("- With small UCI datasets, deep learning can underperform tree-based tabular baselines.")
    lines.append("- Results should be checked with multiple seeds or richer multi-semester student data.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    results_path = PROJECT_ROOT / "reports" / "results" / "deep_classification_results.csv"
    tables_dir = PROJECT_ROOT / "reports" / "tables"
    results = read_required_csv(results_path, "deep classification results")

    deep_summary = build_deep_summary(results)
    comparison = build_comparison(deep_summary)

    tables_dir.mkdir(parents=True, exist_ok=True)
    deep_summary.to_csv(tables_dir / "deep_classification_summary.csv", index=False)
    comparison.to_csv(tables_dir / "deep_vs_tabular_comparison.csv", index=False)
    write_report(
        deep_summary,
        comparison,
        results,
        tables_dir / "deep_classification_summary_report.txt",
    )

    print(f"Deep summary saved to: {tables_dir / 'deep_classification_summary.csv'}")
    print(f"Comparison saved to: {tables_dir / 'deep_vs_tabular_comparison.csv'}")
    print(f"Text report saved to: {tables_dir / 'deep_classification_summary_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
