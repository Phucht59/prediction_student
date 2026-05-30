from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT_PATH = PROJECT_ROOT / "report_temp" / "xapi_experiment_report.md"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "xapi" / "xapi_behavior"


def read_csv_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    data = pd.read_csv(path)
    if "scenario" in data.columns:
        data = data[data["scenario"].ne("early")].copy()
    return None if data.empty else data


def read_required_csv(path: Path) -> pd.DataFrame:
    data = read_csv_optional(path)
    if data is None:
        raise FileNotFoundError(f"Missing or empty CSV: {path}")
    return data


def select_test_for_validation_best(
    data: pd.DataFrame,
    key_columns: list[str],
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    group_columns = group_columns or ["dataset", "scenario"]
    for group_key, group in data.groupby(group_columns, sort=False, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        val = group[group["split"].eq("val")]
        if val.empty:
            continue
        selected = val.loc[val["f1_macro"].idxmax()]
        test = group[group["split"].eq("test")].copy()
        for column in key_columns:
            test = test[test[column].eq(selected[column])]
        if test.empty:
            continue
        test_row = test.iloc[0]
        row = {
            "val_f1_macro": selected["f1_macro"],
            "val_accuracy": selected["accuracy"],
            "val_recall_weak": selected["recall_weak"],
            "test_f1_macro": test_row["f1_macro"],
            "test_accuracy": test_row["accuracy"],
            "test_recall_weak": test_row["recall_weak"],
            "test_f1_weak": test_row["f1_weak"],
        }
        for column, value in zip(group_columns, group_key):
            row[column] = value
        for column in key_columns:
            row[column] = selected[column]
        rows.append(row)
    return pd.DataFrame(rows)


def build_baseline_summary() -> pd.DataFrame:
    data = read_required_csv(PROJECT_ROOT / "reports" / "results" / "xapi_baseline_results.csv")
    summary = select_test_for_validation_best(data, ["model_name", "imbalance_strategy"])
    return summary[
        [
            "dataset",
            "scenario",
            "model_name",
            "imbalance_strategy",
            "val_f1_macro",
            "test_f1_macro",
            "test_accuracy",
            "test_recall_weak",
            "test_f1_weak",
        ]
    ]


def build_deep_summary() -> pd.DataFrame:
    data = read_required_csv(PROJECT_ROOT / "reports" / "results" / "xapi_deep_results.csv")
    for column, default in (
        ("imbalance_strategy", "none"),
        ("feature_selection", "none"),
        ("fusion", "concat"),
        ("label_smoothing", 0.0),
        ("scheduler", "none"),
        ("split_mode", "processed"),
        ("model_preset", "default"),
        ("early_stopping", "val_f1"),
    ):
        if column not in data.columns:
            data[column] = default
        else:
            data[column] = data[column].fillna(default)
    preferred = data[data["model_name"].eq("cnn_bilstm_xapi")].copy()
    if not preferred.empty:
        data = preferred
    default_preset = data[data["model_preset"].eq("default")].copy()
    if not default_preset.empty:
        data = default_preset
    summary = select_test_for_validation_best(
        data,
        ["model_name", "loss_weight", "imbalance_strategy", "feature_selection", "fusion", "label_smoothing", "scheduler"],
        group_columns=["dataset", "scenario", "split_mode", "model_preset", "early_stopping"],
    )
    return summary[
        [
            "dataset",
            "scenario",
            "split_mode",
            "model_preset",
            "early_stopping",
            "model_name",
            "loss_weight",
            "imbalance_strategy",
            "feature_selection",
            "fusion",
            "label_smoothing",
            "scheduler",
            "val_f1_macro",
            "test_f1_macro",
            "test_accuracy",
            "test_recall_weak",
            "test_f1_weak",
        ]
    ]


def mean_std(group: pd.DataFrame, column: str) -> tuple[float, float]:
    return float(group[column].mean()), float(group[column].std(ddof=1) if len(group) > 1 else 0.0)


def build_final_summary() -> pd.DataFrame:
    columns = [
        "dataset",
        "scenario",
        "model_name",
        "loss_weight",
        "imbalance_strategy",
        "feature_selection",
        "n_seeds",
        "accuracy_mean",
        "accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "recall_weak_mean",
        "recall_weak_std",
        "f1_weak_mean",
        "f1_weak_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
    ]
    data = read_csv_optional(PROJECT_ROOT / "reports" / "results" / "xapi_final_multiseed_results.csv")
    if data is None:
        return pd.DataFrame(columns=columns)
    for column, default in (
        ("imbalance_strategy", "none"),
        ("feature_selection", "none"),
    ):
        if column not in data.columns:
            data[column] = default
    test = data[data["split"].eq("test")].copy()
    rows: list[dict] = []
    for (dataset, scenario), group in test.groupby(["dataset", "scenario"], sort=False):
        first = group.iloc[0]
        row = {
            "dataset": dataset,
            "scenario": scenario,
            "model_name": first["model_name"],
            "loss_weight": first["loss_weight"],
            "imbalance_strategy": first["imbalance_strategy"],
            "feature_selection": first["feature_selection"],
            "n_seeds": int(group["seed"].nunique()),
        }
        for output_name, column in (
            ("accuracy", "accuracy"),
            ("macro_f1", "f1_macro"),
            ("recall_weak", "recall_weak"),
            ("f1_weak", "f1_weak"),
            ("balanced_accuracy", "balanced_accuracy"),
        ):
            mean, std = mean_std(group, column)
            row[f"{output_name}_mean"] = mean
            row[f"{output_name}_std"] = std
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def fmt(value) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def fmt_mean_std(mean, std) -> str:
    return f"{float(mean):.4f}+/-{float(std):.4f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    lines.extend(" | ".join(row) for row in rows)
    return lines


def load_metadata() -> dict:
    metadata_path = PROCESSED_DIR / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def write_report(baseline: pd.DataFrame, deep: pd.DataFrame, final: pd.DataFrame) -> None:
    metadata = load_metadata()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# xAPI Experiment Report", ""]
    lines.append("## Dataset info")
    lines.append(f"- Dataset: xAPI-Edu-Data")
    lines.append(f"- Rows: {metadata.get('n_rows_total', '')}")
    lines.append(f"- Scenario: xapi_behavior")
    lines.append(f"- Target: Class")
    lines.append("- Class mapping: L -> 0 = Low, M -> 1 = Middle, H -> 2 = High")
    lines.append(f"- Class distribution total: {metadata.get('class_distribution_total', {})}")
    lines.append("")

    lines.append("## Preprocessing")
    lines.append("- Split: train/validation/test = 70/15/15")
    lines.append("- Stratified by target_class")
    lines.append("- Numeric: SimpleImputer median + MinMaxScaler")
    lines.append("- Categorical: SimpleImputer most_frequent + OneHotEncoder(handle_unknown='ignore')")
    lines.append("- Preprocessor fit only on train, validation/test only transformed")
    lines.append(f"- Processed output: {PROCESSED_DIR}")
    lines.append(f"- Leakage checks: {metadata.get('leakage_checks', {})}")
    lines.append("")

    lines.append("## Baseline results")
    lines.extend(
        markdown_table(
            ["Dataset", "Scenario", "Model", "Strategy", "Val Macro-F1", "Test Macro-F1", "Test Accuracy", "Test Recall weak"],
            [
                [
                    row.dataset,
                    row.scenario,
                    row.model_name,
                    row.imbalance_strategy,
                    fmt(row.val_f1_macro),
                    fmt(row.test_f1_macro),
                    fmt(row.test_accuracy),
                    fmt(row.test_recall_weak),
                ]
                for row in baseline.itertuples()
            ],
        )
    )
    lines.append("")

    lines.append("## Deep results")
    lines.extend(
        markdown_table(
            [
                "Dataset",
                "Scenario",
                "Split mode",
                "Preset",
                "Early stopping",
                "Model",
                "Loss weight",
                "Imbalance",
                "Feature selection",
                "Val Macro-F1",
                "Test Macro-F1",
                "Test Accuracy",
                "Test Recall weak",
            ],
            [
                [
                    row.dataset,
                    row.scenario,
                    getattr(row, "split_mode", "processed"),
                    getattr(row, "model_preset", "default"),
                    getattr(row, "early_stopping", "val_f1"),
                    row.model_name,
                    row.loss_weight,
                    row.imbalance_strategy,
                    row.feature_selection,
                    fmt(row.val_f1_macro),
                    fmt(row.test_f1_macro),
                    fmt(row.test_accuracy),
                    fmt(row.test_recall_weak),
                ]
                for row in deep.itertuples()
            ],
        )
    )
    lines.append("")

    if not final.empty:
        lines.append("## Final multi-seed results")
        lines.extend(
            markdown_table(
                [
                    "Dataset",
                    "Scenario",
                    "Model",
                    "Loss weight",
                    "Imbalance",
                    "Feature selection",
                    "Accuracy mean+/-std",
                    "Macro-F1 mean+/-std",
                    "Recall weak mean+/-std",
                    "F1 weak mean+/-std",
                ],
                [
                    [
                        row.dataset,
                        row.scenario,
                        row.model_name,
                        row.loss_weight,
                        row.imbalance_strategy,
                        row.feature_selection,
                        fmt_mean_std(row.accuracy_mean, row.accuracy_std),
                        fmt_mean_std(row.macro_f1_mean, row.macro_f1_std),
                        fmt_mean_std(row.recall_weak_mean, row.recall_weak_std),
                        fmt_mean_std(row.f1_weak_mean, row.f1_weak_std),
                    ]
                    for row in final.itertuples()
                ],
            )
        )
    else:
        lines.append("## Final multi-seed results")
        lines.append("- Not generated in the cleaned final pipeline. Run a dedicated multi-seed experiment if needed.")
    lines.append("")

    lines.append("## Nhan xet")
    deep_row = deep.iloc[0]
    lines.append(
        "- Best deep model by validation Macro-F1: "
        f"{deep_row['model_name']} with loss_weight={deep_row['loss_weight']}, "
        f"imbalance={deep_row['imbalance_strategy']}, feature_selection={deep_row['feature_selection']}."
    )
    if not final.empty:
        final_row = final.iloc[0]
        lines.append(f"- Final multi-seed Macro-F1 mean: {final_row['macro_f1_mean']:.4f}.")
    lines.append("- xAPI has no G1/G2/G3 grade-period setup, so it uses scenario xapi_behavior.")
    lines.append("- CNN-BiLSTM-XAPI uses all processed xAPI features as one input tensor.")
    lines.append("- xAPI is classification-only in this project; no G3 regression is created.")
    lines.append("")

    lines.append("## Han che")
    lines.append("- xAPI target Class is already categorical and not directly equivalent to G3-based labels.")
    lines.append("- CNN-BiLSTM-XAPI treats the processed feature vector as an ordered pseudo-sequence.")
    lines.append("- Traditional baselines remain comparison references; final thesis model strategy prioritizes CNN-BiLSTM-XAPI for xAPI.")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    baseline = build_baseline_summary()
    deep = build_deep_summary()
    final = build_final_summary()
    baseline.to_csv(TABLES_DIR / "xapi_baseline_summary.csv", index=False)
    deep.to_csv(TABLES_DIR / "xapi_deep_summary.csv", index=False)
    if not final.empty:
        final.to_csv(TABLES_DIR / "xapi_final_multiseed_summary.csv", index=False)
    write_report(baseline, deep, final)
    print(f"Saved: {TABLES_DIR / 'xapi_baseline_summary.csv'}")
    print(f"Saved: {TABLES_DIR / 'xapi_deep_summary.csv'}")
    if not final.empty:
        print(f"Saved: {TABLES_DIR / 'xapi_final_multiseed_summary.csv'}")
    print(f"Saved: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
