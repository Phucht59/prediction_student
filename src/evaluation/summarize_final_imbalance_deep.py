from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


OUTPUT_PATH = PROJECT_ROOT / "reports" / "tables" / "final_imbalance_deep_comparison.csv"


def read_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path)
    return data if not data.empty else pd.DataFrame()


def _last_by_key(data: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if data.empty:
        return data
    return data.drop_duplicates(subset=keys, keep="last")


def student_rows() -> pd.DataFrame:
    path = PROJECT_ROOT / "reports" / "results" / "deep_classification_results.csv"
    data = read_optional(path)
    if data.empty:
        return data
    if "scenario" in data.columns:
        data = data[data["scenario"].ne("early")].copy()
    standard_imbalance = data["loss_weight"].eq("none") & data["imbalance_strategy"].isin(["none", "smote", "adasyn"])
    class_weight = data["loss_weight"].eq("balanced") & data["imbalance_strategy"].eq("none")
    filtered = data[
        data["dataset"].isin(["student-mat", "student-por", "student-combined"])
        & data["scenario"].eq("late")
        & data["model_name"].eq("clsv2")
        & data["split"].eq("test")
        & (standard_imbalance | class_weight)
    ].copy()
    if filtered.empty:
        return filtered
    if "mixup_alpha" in filtered.columns:
        filtered = filtered[filtered["mixup_alpha"].fillna(0.0).astype(float).eq(0.0)].copy()
        if filtered.empty:
            return filtered
    filtered["source_file"] = str(path.relative_to(PROJECT_ROOT))
    filtered["scenario"] = filtered["scenario"].fillna("late")
    return _last_by_key(
        filtered,
        ["dataset", "scenario", "model_name", "loss_weight", "imbalance_strategy", "seed"],
    )


def xapi_rows() -> pd.DataFrame:
    path = PROJECT_ROOT / "reports" / "results" / "xapi_deep_results.csv"
    data = read_optional(path)
    if data.empty:
        return data
    if "scenario" in data.columns:
        data = data[data["scenario"].ne("early")].copy()
    standard_imbalance = data["loss_weight"].eq("none") & data["imbalance_strategy"].isin(["none", "smote", "adasyn"])
    class_weight = data["loss_weight"].eq("balanced") & data["imbalance_strategy"].eq("none")
    filtered = data[
        data["dataset"].eq("xapi")
        & data["scenario"].eq("xapi_behavior")
        & data["model_name"].eq("cnn_bilstm_xapi")
        & data["split"].eq("test")
        & (standard_imbalance | class_weight)
        & data.get("split_mode", pd.Series("processed", index=data.index)).fillna("processed").eq("processed")
        & data.get("model_preset", pd.Series("default", index=data.index)).fillna("default").eq("default")
    ].copy()
    if filtered.empty:
        return filtered
    for column, default in (("split_mode", "processed"), ("model_preset", "default")):
        if column not in filtered.columns:
            filtered[column] = default
        else:
            filtered[column] = filtered[column].fillna(default)
    filtered["source_file"] = str(path.relative_to(PROJECT_ROOT))
    return _last_by_key(
        filtered,
        ["dataset", "scenario", "model_name", "loss_weight", "imbalance_strategy", "seed", "split_mode", "model_preset"],
    )


def build_summary() -> pd.DataFrame:
    rows = pd.concat([student_rows(), xapi_rows()], ignore_index=True, sort=False)
    if rows.empty:
        return rows
    for column in ["imbalance_effective_strategy", "variant_name", "pr_auc_macro", "pr_auc_weighted", "pr_auc_weak"]:
        if column not in rows.columns:
            rows[column] = ""
    rows["comparison_strategy"] = rows["imbalance_strategy"]
    class_weight_mask = rows["loss_weight"].eq("balanced") & rows["imbalance_strategy"].eq("none")
    rows.loc[class_weight_mask, "comparison_strategy"] = "class_weight_balanced"
    selected = rows[
        [
            "dataset",
            "scenario",
            "model_name",
            "loss_weight",
            "imbalance_strategy",
            "comparison_strategy",
            "imbalance_effective_strategy",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "pr_auc_macro",
            "recall_weak",
            "pr_auc_weak",
            "f1_weak",
            "seed",
            "variant_name",
            "source_file",
        ]
    ].copy()
    selected["rank_within_dataset"] = selected.groupby(["dataset", "scenario"])["f1_macro"].rank(
        method="dense",
        ascending=False,
    )
    return selected.sort_values(["dataset", "scenario", "rank_within_dataset", "comparison_strategy"])


def main() -> int:
    summary = build_summary()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if summary.empty and OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 2:
        print(
            "No raw deep result CSVs found; keeping existing final comparison table: "
            f"{OUTPUT_PATH}"
        )
        return 0
    summary.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved final imbalance comparison: {OUTPUT_PATH}")
    if not summary.empty:
        display_columns = [
            "dataset",
            "scenario",
            "model_name",
            "comparison_strategy",
            "accuracy",
            "f1_macro",
            "pr_auc_macro",
            "recall_weak",
            "rank_within_dataset",
        ]
        print(summary[display_columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
