from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.common import ExperimentConfig, SCENARIOS, STUDENT_DATASETS, ensure_technical_report_dirs
from src.experiments.deep_debug import DebugRunConfig, run_deep_debug_suite
from src.utils import set_seed, setup_logger

logger = setup_logger("debug_deep_model")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug PyTorch deep model branches before final experiments.")
    parser.add_argument("--datasets", nargs="+", choices=STUDENT_DATASETS, default=list(STUDENT_DATASETS))
    parser.add_argument("--scenarios", nargs="+", choices=sorted(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def _markdown_table(frame: pd.DataFrame, cols: list[str], limit: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    out = frame[[col for col in cols if col in frame.columns]].head(limit).copy()
    for col in out.select_dtypes(include=["float"]).columns:
        out[col] = out[col].map(lambda x: f"{x:.4f}")
    header = "| " + " | ".join(out.columns) + " |"
    separator = "| " + " | ".join("---" for _ in out.columns) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in out.to_numpy()]
    return "\n".join([header, separator, *rows])


def write_debug_report(outputs: dict[str, pd.DataFrame], quick: bool) -> Path:
    dirs = ensure_technical_report_dirs()
    report_path = dirs["ablation"] / "deep_debug_summary.md"
    overfit = outputs["overfit"]
    cv = outputs["deep_ablation_cv"]
    locked = outputs["deep_ablation_locked_test"]
    threshold_tuning = outputs["low_class_threshold_tuning"]
    baseline_vs_deep = outputs["baseline_vs_deep_same_scenario"]
    evaluated_cv = cv[cv["status"] == "evaluated"] if "status" in cv else cv
    cv_summary = (
        evaluated_cv.groupby(["dataset", "scenario", "variant", "config_id"], as_index=False)
        .agg(macro_f1=("macro_f1", "mean"), recall_low=("recall_low", "mean"), f1_low=("f1_low", "mean"), rmse=("rmse", "mean"), r2=("r2", "mean"))
        .sort_values("macro_f1", ascending=False)
        if not evaluated_cv.empty
        else pd.DataFrame()
    )
    lines = [
        "# Deep Model Debug Report",
        "",
        f"- Mode: {'quick' if quick else 'configured'}",
        "- Purpose: debug PyTorch deep branch before claiming CNN-BiLSTM as the main model.",
        "- Early scenario has no real sequence; only context MLP variants are evaluated.",
        "- Thresholds are tuned from OOF train-pool probabilities, never from locked test.",
        "- Baseline-vs-deep rows use CV-selected baselines and same locked test per dataset/scenario.",
        "- Main RMSE/R2 now use class-to-point mapping; regression-head metrics are separate columns.",
        "- Regression head should not be claimed while `regression_head_rmse` remains high.",
        "",
        "## Overfit Sanity",
        "",
        _markdown_table(overfit, ["dataset", "scenario", "variant", "config_id", "status", "macro_f1", "recall_low", "f1_low", "rmse", "r2", "regression_head_rmse"]),
        "",
        "## Branch Ablation CV",
        "",
        _markdown_table(cv_summary, ["dataset", "scenario", "variant", "config_id", "macro_f1", "recall_low", "f1_low", "rmse", "r2"]),
        "",
        "## Low-Class Threshold Tuning",
        "",
        _markdown_table(
            threshold_tuning.sort_values(["dataset", "scenario", "macro_f1"], ascending=[True, True, False]),
            ["dataset", "scenario", "variant", "config_id", "prediction_mode", "threshold_low", "macro_f1", "recall_low", "f1_low"],
        ),
        "",
        "## Branch Ablation Locked Test",
        "",
        _markdown_table(
            locked.sort_values("macro_f1", ascending=False) if "macro_f1" in locked else locked,
            ["dataset", "scenario", "variant", "config_id", "prediction_mode", "status", "macro_f1", "recall_low", "f1_low", "rmse", "r2", "regression_head_rmse"],
        ),
        "",
        "## Baseline Vs Deep",
        "",
        _markdown_table(
            baseline_vs_deep.sort_values("macro_f1_gap_deep_minus_baseline", ascending=False) if "macro_f1_gap_deep_minus_baseline" in baseline_vs_deep else baseline_vs_deep,
            [
                "dataset",
                "scenario",
                "baseline_model",
                "baseline_strategy",
                "baseline_locked_macro_f1",
                "deep_variant",
                "deep_config_id",
                "deep_prediction_mode",
                "deep_locked_macro_f1",
                "macro_f1_gap_deep_minus_baseline",
                "recall_low_gap_deep_minus_baseline",
            ],
        ),
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    (dirs["ablation"] / "deep_debug_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    args = _parse_args()
    if args.quick:
        args.datasets = args.datasets[:1]
        args.scenarios = args.scenarios[:2]
        args.cv_folds = 2
        args.epochs = 5
        args.sample_size = 48
    set_seed(args.seed)
    config = ExperimentConfig(seed=args.seed, cv_folds=args.cv_folds)
    run_config = DebugRunConfig(max_epochs=args.epochs, overfit_sample_size=args.sample_size)
    outputs = run_deep_debug_suite(args.datasets, args.scenarios, config, run_config)
    report_path = write_debug_report(outputs, args.quick)
    logger.info("Deep debug report written to %s", report_path)


if __name__ == "__main__":
    main()
