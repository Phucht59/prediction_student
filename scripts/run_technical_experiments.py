from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.baselines import BaselineRunConfig
from src.experiments.common import (
    ExperimentConfig,
    IMBALANCE_STRATEGIES,
    SCENARIOS,
    STUDENT_DATASETS,
    ensure_technical_report_dirs,
)
from src.experiments.imbalance import DeepRunConfig
from src.experiments.scenarios import run_scenario_suite
from src.utils import set_seed, setup_logger

logger = setup_logger("run_technical_experiments")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final technical Student experiments without ADASYN.")
    parser.add_argument("--datasets", nargs="+", choices=STUDENT_DATASETS, default=list(STUDENT_DATASETS))
    parser.add_argument("--scenarios", nargs="+", choices=sorted(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-strategies", nargs="+", default=["none", "class_weight", "smotenc", "random_oversampling"])
    parser.add_argument("--deep-strategies", nargs="+", default=list(IMBALANCE_STRATEGIES))
    parser.add_argument("--ensemble-seeds", nargs="+", type=int, default=[42, 123, 155])
    parser.add_argument("--deep-epochs", type=int, default=15)
    parser.add_argument("--baseline-max-iter", type=int, default=300)
    parser.add_argument("--baseline-estimators", type=int, default=200)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-deep", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Run a small end-to-end smoke experiment.")
    return parser.parse_args()


def _markdown_table(frame: pd.DataFrame, metric: str = "macro_f1", limit: int = 12) -> str:
    if frame.empty:
        return "_No rows._"
    cols = [
        col for col in [
            "dataset",
            "scenario",
            "strategy",
            "model",
            "prediction_mode",
            metric,
            "recall_low",
            "f1_low",
            "rmse",
            "r2",
            "regression_head_rmse",
        ] if col in frame.columns
    ]
    out = frame.sort_values(metric, ascending=False).head(limit)[cols].copy()
    for col in out.select_dtypes(include=["float"]).columns:
        out[col] = out[col].map(lambda x: f"{x:.4f}")
    header = "| " + " | ".join(out.columns) + " |"
    separator = "| " + " | ".join("---" for _ in out.columns) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in out.to_numpy()]
    return "\n".join([header, separator, *rows])


def _cv_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    group_keys = [key for key in ["dataset", "scenario", "strategy", "model"] if key in frame.columns]
    grouped = frame.groupby(group_keys, as_index=False).agg(
        macro_f1=("macro_f1", "mean"),
        recall_low=("recall_low", "mean"),
        f1_low=("f1_low", "mean"),
        rmse=("rmse", "mean"),
        r2=("r2", "mean"),
    )
    return grouped


def write_markdown_report(outputs: dict[str, pd.DataFrame], quick: bool) -> Path:
    dirs = ensure_technical_report_dirs()
    report_path = dirs["scenarios"].parent / "technical_experiment_report.md"
    baseline_cv = _cv_summary(outputs.get("baseline_cv", pd.DataFrame()))
    deep_cv = _cv_summary(outputs.get("deep_cv", pd.DataFrame()))
    baseline_locked = outputs.get("baseline_locked", pd.DataFrame())
    deep_locked = outputs.get("deep_locked", pd.DataFrame())
    lines = [
        "# Technical Experiment Summary",
        "",
        f"- Mode: {'quick smoke test' if quick else 'full configured run'}",
        "- Student datasets only: student-mat and/or student-por; student-combine is not used.",
        "- ADASYN is not used. Mixed categorical/numerical oversampling uses SMOTENC; random oversampling duplicates rows.",
        "- Locked test is evaluated after CV/OOF threshold tuning and is not used for model selection.",
        "- Required metrics are exported: Accuracy, Macro Precision, Macro Recall, Macro F1, Recall Low, F1 Low, RMSE, R2.",
        "- Main RMSE/R2 use a fixed class-to-point mapping; deep regression-head RMSE/R2 are exported separately when available.",
        "",
        "## Baseline CV Top Rows",
        "",
        _markdown_table(baseline_cv),
        "",
        "## CNN-BiLSTM CV Top Rows",
        "",
        _markdown_table(deep_cv),
        "",
        "## Baseline Locked Test Top Rows",
        "",
        _markdown_table(baseline_locked),
        "",
        "## CNN-BiLSTM Locked Test Top Rows",
        "",
        _markdown_table(deep_locked),
        "",
        "## Output Locations",
        "",
        "- `reports/final/scenarios/`",
        "- `reports/final/baselines/`",
        "- `reports/final/imbalance/`",
        "- `reports/final/ablation/`",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    args = _parse_args()
    if args.quick:
        args.cv_folds = min(args.cv_folds, 2)
        args.datasets = args.datasets[:1]
        args.scenarios = args.scenarios[:1]
        args.baseline_strategies = ["none"]
        args.deep_strategies = ["smotenc_focal_loss"]
        args.ensemble_seeds = [args.seed]
        args.deep_epochs = 2
        args.baseline_max_iter = 60
        args.baseline_estimators = 30

    invalid_deep = sorted(set(args.deep_strategies) - set(IMBALANCE_STRATEGIES))
    if invalid_deep:
        raise ValueError(f"Unsupported deep strategies: {invalid_deep}")

    set_seed(args.seed)
    config = ExperimentConfig(seed=args.seed, cv_folds=args.cv_folds)
    baseline_config = BaselineRunConfig(max_iter=args.baseline_max_iter, n_estimators=args.baseline_estimators)
    deep_config = DeepRunConfig(max_epochs=args.deep_epochs, ensemble_seeds=tuple(args.ensemble_seeds))
    logger.info("Starting technical experiments with config=%s", config)
    outputs = run_scenario_suite(
        datasets=args.datasets,
        scenarios=args.scenarios,
        baseline_strategies=args.baseline_strategies,
        deep_strategies=args.deep_strategies,
        config=config,
        baseline_config=baseline_config,
        deep_config=deep_config,
        include_baselines=not args.skip_baselines,
        include_deep=not args.skip_deep,
    )
    report_path = write_markdown_report(outputs, args.quick)
    logger.info("Technical report written to %s", report_path)


if __name__ == "__main__":
    main()
