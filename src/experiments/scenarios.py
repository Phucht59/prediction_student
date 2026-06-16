from __future__ import annotations

import pandas as pd

from src.experiments.baselines import BaselineRunConfig, run_baseline_suite
from src.experiments.common import (
    ExperimentConfig,
    SCENARIOS,
    STUDENT_DATASETS,
    ensure_technical_report_dirs,
    save_json,
    summarize_cv,
    write_config,
)
from src.experiments.imbalance import DeepRunConfig, run_imbalance_suite
from src.utils import setup_logger

logger = setup_logger("scenario_experiments")


def run_scenario_suite(
    datasets: list[str] | None = None,
    scenarios: list[str] | None = None,
    baseline_strategies: list[str] | None = None,
    deep_strategies: list[str] | None = None,
    config: ExperimentConfig | None = None,
    baseline_config: BaselineRunConfig | None = None,
    deep_config: DeepRunConfig | None = None,
    include_baselines: bool = True,
    include_deep: bool = True,
) -> dict[str, pd.DataFrame]:
    config = config or ExperimentConfig()
    datasets = datasets or list(STUDENT_DATASETS)
    scenarios = scenarios or list(SCENARIOS)
    baseline_strategies = baseline_strategies or ["none", "class_weight", "smotenc", "random_oversampling"]
    deep_strategies = deep_strategies or ["none", "class_weight", "smotenc", "random_oversampling", "focal_loss", "smotenc_focal_loss"]

    all_baseline_cv = []
    all_baseline_locked = []
    all_deep_cv = []
    all_deep_locked = []
    for dataset_name in datasets:
        if dataset_name not in STUDENT_DATASETS:
            raise ValueError("Scenario suite only supports student-mat and student-por; student-combine is excluded.")
        for scenario in scenarios:
            logger.info("Scenario suite: dataset=%s scenario=%s", dataset_name, scenario)
            if include_baselines:
                cv_df, locked_df = run_baseline_suite(
                    dataset_name,
                    scenario,
                    baseline_strategies,
                    config,
                    baseline_config,
                )
                all_baseline_cv.append(cv_df)
                all_baseline_locked.append(locked_df)
            if include_deep:
                cv_df, locked_df = run_imbalance_suite(
                    dataset_name,
                    scenario,
                    deep_strategies,
                    config,
                    deep_config,
                )
                all_deep_cv.append(cv_df)
                all_deep_locked.append(locked_df)

    dirs = ensure_technical_report_dirs()
    outputs: dict[str, pd.DataFrame] = {}
    if all_baseline_cv:
        outputs["baseline_cv"] = pd.concat(all_baseline_cv, ignore_index=True)
        outputs["baseline_locked"] = pd.concat(all_baseline_locked, ignore_index=True)
        outputs["baseline_cv"].to_csv(dirs["scenarios"] / "baseline_cv_all.csv", index=False)
        outputs["baseline_locked"].to_csv(dirs["scenarios"] / "baseline_locked_test_all.csv", index=False)
    if all_deep_cv:
        outputs["deep_cv"] = pd.concat(all_deep_cv, ignore_index=True)
        outputs["deep_locked"] = pd.concat(all_deep_locked, ignore_index=True)
        outputs["deep_cv"].to_csv(dirs["scenarios"] / "cnn_bilstm_cv_all.csv", index=False)
        outputs["deep_locked"].to_csv(dirs["scenarios"] / "cnn_bilstm_locked_test_all.csv", index=False)

    summary = {}
    for name, frame in outputs.items():
        if name.endswith("_cv"):
            group_keys = [key for key in ["dataset", "scenario", "strategy", "model"] if key in frame.columns]
            summary[name] = [
                {**dict(zip(group_keys, keys if isinstance(keys, tuple) else (keys,))), **summarize_cv(group.to_dict("records"))}
                for keys, group in frame.groupby(group_keys)
            ]
        else:
            summary[name] = frame.to_dict("records")
    save_json(dirs["scenarios"] / "technical_scenario_summary.json", summary)
    write_config(
        dirs["scenarios"] / "technical_scenario_config.json",
        config,
        {
            "datasets": datasets,
            "scenarios": scenarios,
            "baseline_strategies": baseline_strategies,
            "deep_strategies": deep_strategies,
            "include_baselines": include_baselines,
            "include_deep": include_deep,
        },
    )
    return outputs

