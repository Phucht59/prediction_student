from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STUDENT_DATASETS = {"all", "student", "student-mat", "student-por", "student-combined"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single training entrypoint for all maintained project models."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "deep", "baseline"],
        default="all",
        help="all trains baselines and deep models; use deep or baseline to run only one group.",
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "student", "student-mat", "student-por", "student-combined", "xapi"],
        default="all",
    )
    parser.add_argument("--scenario", choices=["all", "mid", "late"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_module(module: str, args: list[str]) -> None:
    command = [sys.executable, "-m", module, *args]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def trains_student_dataset(dataset: str) -> bool:
    return dataset in STUDENT_DATASETS


def trains_xapi_dataset(dataset: str) -> bool:
    return dataset in {"all", "xapi"}


def student_dataset_arg(dataset: str) -> str:
    return "all" if dataset in {"all", "student"} else dataset


def train_student_baselines(args: argparse.Namespace) -> None:
    dataset = student_dataset_arg(args.dataset)
    common = ["--dataset", dataset, "--scenario", args.scenario, "--seed", str(args.seed)]
    run_module("src.train.train_baselines", [*common, "--task", "all"])
    run_module("src.train.train_imbalance_baselines", common)


def train_xapi_baselines(args: argparse.Namespace) -> None:
    run_module("src.train.train_xapi_baselines", ["--seed", str(args.seed)])


def train_student_deep(args: argparse.Namespace) -> None:
    dataset = student_dataset_arg(args.dataset)
    common = [
        "--dataset",
        dataset,
        "--scenario",
        args.scenario,
        "--model",
        "clsv2",
        "--feature-selection",
        "pearson_chi2",
        "--max-features",
        "40",
        "--epochs",
        "80",
        "--patience",
        "12",
        "--batch-size",
        "32",
        "--lr",
        "0.001",
        "--seed",
        str(args.seed),
    ]
    for strategy, loss_weight in (
        ("none", "none"),
        ("smote", "none"),
        ("adasyn", "none"),
        ("none", "balanced"),
    ):
        run_module(
            "src.train.train_deep",
            [*common, "--oversampling", strategy, "--loss-weight", loss_weight],
        )
    run_module(
        "src.train.train_xapi_deep",
        [
            "--model",
            "cnn_bilstm_xapi",
            "--model-preset",
            "paper_deep",
            "--imbalance-strategy",
            "all",
            "--loss-weight",
            "all",
            "--scheduler",
            "plateau",
            "--label-smoothing",
            "0.05",
            "--epochs",
            "150",
            "--patience",
            "20",
            "--seed",
            str(args.seed),
        ],
    )


def train_xapi_deep(args: argparse.Namespace) -> None:
    common = [
        "--dataset",
        "xapi",
        "--model",
        "cnn_bilstm_xapi",
        "--feature-selection",
        "pearson_chi2",
        "--max-features",
        "56",
        "--epochs",
        "100",
        "--patience",
        "15",
        "--batch-size",
        "16",
        "--lr",
        "0.001",
        "--seed",
        str(args.seed),
        "--fusion",
        "concat",
    ]
    for strategy, loss_weight in (
        ("none", "none"),
        ("smote", "none"),
        ("adasyn", "none"),
        ("none", "balanced"),
    ):
        run_module(
            "src.train.train_deep",
            [*common, "--oversampling", strategy, "--loss-weight", loss_weight],
        )


def main() -> None:
    args = parse_args()

    if args.mode in {"all", "baseline"}:
        if trains_student_dataset(args.dataset):
            train_student_baselines(args)
        if trains_xapi_dataset(args.dataset):
            train_xapi_baselines(args)

    if args.mode in {"all", "deep"}:
        if trains_student_dataset(args.dataset):
            train_student_deep(args)
        if trains_xapi_dataset(args.dataset):
            train_xapi_deep(args)

    print("Training finished.", flush=True)


if __name__ == "__main__":
    main()
