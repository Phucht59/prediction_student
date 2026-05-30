from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified deep-learning training wrapper.")
    parser.add_argument("--dataset", choices=["all", "student-mat", "student-por", "student-combined", "xapi"], required=True)
    parser.add_argument("--scenario", choices=["all", "mid", "late"], default="all")
    parser.add_argument("--model", default="auto")
    parser.add_argument(
        "--oversampling",
        "--imbalance-strategy",
        dest="oversampling",
        default="auto",
        help=(
            "auto uses SMOTE for student datasets and ADASYN for xAPI; explicit values include "
            "none, smote, adasyn, borderline_smote, class_weight_balanced."
        ),
    )
    parser.add_argument("--loss-weight", default="none")
    parser.add_argument("--feature-selection", default="pearson_chi2")
    parser.add_argument("--max-features", type=int, default=56)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--fusion", choices=["concat", "gated"], default="concat")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--split-mode", choices=["processed", "holdout80"], default="processed")
    parser.add_argument("--early-stopping", choices=["val_f1", "none"], default="val_f1")
    parser.add_argument("--model-preset", choices=["default", "simple"], default="default")
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--swa", action="store_true")
    parser.add_argument("--cv", type=int, default=0)
    return parser.parse_args()


def resolve_imbalance_strategy(dataset: str, requested: str) -> str:
    """Resolve the final project default imbalance protocol."""
    if requested == "auto":
        return "adasyn" if dataset == "xapi" else "smote"
    return requested


def resolve_imbalance_and_loss(dataset: str, requested_strategy: str, requested_loss: str) -> tuple[str, str]:
    if requested_strategy == "class_weight_balanced":
        return "none", "balanced"
    return resolve_imbalance_strategy(dataset, requested_strategy), requested_loss


def run_module(module: str, extra_args: list[str]) -> int:
    command = [sys.executable, "-m", module, *extra_args]
    print("Running:", " ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    return int(completed.returncode)


def main() -> int:
    args = parse_args()
    if args.dataset == "xapi":
        oversampling, loss_weight = resolve_imbalance_and_loss("xapi", args.oversampling, args.loss_weight)
        if args.cv and args.cv > 1:
            return run_module(
                "src.train.train_xapi_cv",
                [
                    "--folds",
                    str(args.cv),
                    "--model",
                    "cnn_bilstm_xapi" if args.model == "auto" else args.model,
                    "--imbalance-strategy",
                    oversampling,
                    "--feature-selection",
                    args.feature_selection,
                    "--max-features",
                    str(args.max_features),
                    "--seed",
                    str(args.seed),
                    "--epochs",
                    str(args.epochs),
                    "--patience",
                    str(args.patience),
                    "--batch-size",
                    str(args.batch_size),
                    "--lr",
                    str(args.lr),
                    "--fusion",
                    args.fusion,
                    "--label-smoothing",
                    str(args.label_smoothing),
                    "--scheduler",
                    args.scheduler,
                    "--loss-weight",
                    loss_weight,
                    *(["--swa"] if args.swa else []),
                ],
            )
        model = "cnn_bilstm_xapi" if args.model == "auto" else args.model
        return run_module(
            "src.train.train_xapi_deep",
            [
                "--model",
                model,
                "--imbalance-strategy",
                oversampling,
                "--loss-weight",
                loss_weight,
                "--feature-selection",
                args.feature_selection,
                "--max-features",
                str(args.max_features),
                "--seed",
                str(args.seed),
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--fusion",
                args.fusion,
                "--label-smoothing",
                str(args.label_smoothing),
                "--scheduler",
                args.scheduler,
                "--split-mode",
                args.split_mode,
                "--early-stopping",
                args.early_stopping,
                "--model-preset",
                args.model_preset,
                *([] if args.weight_decay is None else ["--weight-decay", str(args.weight_decay)]),
                *([] if args.dropout is None else ["--dropout", str(args.dropout)]),
                *(["--swa"] if args.swa else []),
            ],
        )

    model = "auto" if args.model in {"auto", "cnn_bilstm_xapi", "cls_xapi"} else args.model
    datasets = ["student-mat", "student-por", "student-combined"] if args.dataset == "all" else [args.dataset]
    exit_code = 0
    for dataset in datasets:
        oversampling, loss_weight = resolve_imbalance_and_loss(dataset, args.oversampling, args.loss_weight)
        exit_code = max(
            exit_code,
            run_module(
                "src.train.train_deep_classification",
                [
                    "--dataset",
                    dataset,
                    "--scenario",
                    args.scenario,
                    "--model",
                    model,
                    "--imbalance-strategy",
                    oversampling,
                    "--loss-weight",
                    loss_weight,
                    "--feature-selection",
                    args.feature_selection,
                    "--max-features",
                    str(args.max_features),
                    "--seed",
                    str(args.seed),
                    "--epochs",
                    str(args.epochs),
                    "--patience",
                    str(args.patience),
                    "--batch-size",
                    str(args.batch_size),
                    "--lr",
                    str(args.lr),
                    "--mixup-alpha",
                    str(args.mixup_alpha),
                ],
            ),
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
