from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.v28 import (
    OPTIONAL_TASKS,
    REQUIRED_TASKS,
    SEED_ENSEMBLE_11,
    SEED_ENSEMBLE_5,
    V28RunConfig,
    run_v28_experiments,
)
from src.utils import set_seed, setup_logger

logger = setup_logger("run_v28_experiments")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real V28 CNN-BiLSTM experiments with OOF threshold tuning.")
    parser.add_argument("--cv-folds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--ensemble-11", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = list(REQUIRED_TASKS)
    if args.include_optional:
        tasks.extend(OPTIONAL_TASKS)
    if args.quick:
        tasks = tasks[:1]
        args.cv_folds = 2
        args.epochs = 3
    seeds = SEED_ENSEMBLE_11 if args.ensemble_11 else SEED_ENSEMBLE_5
    config = V28RunConfig(
        seed=args.seed,
        cv_folds=args.cv_folds,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        ensemble_seeds=tuple(seeds),
        use_optional_tasks=args.include_optional,
    )
    set_seed(args.seed)
    logger.info("Starting V28 experiments for tasks=%s", tasks)
    run_v28_experiments(tasks, config)
    logger.info("V28 experiments complete.")


if __name__ == "__main__":
    main()

