from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.v29 import REQUIRED_TASKS, V29RunConfig, run_v29_experiments
from src.experiments.v28 import SEED_ENSEMBLE_5
from src.utils import set_seed, setup_logger

logger = setup_logger("run_v29_experiments")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled V29 CNN-BiLSTM ablation with OOF threshold tuning.")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = list(REQUIRED_TASKS)
    if args.quick:
        tasks = tasks[:1]
        args.cv_folds = 2
        args.epochs = 3
        args.patience = 2
    config = V29RunConfig(
        seed=args.seed,
        cv_folds=args.cv_folds,
        max_epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        ensemble_seeds=SEED_ENSEMBLE_5,
    )
    set_seed(args.seed)
    logger.info("Starting V29 experiments for tasks=%s", tasks)
    run_v29_experiments(tasks, config)
    logger.info("V29 experiments complete.")


if __name__ == "__main__":
    main()
