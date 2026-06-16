from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.v28 import SEED_ENSEMBLE_11, SEED_ENSEMBLE_5
from src.experiments.v30 import V30RunConfig, run_v30_experiments
from src.utils import set_seed, setup_logger

logger = setup_logger("run_v30_experiments")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run focused V30 experiments for student-mat late and xAPI only.")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ensemble-11", action="store_true", default=True)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.cv_folds = 2
        args.epochs = 3
        args.patience = 2
        args.ensemble_11 = False
    config = V30RunConfig(
        seed=args.seed,
        cv_folds=args.cv_folds,
        max_epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        ensemble_seeds=SEED_ENSEMBLE_11 if args.ensemble_11 else SEED_ENSEMBLE_5,
    )
    set_seed(args.seed)
    logger.info("Starting focused V30 experiments.")
    run_v30_experiments(config)
    logger.info("V30 experiments complete.")


if __name__ == "__main__":
    main()
