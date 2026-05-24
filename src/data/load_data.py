from pathlib import Path

import pandas as pd

from src.utils.config import load_config, make_path


def load_dataset(dataset_name: str, config_path: str = "config.yaml") -> pd.DataFrame:
    config = load_config(config_path)

    if dataset_name not in config["datasets"]:
        available_names = ", ".join(config["datasets"].keys())
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available datasets: {available_names}")

    dataset_path = make_path(config["datasets"][dataset_name]["path"])

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file is missing: {dataset_path}\n"
            f"Please place the file in the data/raw folder before running the pipeline."
        )

    return read_csv_file(dataset_path)


def read_csv_file(file_path: str | Path) -> pd.DataFrame:
    # sep=None lets pandas detect comma or semicolon separated files.
    return pd.read_csv(file_path, sep=None, engine="python")

