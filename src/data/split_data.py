import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import load_config, make_path


def split_dataset(
    prepared_data: pd.DataFrame,
    test_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify_labels = prepared_data["target"]

    return train_test_split(
        prepared_data,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify_labels,
    )


def save_train_test_split(
    dataset_name: str,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    config_path: str = "config.yaml",
) -> None:
    config = load_config(config_path)
    dataset_config = config["datasets"][dataset_name]

    train_path = make_path(dataset_config["train_path"])
    test_path = make_path(dataset_config["test_path"])
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)


def load_train_test_split(
    dataset_name: str,
    config_path: str = "config.yaml",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_config(config_path)
    dataset_config = config["datasets"][dataset_name]

    train_path = make_path(dataset_config["train_path"])
    test_path = make_path(dataset_config["test_path"])

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Train/test files for '{dataset_name}' are missing. Run scripts/run_prepare.py first."
        )

    return pd.read_csv(train_path), pd.read_csv(test_path)

