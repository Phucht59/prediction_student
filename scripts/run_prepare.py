from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.prepare_data import prepare_dataset
from src.data.split_data import save_train_test_split, split_dataset
from src.utils.config import load_config
from src.utils.set_seed import set_random_seed


def main() -> None:
    config = load_config()
    set_random_seed(config["random_seed"])

    for dataset_name in config["datasets"]:
        print(f"Preparing {dataset_name}...")
        prepared_data = prepare_dataset(dataset_name)
        train_data, test_data = split_dataset(
            prepared_data,
            config["test_size"],
            config["random_seed"],
        )
        save_train_test_split(dataset_name, train_data, test_data)

    print("Data preparation finished.")


if __name__ == "__main__":
    main()

