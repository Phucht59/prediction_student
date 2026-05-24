from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.split_data import load_train_test_split
from src.train.train_deep import train_deep_model
from src.utils.config import load_config, make_path
from src.utils.save_file import save_results
from src.utils.set_seed import set_random_seed


def main() -> None:
    config = load_config()
    set_random_seed(config["random_seed"])
    results = []

    for dataset_name in config["datasets"]:
        train_data, test_data = load_train_test_split(dataset_name)

        for model_name in config["deep_models"]:
            print(f"Training {model_name} on {dataset_name}...")
            result = train_deep_model(
                dataset_name,
                model_name,
                train_data,
                test_data,
                config["deep_learning"],
            )
            results.append(result)

    save_results(results, make_path("results/metrics/deep_results.csv"))
    print("Deep learning training finished.")


if __name__ == "__main__":
    main()

