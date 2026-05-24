from pathlib import Path

import pandas as pd


def create_parent_folder(file_path: str | Path) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def save_results(results: list[dict], file_path: str | Path) -> None:
    create_parent_folder(file_path)
    pd.DataFrame(results).to_csv(file_path, index=False)


def append_results(results: list[dict], file_path: str | Path) -> None:
    create_parent_folder(file_path)
    new_data = pd.DataFrame(results)

    if Path(file_path).exists():
        old_data = pd.read_csv(file_path)
        new_data = pd.concat([old_data, new_data], ignore_index=True)

    new_data.to_csv(file_path, index=False)

