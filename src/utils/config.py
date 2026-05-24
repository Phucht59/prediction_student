from pathlib import Path

import yaml


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = get_project_root() / path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def make_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return get_project_root() / path

