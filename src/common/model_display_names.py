from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING_PATH = ROOT / "configs" / "model_display_names.yaml"


@lru_cache(maxsize=None)
def load_model_display_names(path: str | Path = DEFAULT_MAPPING_PATH) -> dict[str, dict[str, str]]:
    """Load the presentation mapping without modifying scientific candidate IDs."""

    mapping_path = Path(path)
    payload = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Model display-name mapping must be an object: {mapping_path}")

    normalized: dict[str, dict[str, str]] = {}
    for candidate_id, metadata in payload.items():
        if not isinstance(candidate_id, str) or not isinstance(metadata, dict):
            raise ValueError(f"Invalid model display-name entry: {candidate_id!r}")
        display_name = metadata.get("display_name")
        category = metadata.get("category")
        if not isinstance(display_name, str) or not display_name.strip():
            raise ValueError(f"Missing display_name for {candidate_id}")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"Missing category for {candidate_id}")
        normalized[candidate_id] = {str(key): str(value) for key, value in metadata.items()}
    return normalized


def get_model_metadata(candidate_id: str, path: str | Path = DEFAULT_MAPPING_PATH) -> dict[str, str]:
    """Return display metadata, falling back safely to the unchanged internal ID."""

    metadata = load_model_display_names(path).get(candidate_id)
    if metadata is None:
        return {"display_name": candidate_id, "category": "Unmapped"}
    return dict(metadata)


def get_display_name(candidate_id: str, path: str | Path = DEFAULT_MAPPING_PATH) -> str:
    return get_model_metadata(candidate_id, path)["display_name"]


def add_display_name(
    record: Mapping[str, Any],
    *,
    candidate_key: str = "candidate_id",
    path: str | Path = DEFAULT_MAPPING_PATH,
) -> dict[str, Any]:
    """Add presentation fields while preserving the original candidate identifier."""

    result = dict(record)
    candidate_id = str(result[candidate_key])
    metadata = get_model_metadata(candidate_id, path)
    result["display_name"] = metadata["display_name"]
    result["model_category"] = metadata["category"]
    return result
