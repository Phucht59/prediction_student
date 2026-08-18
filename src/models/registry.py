"""Read-only access to the public model registry."""

from src.release.catalog import OFFICIAL_MODELS, RECOMMENDATION_SYSTEM


def official_registry() -> dict[str, dict[str, object]]:
    return {
        metadata["model_id"]: metadata for metadata in OFFICIAL_MODELS.values()
    }
