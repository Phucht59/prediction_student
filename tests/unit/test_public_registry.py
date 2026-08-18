from src.models.registry import official_registry


def test_public_registry_contains_three_models_and_recommendation() -> None:
    registry = official_registry()
    assert set(registry) == {
        "cnn_bilstm_mat",
        "cnn_bilstm_por",
        "cnn_bilstm_oulad",
    }
