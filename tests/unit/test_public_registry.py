from src.models.registry import official_registry


def test_public_registry_contains_three_models_and_recommendation() -> None:
    registry = official_registry()
    assert set(registry) == {"student_mat", "student_por", "oulad", "recommendation"}
    assert registry["student_mat"]["model_id"] == "cnn_bilstm_mat"
    assert registry["student_por"]["model_id"] == "cnn_bilstm_por"
    assert registry["oulad"]["model_id"] == "cnn_bilstm_oulad"
