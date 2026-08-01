from src.recommend_hybrid.validation import validate_authority, validate_catalog


def test_phase1_authority_remains_locked(root):
    assert validate_authority(root)["status"] == "PASS"


def test_phase2_catalog_validation(root):
    result = validate_catalog(root)
    assert result["status"] == "PASS"
    assert result["final_evaluation_interventions"] == 0
