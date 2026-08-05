from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
FORBIDDEN = ("v2_1", "two_stage_v3", "two_stage_v4", "hybrid_only_final", "outcome_grounded", "xgboost", "lightgbm", "lambdamart")


def test_final_runtime_has_no_experiment_references():
    paths = list((ROOT / "src/recommend_hybrid/final").rglob("*.py"))
    paths += list((ROOT / "configs/recommend_hybrid/final").rglob("*"))
    text = "\n".join(path.read_text(errors="ignore").lower() for path in paths if path.is_file())
    assert not any(token in text for token in FORBIDDEN)
