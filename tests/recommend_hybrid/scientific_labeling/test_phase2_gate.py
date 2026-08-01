from pathlib import Path

import pandas as pd

from src.recommend_hybrid.weak_supervision.labels import LF_ABSTAIN
from src.recommend_hybrid.weak_supervision.split import split_for_student

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "artifacts/recommend_hybrid/scientific_labeling"


def test_split_is_student_grouped_and_candidates_are_unique() -> None:
    frame = pd.read_parquet(ARTIFACT / "candidates.parquet")
    assert not frame.duplicated(["query_id", "action_id"]).any()
    assert not frame.groupby("student_key").split.nunique().gt(1).any()
    assert split_for_student("stable-student") == split_for_student("stable-student")


def test_silver_probabilities_and_safety_policy() -> None:
    frame = pd.read_parquet(ARTIFACT / "silver_labels.parquet")
    assert ((frame[["silver_prob_0", "silver_prob_1", "silver_prob_2"]].sum(axis=1) - 1).abs() < 1e-6).all()
    assert LF_ABSTAIN not in {0, 1, 2}
    retained = frame[frame.silver_status == "RETAINED"]
    assert retained["silver_label"].isin([0, 1, 2]).all()
    assert (retained.loc[retained.human_review_required, "silver_label"] != 2).all()
    gaps = retained.action_status == "INSUFFICIENT_EVIDENCE"
    assert (retained.loc[gaps, "silver_label"] != 2).all()
