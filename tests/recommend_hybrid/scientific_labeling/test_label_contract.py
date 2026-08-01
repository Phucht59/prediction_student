from dataclasses import fields

from src.recommend_hybrid.weak_supervision.contracts import CandidateActionExample
from src.recommend_hybrid.weak_supervision.labels import (
    LF_ABSTAIN,
    RELEVANCE_VALUES,
    TARGET_VALUES,
)
from src.recommend_hybrid.weak_supervision.validation import PROHIBITED_CANDIDATE_FIELDS


def test_labels_and_relevance_are_locked() -> None:
    assert TARGET_VALUES == {0, 1, 2}
    assert RELEVANCE_VALUES == {0, 1, 2}
    assert LF_ABSTAIN == -1
    assert LF_ABSTAIN not in TARGET_VALUES


def test_candidate_schema_has_no_sensitive_attribute() -> None:
    names = {field.name for field in fields(CandidateActionExample)}
    assert names.isdisjoint(PROHIBITED_CANDIDATE_FIELDS)
