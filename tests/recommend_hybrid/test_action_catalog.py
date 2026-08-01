from src.recommend_hybrid.action_catalog import OBSERVED_EVIDENCE_FIELDS
from src.recommend_hybrid.contracts import Stage


def test_action_ids_unique(catalog):
    ids = [action.action_id for action in catalog.actions]
    assert len(ids) == len(set(ids))


def test_action_stage_valid(catalog):
    assert all(Stage.FINAL_EVALUATION not in action.applicable_stages for action in catalog.actions)


def test_action_prerequisites_valid(catalog):
    ids = {action.action_id for action in catalog.actions}
    assert all(set(action.prerequisites) <= ids for action in catalog.actions)
    assert all(set(action.required_evidence) <= OBSERVED_EVIDENCE_FIELDS for action in catalog.actions)


def test_action_workload_valid(catalog):
    assert all(0 < action.weekly_minutes <= 180 for action in catalog.actions)
