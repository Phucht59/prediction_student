from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.recommend_hybrid.common.plan_contracts import PlanStatus
from src.recommend_hybrid.common.service import HybridRecommendationService
from src.recommend_hybrid.persistence import InMemoryPlanRepository, JsonPlanRepository
from src.recommend_hybrid.common.policy_contracts import DatasetId

from .conftest import oulad_request, uci_request


def test_save_retrieve_round_trip(pipeline, uci_prediction):
    repository = InMemoryPlanRepository()
    service = HybridRecommendationService(pipeline, repository)
    plan = service.generate(uci_request(uci_prediction))
    assert service.retrieve(plan.plan_id).to_dict() == plan.to_dict()


def test_append_safe_versioning(tmp_path, pipeline, uci_prediction):
    repository = JsonPlanRepository(tmp_path)
    plan = pipeline.generate(uci_request(uci_prediction))
    repository.save(plan)
    repository.save(plan)
    changed = replace(plan, policy_version="different")
    with pytest.raises(ValueError, match="overwrite"):
        repository.save(changed)


def test_deterministic_replay(pipeline, oulad_prediction):
    repository = InMemoryPlanRepository()
    service = HybridRecommendationService(pipeline, repository)
    plan = service.generate(oulad_request(oulad_prediction))
    assert service.replay(plan.plan_id).to_dict() == plan.to_dict()


def test_dry_run_does_not_persist(pipeline, uci_prediction):
    repository = InMemoryPlanRepository()
    service = HybridRecommendationService(pipeline, repository)
    plan = service.generate(uci_request(uci_prediction), dry_run=True)
    assert service.retrieve(plan.plan_id) is None


def test_cli_dry_run_outputs_plan(tmp_path):
    fixture = {
        "student_key": "cli-student",
        "course_key": "cli-course",
        "created_at": "2026-08-01T00:00:00Z",
        "g1": 8,
        "g2": None,
        "absences": 12,
        "study_time": 1,
        "previous_failures": 1,
        "next_assessment_available": True,
        "prediction": {
            "predicted_class": 0,
            "class_probabilities": [0.7, 0.2, 0.1],
            "confidence": 0.7,
            "uncertainty": 0.45,
            "seed_disagreement": 0.03,
            "checkpoint_lineage": ["frozen_cli_fixture"],
            "architecture_authority": "RECOMMEND_HYBRID_MODEL_AUTHORITY",
        },
    }
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/recommend_hybrid/generate_plan.py"),
            "--dataset",
            "student_mat",
            "--input",
            str(path),
            "--dry-run",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        encoding="utf-8",
    )
    assert json.loads(result.stdout)["dataset_id"] == "student_mat"


def test_legacy_rows_untouched(tmp_path, pipeline, uci_prediction):
    legacy = tmp_path / "legacy.json"
    legacy.write_text('{"legacy":true}\n', encoding="utf-8")
    before = legacy.read_bytes()
    JsonPlanRepository(tmp_path / "new").save(pipeline.generate(uci_request(uci_prediction)))
    assert legacy.read_bytes() == before


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("mat", {PlanStatus.FULL, PlanStatus.PARTIAL}),
        ("por", {PlanStatus.FULL, PlanStatus.PARTIAL}),
        ("oulad", {PlanStatus.FULL, PlanStatus.PARTIAL}),
        ("abstain", {PlanStatus.ABSTAIN}),
        ("evaluation", {PlanStatus.EVALUATION_ONLY}),
    ],
)
def test_end_to_end_fixtures(kind, expected, pipeline, uci_prediction, oulad_prediction):
    if kind == "mat":
        request = uci_request(uci_prediction, DatasetId.STUDENT_MAT)
    elif kind == "por":
        request = uci_request(uci_prediction, DatasetId.STUDENT_POR)
    elif kind == "oulad":
        request = oulad_request(oulad_prediction)
    elif kind == "abstain":
        request = oulad_request(None, cutoff=19, max_observation_cutoff=None)
    else:
        request = oulad_request(oulad_prediction, cutoff=100, max_observation_cutoff=None)
    plan = pipeline.generate(request)
    assert plan.automation_status in expected
    assert plan.dataset_id in {"student_mat", "student_por", "oulad"}
