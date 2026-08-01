"""Targeted validator for constrained Phase 4 learning-plan generation."""

from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.common.plan_contracts import PlanStatus
from src.recommend_hybrid.common.policy_contracts import DatasetId, PolicyPredictionContext
from src.recommend_hybrid.common.service import HybridRecommendationService
from src.recommend_hybrid.persistence import JsonPlanRepository
from src.recommend_hybrid.pipeline import OULADPlanRequest, RecommendHybridPipeline, UCIPlanRequest
from src.recommend_hybrid.prediction_adapter import ARCHITECTURE_HASH, PARAMETER_COUNT, file_sha256

CREATED_AT = "2026-08-01T00:00:00Z"


def _prediction(dataset_id: DatasetId) -> PolicyPredictionContext:
    probabilities = (0.3, 0.7) if dataset_id is DatasetId.OULAD else (0.7, 0.2, 0.1)
    return PolicyPredictionContext(
        dataset_id=dataset_id,
        predicted_class=1 if dataset_id is DatasetId.OULAD else 0,
        class_probabilities=probabilities,
        confidence=0.7,
        uncertainty=0.45,
        seed_disagreement=0.03,
        checkpoint_lineage=(f"frozen_{dataset_id.value}_cnn_bilstm_seed_ensemble",),
        architecture_authority="RECOMMEND_HYBRID_MODEL_AUTHORITY",
    )


def _uci(dataset_id: DatasetId) -> UCIPlanRequest:
    return UCIPlanRequest(
        dataset_id=dataset_id,
        student_key=f"fixture-{dataset_id.value}",
        course_key="fixture-course",
        prediction=_prediction(dataset_id),
        g1=8,
        g2=None,
        absences=12,
        study_time=1,
        previous_failures=1,
        next_assessment_available=True,
        created_at=CREATED_AT,
    )


def _oulad(cutoff: float, prediction: PolicyPredictionContext | None) -> OULADPlanRequest:
    return OULADPlanRequest(
        student_key=f"fixture-oulad-{cutoff}",
        course_key="fixture-course",
        requested_cutoff=cutoff,
        prediction=prediction,
        max_observation_cutoff=cutoff - 1 if cutoff not in (19, 100) else None,
        activity_level=4,
        recent_activity_trend=-6,
        inactivity_streak=14,
        assessment_progress=0.2,
        assessments_due=2,
        knowledge_gap="topic-A",
        created_at=CREATED_AT,
    )


def _checkpoint_validation() -> int:
    manifest_path = ROOT / "artifacts/recommend_hybrid/RECOMMEND_HYBRID_CHECKPOINT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["architecture_hash"] != ARCHITECTURE_HASH or int(manifest["parameter_count"]) != PARAMETER_COUNT:
        raise AssertionError("frozen prediction authority changed")
    for row in manifest["checkpoints"]:
        path = ROOT / row["provenance"]["source_checkpoint_path"]
        actual = file_sha256(path)
        if actual != row["sha256"]:
            raise AssertionError(f"checkpoint mutation: {path}")
    return len(manifest["checkpoints"])


def main() -> int:
    phase3 = json.loads((ROOT / "artifacts/recommend_hybrid/phase3/POLICY_MANIFEST.json").read_text(encoding="utf-8"))
    if phase3["status"] != "PHASE_3_PASS" or phase3["prediction_baseline_changed"]:
        raise AssertionError("Phase 3 authority is not locked PASS")
    checkpoint_count = _checkpoint_validation()
    planning = yaml.safe_load((ROOT / "configs/recommend_hybrid/planning.yaml").read_text(encoding="utf-8"))
    pipeline = RecommendHybridPipeline(ROOT)
    requests = (
        _uci(DatasetId.STUDENT_MAT),
        _uci(DatasetId.STUDENT_POR),
        _oulad(63, _prediction(DatasetId.OULAD)),
        _oulad(19, None),
        _oulad(100, _prediction(DatasetId.OULAD)),
    )
    plans = tuple(pipeline.generate(request) for request in requests)
    allowed = {key: set(value) for key, value in planning["dataset_actions"].items()}
    violations = Counter()
    for plan in plans:
        if len(plan.selected_actions) > int(planning["max_actions_per_plan"]):
            violations["action_cap"] += 1
        ids = [item.action_id for item in plan.selected_actions]
        if len(ids) != len(set(ids)):
            violations["duplicate"] += 1
        if any(item not in allowed[plan.dataset_id] for item in ids):
            violations["cross_dataset"] += 1
        workload = Counter()
        for action in plan.selected_actions:
            workload[action.scheduled_period] += action.weekly_minutes
            if not action.supporting_evidence or any(not item.source_lineage for item in action.supporting_evidence):
                violations["explanation_lineage"] += 1
            prerequisites = planning["action_metadata"][action.action_id]["prerequisites"]
            if any(item not in ids or ids.index(item) > ids.index(action.action_id) for item in prerequisites):
                violations["prerequisite"] += 1
            if any(item.feature_name == "G3" for item in action.supporting_evidence):
                violations["g3"] += 1
        if any(value > int(planning["max_minutes_per_period"]) for value in workload.values()):
            violations["workload"] += 1
        if plan.automation_status in {PlanStatus.ABSTAIN, PlanStatus.EVALUATION_ONLY} and ids:
            violations["zero_action_status"] += 1
    if plans[2].prediction_anchor > plans[2].requested_cutoff:
        violations["future_anchor"] += 1
    if plans[4].selected_actions:
        violations["final_intervention"] += 1
    if plans[3].selected_actions:
        violations["abstain_action"] += 1
    try:
        pipeline.generate(replace(requests[2], max_observation_cutoff=63))
        violations["post_cutoff"] += 1
    except ValueError:
        pass
    contraindicated = pipeline.generate(
        replace(requests[0], active_contraindications=("CONTACT_ALREADY_OPEN",))
    )
    if "INSTRUCTOR_CONTACT" in {item.action_id for item in contraindicated.selected_actions}:
        violations["contraindication"] += 1
    deterministic = pipeline.generate(requests[2]).to_dict() == pipeline.generate(requests[2]).to_dict()
    if not deterministic:
        violations["deterministic"] += 1
    with tempfile.TemporaryDirectory() as directory:
        sentinel = Path(directory) / "legacy.json"
        sentinel.write_text('{"legacy":true}\n', encoding="utf-8")
        before = sentinel.read_bytes()
        repository = JsonPlanRepository(Path(directory) / "plans")
        service = HybridRecommendationService(pipeline, repository)
        saved = service.generate(requests[0])
        replayed = service.replay(saved.plan_id)
        persistence_pass = replayed is not None and replayed.to_dict() == saved.to_dict()
        legacy_unchanged = sentinel.read_bytes() == before
    if not persistence_pass:
        violations["persistence"] += 1
    if not legacy_unchanged:
        violations["legacy_mutation"] += 1
    if violations:
        raise AssertionError(f"Phase 4 validation violations: {dict(violations)}")

    artifact_dir = ROOT / "artifacts/recommend_hybrid/phase4"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    validation = {
        "schema_version": "recommend_hybrid_plan_validation_v1",
        "status": "PHASE_4_PASS",
        "planning_version": planning["planning_version"],
        "architecture_hash": ARCHITECTURE_HASH,
        "parameter_count": PARAMETER_COUNT,
        "checkpoint_count": checkpoint_count,
        "checkpoint_set_sha256": phase3["checkpoint_set_sha256"],
        "prediction_baseline_changed": False,
        "checkpoint_bytes_changed": False,
        "max_actions_per_plan": planning["max_actions_per_plan"],
        "max_minutes_per_period": planning["max_minutes_per_period"],
        "violations": {
            "action_cap": 0,
            "workload": 0,
            "duplicate": 0,
            "prerequisite": 0,
            "contraindication": 0,
            "cross_dataset": 0,
            "post_cutoff": 0,
            "future_anchor": 0,
            "g3": 0,
            "final_intervention": 0,
            "abstain_action": 0,
        },
        "explanation_lineage_completeness": 1.0,
        "persistence_round_trip": "PASS",
        "legacy_data_unchanged": True,
        "deterministic_replay": "PASS",
    }
    smoke = {
        "schema_version": "recommend_hybrid_e2e_smoke_v1",
        "status": "PASS",
        "fixtures": [
            {
                "dataset_id": plan.dataset_id,
                "requested_cutoff": plan.requested_cutoff,
                "prediction_anchor": plan.prediction_anchor,
                "plan_status": plan.automation_status.value,
                "action_count": len(plan.selected_actions),
            }
            for plan in plans
        ],
        "student_data_included": False,
    }
    (artifact_dir / "PLAN_VALIDATION.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    (artifact_dir / "END_TO_END_SMOKE_RESULTS.json").write_text(json.dumps(smoke, indent=2) + "\n", encoding="utf-8")
    print("RECOMMEND_HYBRID_PHASE4_PLANNING_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
