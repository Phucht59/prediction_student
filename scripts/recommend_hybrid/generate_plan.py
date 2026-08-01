"""Generate one recommend_hybrid learning plan from a safe JSON fixture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.common.policy_contracts import DatasetId, PolicyPredictionContext
from src.recommend_hybrid.common.service import HybridRecommendationService
from src.recommend_hybrid.persistence import InMemoryPlanRepository, JsonPlanRepository
from src.recommend_hybrid.pipeline import OULADPlanRequest, RecommendHybridPipeline, UCIPlanRequest


def _prediction(payload: dict | None, dataset_id: DatasetId) -> PolicyPredictionContext | None:
    if payload is None:
        return None
    return PolicyPredictionContext(
        dataset_id=dataset_id,
        predicted_class=int(payload["predicted_class"]),
        class_probabilities=tuple(float(value) for value in payload["class_probabilities"]),
        confidence=float(payload["confidence"]),
        uncertainty=float(payload["uncertainty"]),
        seed_disagreement=float(payload["seed_disagreement"]),
        checkpoint_lineage=tuple(payload["checkpoint_lineage"]),
        architecture_authority=payload["architecture_authority"],
        representation_lineage=tuple(payload.get("representation_lineage", ())),
        embedding_dimensions=tuple(payload.get("embedding_dimensions", (64, 32))),
    )


def _request(dataset_id: DatasetId, payload: dict):
    prediction = _prediction(payload.get("prediction"), dataset_id)
    common = {
        "student_key": payload["student_key"],
        "course_key": payload["course_key"],
        "active_contraindications": tuple(payload.get("active_contraindications", ())),
        "created_at": payload["created_at"],
    }
    if dataset_id in {DatasetId.STUDENT_MAT, DatasetId.STUDENT_POR}:
        if prediction is None:
            raise ValueError("UCI planning requires a frozen prediction context")
        return UCIPlanRequest(
            dataset_id=dataset_id,
            prediction=prediction,
            g1=payload.get("g1"),
            g2=payload.get("g2"),
            absences=payload.get("absences"),
            study_time=payload.get("study_time"),
            previous_failures=payload.get("previous_failures"),
            next_assessment_available=payload.get("next_assessment_available"),
            requested_cutoff=payload.get("requested_cutoff"),
            stage_evidence_known=payload.get("stage_evidence_known", True),
            extra_features=payload.get("extra_features"),
            **common,
        )
    return OULADPlanRequest(
        requested_cutoff=float(payload["requested_cutoff"]),
        prediction=prediction,
        max_observation_cutoff=payload.get("max_observation_cutoff"),
        activity_level=payload.get("activity_level"),
        recent_activity_trend=payload.get("recent_activity_trend"),
        inactivity_streak=payload.get("inactivity_streak"),
        assessment_progress=payload.get("assessment_progress"),
        assessments_due=payload.get("assessments_due"),
        grade_trend=payload.get("grade_trend"),
        grade_release_verified=payload.get("grade_release_verified", False),
        knowledge_gap=payload.get("knowledge_gap"),
        **common,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=[item.value for item in DatasetId])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--store-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and args.store_dir is None:
        parser.error("--store-dir is required unless --dry-run is used")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    repository = InMemoryPlanRepository() if args.dry_run else JsonPlanRepository(args.store_dir)
    service = HybridRecommendationService(RecommendHybridPipeline(ROOT), repository)
    plan = service.generate(_request(DatasetId(args.dataset), payload), dry_run=args.dry_run)
    output = json.dumps(plan.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        sys.stdout.buffer.write(output.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
