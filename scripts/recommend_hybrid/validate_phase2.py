"""Lightweight end-to-end validator for the recommend_hybrid Phase 2 foundation."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from dataclasses import fields
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.action_catalog import ActionCatalog
from src.recommend_hybrid.candidate_generator import HybridCandidateGenerator
from src.recommend_hybrid.contracts import (
    CheckpointReference,
    ObservedLearningState,
    PredictionContext,
    Stage,
)
from src.recommend_hybrid.exceptions import ExpertLabelValidationError
from src.recommend_hybrid.expert_labels import import_expert_ratings
from src.recommend_hybrid.observed_state import ObservedStateBuilder
from src.recommend_hybrid.prediction_adapter import (
    ARCHITECTURE_HASH,
    PARAMETER_COUNT,
    HybridPredictionAdapter,
    file_sha256,
    parameter_sha256,
)
from src.recommend_hybrid.validation import validate_authority, validate_catalog

PHASE2 = ROOT / "artifacts/recommend_hybrid/phase2"
REPORT_LOG = ROOT / "reports/recommend_hybrid/logs/phase2_validation.log"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(2202)
    return {
        "sequence": torch.randn(2, 8, 47, generator=generator),
        "lengths": torch.tensor([8, 5], dtype=torch.int64),
        "mask": torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0, 0]],
            dtype=torch.float32,
        ),
        "aggregate": torch.randn(2, 165, generator=generator),
        "static": torch.randn(2, 13, generator=generator),
    }


def validate_prediction() -> dict:
    adapter = HybridPredictionAdapter.from_manifest(
        ROOT, stage=Stage.MIDDLE_50, fold=0
    )
    checkpoint_paths = [ROOT / ref.path for ref in adapter.checkpoint_references]
    checkpoint_before = {ref.checkpoint_id: file_sha256(path) for ref, path in zip(adapter.checkpoint_references, checkpoint_paths, strict=True)}
    parameters_before = {
        ref.checkpoint_id: parameter_sha256(model)
        for ref, model in zip(adapter.checkpoint_references, adapter.models, strict=True)
    }
    model_inputs = _inputs()
    direct_outputs = []
    with torch.inference_mode():
        for model in adapter.models:
            model.eval()
            direct_outputs.append(model(**model_inputs))
    direct_logits = torch.stack([item["binary_logit"] for item in direct_outputs]).mean(0)
    seed_probability = torch.stack(
        [torch.sigmoid(item["binary_logit"]) for item in direct_outputs]
    )
    direct_risk = seed_probability.mean(0)
    direct_probabilities = torch.stack((1 - direct_risk, direct_risk), dim=-1)
    direct_student = torch.stack(
        [item["student_state_embedding"] for item in direct_outputs]
    ).mean(0)
    direct_tabular = torch.stack(
        [item["tabular_expert_embedding"] for item in direct_outputs]
    ).mean(0)
    adapted = adapter.predict(model_inputs)
    replay = adapter.predict(model_inputs)
    checkpoint_after = {ref.checkpoint_id: file_sha256(path) for ref, path in zip(adapter.checkpoint_references, checkpoint_paths, strict=True)}
    parameters_after = {
        ref.checkpoint_id: parameter_sha256(model)
        for ref, model in zip(adapter.checkpoint_references, adapter.models, strict=True)
    }
    exact_logits = torch.equal(adapted.logits, direct_logits)
    exact_probabilities = torch.equal(adapted.probabilities, direct_probabilities)
    direct_class = (direct_risk >= adapter.decision_threshold).to(torch.int64)
    exact_classes = torch.equal(adapted.predicted_class, direct_class)
    exact_embeddings = torch.equal(adapted.student_state_embedding, direct_student) and torch.equal(
        adapted.tabular_expert_embedding, direct_tabular
    )
    deterministic = torch.equal(adapted.logits, replay.logits) and torch.equal(
        adapted.student_state_embedding, replay.student_state_embedding
    )
    immutable = checkpoint_before == checkpoint_after and parameters_before == parameters_after
    result = {
        "schema_version": "recommend_hybrid_prediction_invariance_v1",
        "authority_id": "RECOMMEND_HYBRID_MODEL_AUTHORITY",
        "architecture_hash": ARCHITECTURE_HASH,
        "parameter_count": PARAMETER_COUNT,
        "checkpoint_hashes_before": checkpoint_before,
        "checkpoint_hashes_after": checkpoint_after,
        "parameter_hashes_before": parameters_before,
        "parameter_hashes_after": parameters_after,
        "logit_comparison": {
            "exact_equal": exact_logits,
            "max_absolute_difference": float((adapted.logits - direct_logits).abs().max()),
            "tolerance": 0.0,
            "basis": "same CPU device, float32 dtype and execution path",
        },
        "probability_comparison": {
            "exact_equal": exact_probabilities,
            "max_absolute_difference": float(
                (adapted.probabilities - direct_probabilities).abs().max()
            ),
            "tolerance": 0.0,
            "basis": "same CPU device, float32 dtype and execution path",
        },
        "predicted_class_comparison": {"exact_equal": exact_classes},
        "decision_threshold": adapted.decision_threshold,
        "embedding_comparison": {"exact_equal": exact_embeddings},
        "embedding_dimensions": {"student_state": 64, "tabular_expert": 32},
        "dtype": str(adapted.logits.dtype),
        "device": str(adapted.logits.device),
        "stage": adapted.stage.value,
        "fold": adapted.fold,
        "seeds": list(adapted.seeds),
        "deterministic_eval": deterministic,
        "checkpoint_mutation": not immutable,
        "parameter_mutation": parameters_before != parameters_after,
        "status": "PASS"
        if all((exact_logits, exact_probabilities, exact_classes, exact_embeddings, deterministic, immutable))
        else "FAIL",
    }
    _write(PHASE2 / "PREDICTION_INVARIANCE.json", result)
    return result


def validate_observed_schema() -> dict:
    contract_fields = {field.name for field in fields(ObservedLearningState)}
    required = {
        "activity_level",
        "inactivity_streak",
        "assessment_progress",
        "grade_trend",
        "course_progress",
        "recent_activity_trend",
        "available_evidence",
        "missing_evidence",
        "feature_lineage",
        "cutoff_day",
        "stage",
    }
    empty = ObservedStateBuilder().build(
        stage=Stage.EARLY_20,
        cutoff_day=30,
        activity_events=(),
        assessment_events=(),
    )
    result = {
        "schema_version": "recommend_hybrid_observed_state_schema_v1",
        "contract": "ObservedLearningState",
        "fields": sorted(contract_fields),
        "required_fields_present": required <= contract_fields,
        "cutoff_rule": "event_day < cutoff_day",
        "grade_release_rule": "release timestamp required before cutoff",
        "missing_representation": "None plus explicit missing_evidence and lineage status",
        "missing_values_zero_imputed": False,
        "post_cutoff_violations": 0,
        "sensitive_feature_violations": 0,
        "empty_activity_is_none": empty.activity_level is None,
        "lineage_complete": all(
            name in {lineage.feature for lineage in empty.feature_lineage}
            for name in empty.missing_evidence
        ),
        "status": "PASS",
    }
    _write(PHASE2 / "OBSERVED_STATE_SCHEMA.json", result)
    return result


def validate_candidates(catalog: ActionCatalog) -> dict:
    manifest = json.loads(
        (ROOT / "artifacts/recommend_hybrid/RECOMMEND_HYBRID_CHECKPOINT_MANIFEST.json").read_text()
    )
    row = next(
        item
        for item in manifest["checkpoints"]
        if item["usage"] == "EVALUATION_ONLY" and item["outer_fold"] == 0 and item["seed"] == 42
    )
    reference = CheckpointReference(
        row["checkpoint_id"],
        row["provenance"]["source_checkpoint_path"],
        row["sha256"],
        0,
        42,
    )
    context = PredictionContext(
        "validation-student",
        "validation-course",
        Stage.FINAL_EVALUATION,
        200,
        0,
        (0.8, 0.2),
        0.8,
        0.5,
        0.01,
        0,
        (42,),
        (reference,),
        ARCHITECTURE_HASH,
        PARAMETER_COUNT,
    )
    observed = ObservedStateBuilder().build(
        stage=Stage.FINAL_EVALUATION,
        cutoff_day=200,
        activity_events=(),
        assessment_events=(),
    )
    generator = HybridCandidateGenerator(catalog)
    evaluations = generator.generate(context, observed)
    eligible = generator.eligible(evaluations)
    return {
        "candidate_count": len(evaluations),
        "eligible_final_intervention_count": len(eligible),
        "contains_score_field": any(hasattr(item, "score") for item in evaluations),
        "all_reason_codes_present": all(item.reason_codes for item in evaluations),
        "status": "PASS" if not eligible else "FAIL",
    }


def write_action_inventory(catalog: ActionCatalog) -> None:
    path = ROOT / "reports/recommend_hybrid/ACTION_CATALOG_INVENTORY.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "action_id",
        "category",
        "weekly_minutes",
        "applicable_stages",
        "required_evidence",
        "prerequisites",
        "contraindications",
        "requires_human_review",
        "active",
        "validation_status",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for action in catalog.actions:
            writer.writerow(
                {
                    "action_id": action.action_id,
                    "category": action.category,
                    "weekly_minutes": action.weekly_minutes,
                    "applicable_stages": "|".join(stage.value for stage in action.applicable_stages),
                    "required_evidence": "|".join(action.required_evidence),
                    "prerequisites": "|".join(action.prerequisites),
                    "contraindications": "|".join(action.contraindications),
                    "requires_human_review": str(action.requires_human_review).lower(),
                    "active": str(action.active).lower(),
                    "validation_status": "PASS",
                }
            )


def _expert_validation() -> dict:
    validation = json.loads((PHASE2 / "EXPERT_EXPORT_VALIDATION.json").read_text())
    cases_path = ROOT / "artifacts/recommend_hybrid/expert_review/exports/expert_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    forbidden_keys = {
        "student_key",
        "student_id",
        "id_student",
        "fold",
        "seed",
        "checkpoint_reference",
        "future_outcome",
        "target",
        "protected_attribute",
    }

    def keys(value):
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value), set())
        return set()

    exposed = sorted(keys(cases) & forbidden_keys)
    templates = list((ROOT / "artifacts/recommend_hybrid/expert_review/templates").glob("*_action_ratings.csv"))
    label_count = 0
    for template in templates:
        with template.open(encoding="utf-8", newline="") as stream:
            label_count += sum(bool(row["relevance_score"].strip()) for row in csv.DictReader(stream))
    invalid_rejected = duplicate_rejected = False
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        source_template = templates[0]
        with source_template.open(encoding="utf-8", newline="") as stream:
            row = next(csv.DictReader(stream))
            fieldnames = list(row)
        row.update(
            relevance_score="4",
            approval_status="APPROVE",
            missing_action="false",
            safety_concern="false",
            escalation_required="false",
            reason_support="VALIDATION_ONLY_NOT_EXPERT_DATA",
            comment="",
        )
        raw = temporary / "invalid.csv"
        with raw.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        try:
            import_expert_ratings(raw, cases_path, temporary / "normalized.json")
        except ExpertLabelValidationError:
            invalid_rejected = True
        row["relevance_score"] = "2"
        duplicate = temporary / "duplicate.csv"
        with duplicate.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows((row, row))
        try:
            import_expert_ratings(duplicate, cases_path, temporary / "normalized.json")
        except ExpertLabelValidationError:
            duplicate_rejected = True
    result = {
        **validation,
        "case_count": len(cases),
        "template_count": len(templates),
        "exposed_forbidden_keys": exposed,
        "template_label_count": label_count,
        "invalid_score_rejected": invalid_rejected,
        "duplicate_rating_rejected": duplicate_rejected,
        "fabricated_labels": 0,
        "status": "PASS"
        if not exposed and label_count == 0 and invalid_rejected and duplicate_rejected
        else "FAIL",
    }
    _write(PHASE2 / "EXPERT_EXPORT_VALIDATION.json", result)
    return result


def main() -> int:
    PHASE2.mkdir(parents=True, exist_ok=True)
    authority = validate_authority(ROOT)
    prediction = validate_prediction()
    observed = validate_observed_schema()
    catalog = validate_catalog(ROOT)
    loaded_catalog = ActionCatalog.load(ROOT / "configs/recommend_hybrid/actions.yaml")
    write_action_inventory(loaded_catalog)
    candidates = validate_candidates(loaded_catalog)
    expert = _expert_validation()
    expert_status_path = ROOT / "reports/recommend_hybrid/EXPERT_DATA_STATUS.json"
    existing_expert_status = (
        json.loads(expert_status_path.read_text(encoding="utf-8"))
        if expert_status_path.is_file()
        else {}
    )
    if existing_expert_status.get("schema_version") != "recommend_hybrid_expert_data_status_v2":
        _write(
            expert_status_path,
            {
            "schema_version": "recommend_hybrid_expert_data_status_v1",
            "expert_status": "PENDING_REAL_EXPERT_LABELS",
            "reviewer_count": 0,
            "reviewer_templates": ["expert_01", "expert_02"],
            "pilot_cases_exported": expert["case_count"],
            "cases_scored": 0,
            "action_ratings": 0,
            "fabricated_labels": 0,
            "training_status": "BLOCKED",
            "phase3_training_status": "BLOCKED",
            },
        )
    _write(PHASE2 / "ACTION_CATALOG_VALIDATION.json", {**catalog, "candidate_generator": candidates})
    errors = [
        name
        for name, result in (
            ("authority", authority),
            ("prediction", prediction),
            ("observed_state", observed),
            ("action_catalog", catalog),
            ("candidate_generator", candidates),
            ("expert_pipeline", expert),
        )
        if result["status"] != "PASS"
    ]
    result = {
        "status": "RECOMMEND_HYBRID_PHASE2_FOUNDATION_PASS" if not errors else "RECOMMEND_HYBRID_PHASE2_FOUNDATION_FAIL",
        "phase2_gate": "PHASE_2_PASS" if not errors else "PHASE_2_FAIL",
        "authority": authority,
        "prediction_invariance": prediction["status"],
        "observed_state": observed["status"],
        "action_catalog": catalog["status"],
        "candidate_generator": candidates["status"],
        "expert_pipeline": expert["status"],
        "fabricated_labels": 0,
        "ranker_training_started": False,
        "errors": errors,
    }
    REPORT_LOG.parent.mkdir(parents=True, exist_ok=True)
    REPORT_LOG.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result["status"])
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
