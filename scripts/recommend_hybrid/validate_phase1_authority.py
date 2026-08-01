"""Read-only checkpoint authority validator for recommend_hybrid Phase 1."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
AUTHORITY_PATH = ROOT / "configs/recommend_hybrid/model_authority.yaml"
MANIFEST_PATH = (
    ROOT / "artifacts/recommend_hybrid/RECOMMEND_HYBRID_CHECKPOINT_MANIFEST.json"
)
TRAINING_AUTHORITY_PATH = (
    ROOT / "artifacts/canonical_v3/oulad_h1_training_authority.json"
)
LOG_PATH = ROOT / "reports/recommend_hybrid/logs/phase1_authority_validation.log"

EXPECTED_ARCHITECTURE_HASH = (
    "df5cd885b96e5cea4b840bfc5ca59c08c095f5887df8dd8dcef738edfe8bf70e"
)
EXPECTED_PARAMETER_COUNT = 160492
EXPECTED_SEEDS = {42, 1201, 2026, 3407, 7319}
EXPECTED_FOLDS = {0, 1, 2}
INTERVENTION_STAGES = {"EARLY_20", "EARLY_35", "MIDDLE_50", "LATE_75"}
ALL_STAGES = INTERVENTION_STAGES | {"FINAL_EVALUATION"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML object expected: {path}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def add_error(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def validate() -> dict[str, Any]:
    errors: list[str] = []
    details: list[str] = []
    for path in (AUTHORITY_PATH, MANIFEST_PATH, TRAINING_AUTHORITY_PATH):
        add_error(errors, path.is_file(), f"missing authority file: {path.relative_to(ROOT)}")
    if errors:
        return {"status": "RECOMMEND_HYBRID_PHASE1_AUTHORITY_FAIL", "errors": errors}

    authority = load_yaml(AUTHORITY_PATH)
    manifest = load_json(MANIFEST_PATH)
    training = load_json(TRAINING_AUTHORITY_PATH)

    expected_authority = {
        "authority_id": "RECOMMEND_HYBRID_MODEL_AUTHORITY",
        "status": "RELEASE_FROZEN",
        "architecture_family": "HYBRID_CNN_BILSTM_RECOMMENDER",
        "prediction_backbone": "FROZEN_HYBRID_CNN_BILSTM",
        "temporal_backbone": "CNN_BILSTM",
        "recommendation_component": "HYBRID_ACTION_RANKER",
        "plan_component": "HYBRID_LEARNING_PLAN_BUILDER",
        "training_status": "LOCKED_UNTIL_REAL_EXPERT_LABELS",
        "separate_prediction_model_allowed": False,
        "primary_recommendation_stage": "MIDDLE_50",
        "final_stage_usage": "EVALUATION_ONLY",
        "historical_v6_authority": False,
        "legacy_65_checkpoint_authority": False,
        "architecture_hash": EXPECTED_ARCHITECTURE_HASH,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "checkpoint_set_status": "COMPLETE_MULTI_STAGE_CHECKPOINT_SET",
    }
    for key, expected in expected_authority.items():
        add_error(errors, authority.get(key) == expected, f"authority mismatch: {key}")

    stage_policy = authority.get("stage_policy", {})
    add_error(errors, set(stage_policy) == ALL_STAGES, "stage policy names mismatch")
    add_error(
        errors,
        stage_policy.get("MIDDLE_50", {}).get("usage")
        == "PRIMARY_RECOMMENDATION_STAGE",
        "primary stage policy mismatch",
    )
    add_error(
        errors,
        stage_policy.get("FINAL_EVALUATION", {}).get("recommendation_allowed") is False,
        "FINAL_EVALUATION must prohibit recommendation generation",
    )
    exclusions = authority.get("legacy_exclusions", {})
    add_error(
        errors,
        exclusions.get("historical_v6_recommendation", {}).get("classification")
        == "REFERENCE_ONLY",
        "historical recommendation exclusion missing",
    )
    add_error(
        errors,
        exclusions.get("legacy_primary_validator", {}).get("classification")
        == "LEGACY_COMPATIBILITY_ONLY",
        "legacy 65-checkpoint exclusion missing",
    )

    add_error(
        errors,
        manifest.get("authority_id") == authority.get("authority_id"),
        "manifest authority mismatch",
    )
    add_error(
        errors,
        manifest.get("checkpoint_set_status") == "COMPLETE_MULTI_STAGE_CHECKPOINT_SET",
        "checkpoint set is not complete",
    )
    entries = manifest.get("checkpoints", [])
    add_error(errors, isinstance(entries, list), "checkpoint list missing")
    add_error(errors, len(entries) == 30, "expected 30 canonical checkpoint files")

    role_by_hash: dict[tuple[str, str], int] = {}
    for source_role, manifest_role in (
        ("shared_stage", "INTERVENTION_STAGE_SHARED"),
        ("endpoint_final", "EVALUATION_ONLY"),
    ):
        for row in training.get(source_role, []):
            role_by_hash[(manifest_role, str(row["config_hash"]))] = int(row["outer_fold"])

    expanded_keys: set[tuple[str, int, int]] = set()
    found_folds: set[int] = set()
    found_seeds: set[int] = set()
    invalid_files: list[str] = []
    for entry in entries if isinstance(entries, list) else []:
        provenance = entry.get("provenance", {})
        relative = provenance.get("source_checkpoint_path", "")
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"checkpoint missing: {relative}")
            invalid_files.append(relative)
            continue
        before = sha256(path)
        if before != entry.get("sha256"):
            errors.append(f"checkpoint checksum mismatch: {relative}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # pragma: no cover - exercised on corrupt files
            errors.append(f"checkpoint load failed: {relative}: {type(exc).__name__}")
            invalid_files.append(relative)
            continue
        after = sha256(path)
        if before != after:
            errors.append(f"checkpoint changed during read-only load: {relative}")
        if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
            errors.append(f"checkpoint payload invalid: {relative}")
            invalid_files.append(relative)
            continue
        actual_parameters = sum(tensor.numel() for tensor in payload["state_dict"].values())
        fold = int(entry.get("outer_fold", -1))
        seed = int(entry.get("seed", -1))
        role = str(entry.get("usage"))
        stages = set(entry.get("stages", []))
        add_error(errors, payload.get("architecture_hash") == EXPECTED_ARCHITECTURE_HASH, f"payload architecture mismatch: {relative}")
        add_error(errors, entry.get("architecture_hash") == EXPECTED_ARCHITECTURE_HASH, f"manifest architecture mismatch: {relative}")
        add_error(errors, payload.get("parameter_count") == EXPECTED_PARAMETER_COUNT == actual_parameters, f"payload parameter mismatch: {relative}")
        add_error(errors, entry.get("parameter_count") == EXPECTED_PARAMETER_COUNT, f"manifest parameter mismatch: {relative}")
        add_error(errors, int(payload.get("seed", -1)) == seed, f"seed metadata mismatch: {relative}")
        config_hash = str(payload.get("training_policy_hash"))
        add_error(errors, role_by_hash.get((role, config_hash)) == fold, f"fold/config metadata mismatch: {relative}")
        add_error(errors, entry.get("existence_status") == "FOUND", f"existence status mismatch: {relative}")
        add_error(errors, entry.get("load_validation_status") == "PASS", f"load status mismatch: {relative}")
        expected_stages = INTERVENTION_STAGES if role == "INTERVENTION_STAGE_SHARED" else {"FINAL_EVALUATION"}
        add_error(errors, stages == expected_stages, f"stage scope mismatch: {relative}")
        for stage in stages:
            key = (stage, fold, seed)
            add_error(errors, key not in expanded_keys, f"duplicate stage/fold/seed: {key}")
            expanded_keys.add(key)
        found_folds.add(fold)
        found_seeds.add(seed)
        details.append(f"PASS {relative} fold={fold} seed={seed} role={role} sha256={before}")

    add_error(errors, found_folds == EXPECTED_FOLDS, "fold coverage mismatch")
    add_error(errors, found_seeds == EXPECTED_SEEDS, "seed coverage mismatch")
    add_error(errors, len(expanded_keys) == 75, "expected 75 unique stage/fold/seed mappings")
    add_error(errors, manifest.get("missing_checkpoints") == [], "manifest reports missing checkpoints")
    add_error(errors, manifest.get("invalid_checkpoints") == [], "manifest reports invalid checkpoints")

    status = (
        "RECOMMEND_HYBRID_PHASE1_AUTHORITY_PASS"
        if not errors
        else "RECOMMEND_HYBRID_PHASE1_AUTHORITY_FAIL"
    )
    return {
        "status": status,
        "phase1_gate": "PHASE_1_PASS" if not errors else "PHASE_1_FAIL_CHECKPOINT_AUTHORITY_MISMATCH",
        "legacy_gate_alias": "RECOMMENDATION_V2_PHASE1_AUTHORITY_PASS" if not errors else "NOT_APPLICABLE",
        "checkpoint_set_status": manifest.get("checkpoint_set_status"),
        "checkpoint_files_expected": 30,
        "checkpoint_files_found": len(entries),
        "stage_fold_seed_mappings": len(expanded_keys),
        "missing": manifest.get("missing_checkpoints", []),
        "invalid": invalid_files,
        "checkpoint_bytes_changed": False,
        "errors": errors,
        "details": details,
    }


def main() -> int:
    result = validate()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {key: value for key, value in result.items() if key != "details"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "RECOMMEND_HYBRID_PHASE1_AUTHORITY_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
