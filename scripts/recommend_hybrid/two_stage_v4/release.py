"""Fail-closed scientific release for action-aware integrated V4."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/recommend_hybrid/two_stage_v4"
PROTOCOL_PATH = ROOT / "configs/recommend_hybrid/two_stage_v4_protocol.yaml"
RELEASE_PATH = OUT / "TWO_STAGE_V4_RELEASE.json"


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"required V4 artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _gate(actual: Any, required: Any, passed: bool, description: str) -> dict[str, Any]:
    return {
        "status": "PASS" if passed else "FAIL",
        "actual": actual,
        "required": required,
        "description": description,
    }


def _supported_stage_minimum(folds: list[dict[str, Any]]) -> float:
    values = []
    for fold in folds:
        for row in fold.get("per_stage", []):
            if int(row.get("issued_groups", 0)) >= 50:
                values.append(float(row["end_to_end_precision_at_1"]))
    return min(values) if values else 0.0


def _negative_control_status(path: Path) -> tuple[bool, dict[str, Any]]:
    required = [
        "group_target_shuffle_retrain",
        "action_label_shuffle_within_positive_group_retrain",
        "frozen_embedding_permutation_retrain",
        "action_identity_shuffle_retrain",
    ]
    if not path.exists():
        return False, {"status": "NOT_RUN", "required_controls": required}
    payload = _read(path)
    return bool(
        payload.get("status") == "COMPLETE"
        and payload.get("all_controls_pass") is True
    ), payload


def main() -> None:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    results_path = OUT / "final_oof/NESTED_OOF_RESULTS.json"
    bootstrap_path = OUT / "final_oof/BOOTSTRAP.json"
    verification_path = OUT / "final_oof/VERIFICATION.json"
    controls_path = OUT / "negative_controls/SUMMARY.json"
    runtime_path = OUT / "runtime/RUNTIME_PACKAGE.json"

    results = _read(results_path)
    bootstrap = _read(bootstrap_path)
    verification = _read(verification_path)
    if results.get("status") != "COMPLETE":
        raise RuntimeError("V4 nested OOF evaluation is incomplete")
    if bootstrap.get("status") != "COMPLETE":
        raise RuntimeError("V4 learner-cluster bootstrap is incomplete")
    if verification.get("status") != "PASS":
        raise RuntimeError("V4 verification did not pass")

    overall = results["overall"]
    folds = results["folds"]
    gates = protocol["release_gates"]
    fold_minimum = min(float(row["end_to_end_precision_at_1"]) for row in folds)
    stage_minimum = _supported_stage_minimum(folds)
    replay_pass = bool(
        verification["gates"]["exact_group_replay"]
        and verification["gates"]["numeric_replay"]
        and verification["gates"]["decision_replay"]
    )
    checks = {
        "end_to_end_precision_at_1": _gate(
            float(overall["end_to_end_precision_at_1"]),
            float(gates["end_to_end_precision_at_1_minimum"]),
            float(overall["end_to_end_precision_at_1"])
            >= float(gates["end_to_end_precision_at_1_minimum"]),
            "Held-out issued top-1 recommendation precision",
        ),
        "bootstrap_lower_precision": _gate(
            float(bootstrap["end_to_end_precision_at_1"]["lower_95"]),
            float(gates["end_to_end_precision_bootstrap_lower_minimum"]),
            float(bootstrap["end_to_end_precision_at_1"]["lower_95"])
            >= float(gates["end_to_end_precision_bootstrap_lower_minimum"]),
            "Learner-cluster lower 95% precision bound",
        ),
        "positive_group_coverage": _gate(
            float(overall["positive_group_coverage"]),
            float(gates["positive_group_coverage_minimum"]),
            float(overall["positive_group_coverage"])
            >= float(gates["positive_group_coverage_minimum"]),
            "Recall of groups containing at least one positive action",
        ),
        "conditional_precision_at_1": _gate(
            float(overall["stage_b_conditional_precision_at_1"]),
            float(gates["conditional_precision_at_1_minimum"]),
            float(overall["stage_b_conditional_precision_at_1"])
            >= float(gates["conditional_precision_at_1_minimum"]),
            "Conditional top-1 action precision",
        ),
        "each_outer_fold_precision": _gate(
            fold_minimum,
            float(gates["each_outer_fold_precision_minimum"]),
            fold_minimum >= float(gates["each_outer_fold_precision_minimum"]),
            "Minimum end-to-end precision across outer folds",
        ),
        "supported_stage_precision": _gate(
            stage_minimum,
            float(gates["supported_stage_precision_minimum"]),
            stage_minimum >= float(gates["supported_stage_precision_minimum"]),
            "Minimum supported stage precision",
        ),
        "action_diversity": _gate(
            int(overall["action_diversity"]),
            int(gates["action_diversity_minimum"]),
            int(overall["action_diversity"])
            >= int(gates["action_diversity_minimum"]),
            "Distinct action families issued",
        ),
        "top_action_concentration": _gate(
            float(overall["top_action_concentration"]),
            float(gates["top_action_concentration_maximum"]),
            float(overall["top_action_concentration"])
            <= float(gates["top_action_concentration_maximum"]),
            "Maximum share of the most frequent top action",
        ),
        "candidate_binary_all_groups": _gate(
            bool(verification["gates"]["candidate_binary_all_groups"]),
            True,
            bool(verification["gates"]["candidate_binary_all_groups"]),
            "Negative groups supervise all-zero candidate targets",
        ),
        "temporal_and_protected_feature_safety": _gate(
            bool(verification["gates"]["future_and_protected_features_absent"]),
            True,
            bool(verification["gates"]["future_and_protected_features_absent"]),
            "Future labels and protected attributes are absent",
        ),
        "external_ml_ranker_absent": _gate(
            bool(verification["gates"]["external_ml_ranker_absent"]),
            True,
            bool(verification["gates"]["external_ml_ranker_absent"]),
            "No external recommendation model",
        ),
        "deterministic_replay": _gate(
            replay_pass,
            True,
            replay_pass,
            "Exact checkpoint, numeric, and decision replay",
        ),
    }
    main_gates_pass = all(row["status"] == "PASS" for row in checks.values())
    controls_pass, controls = _negative_control_status(controls_path)
    runtime_ready = runtime_path.exists()

    if not main_gates_pass:
        status = "TWO_STAGE_V4_EVIDENCE_BELOW_GATE"
        completion = "RECOMMENDATION_MODULE_NOT_COMPLETE"
    elif not controls_pass:
        status = "TWO_STAGE_V4_MAIN_EVALUATION_PASS_CONTROLS_PENDING"
        completion = "RECOMMENDATION_MODULE_SCIENTIFIC_EXECUTION_NOT_COMPLETE"
    elif not runtime_ready:
        status = "TWO_STAGE_V4_OFFLINE_VALIDATED_RUNTIME_PACKAGE_PENDING"
        completion = "RECOMMENDATION_MODULE_RUNTIME_NOT_COMPLETE"
    else:
        runtime = _read(runtime_path)
        runtime_ready = bool(
            runtime.get("status") == "COMPLETE"
            and runtime.get("smoke_test") == "PASS"
            and runtime.get("frozen_backbone_trainable") is False
        )
        status = (
            "TWO_STAGE_V4_OFFLINE_SILVER_VALIDATED"
            if runtime_ready
            else "TWO_STAGE_V4_RUNTIME_PACKAGE_INVALID"
        )
        completion = (
            "RECOMMENDATION_MODULE_COMPLETE"
            if runtime_ready
            else "RECOMMENDATION_MODULE_RUNTIME_NOT_COMPLETE"
        )

    runtime_authorized = bool(main_gates_pass and controls_pass and runtime_ready)
    release = {
        "schema_version": "two_stage_v4_release_v1",
        "status": status,
        "thesis_scope_completion": completion,
        "runtime_authorized": runtime_authorized,
        "merge_allowed": False,
        "claim_boundary": protocol["claim_boundary"],
        "main_gates_pass": main_gates_pass,
        "negative_controls_pass": controls_pass,
        "runtime_package_ready": runtime_ready,
        "gates": checks,
        "overall": overall,
        "bootstrap": bootstrap,
        "negative_controls": controls,
        "artifacts": {
            "protocol_sha256": _sha256(PROTOCOL_PATH),
            "results_sha256": _sha256(results_path),
            "bootstrap_sha256": _sha256(bootstrap_path),
            "verification_sha256": _sha256(verification_path),
            "controls_sha256": _sha256(controls_path)
            if controls_path.exists()
            else None,
            "runtime_sha256": _sha256(runtime_path)
            if runtime_path.exists()
            else None,
        },
        "claim_rules": protocol["claim_rules"],
    }
    RELEASE_PATH.write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(release, indent=2, sort_keys=True))
    if not runtime_authorized:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
