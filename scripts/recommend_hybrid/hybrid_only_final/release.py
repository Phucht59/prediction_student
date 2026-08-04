"""Fail-closed release gate for the final hybrid-only recommender."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/recommend_hybrid/hybrid_only_final"
PROTOCOL_PATH = ROOT / "configs/recommend_hybrid/hybrid_only_final_protocol.yaml"
RUNTIME_CONFIG_PATH = ROOT / "configs/recommend_hybrid/hybrid_only_selected.yaml"
RELEASE_PATH = OUT / "HYBRID_ONLY_RELEASE.json"


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"required release artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _gate(value: bool, actual, required, description: str) -> dict:
    return {
        "status": "PASS" if value else "FAIL",
        "actual": actual,
        "required": required,
        "description": description,
    }


def main() -> None:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    gates_config = protocol["release_gates"]
    results = _read_json(OUT / "evaluation/OOF_RESULTS.json")
    bootstrap = _read_json(OUT / "evaluation/BOOTSTRAP.json")
    verification = _read_json(OUT / "evaluation/VERIFICATION.json")
    selected = _read_json(OUT / "HYBRID_ONLY_SELECTED_CONFIG.json")
    fold_frame = pd.read_csv(OUT / "evaluation/FOLD_METRICS.csv")
    oof = pd.read_parquet(OUT / "evaluation/OOF_PREDICTIONS.parquet")

    if results.get("status") != "COMPLETE":
        raise RuntimeError("OOF evaluation is not complete")
    if bootstrap.get("status") != "COMPLETE":
        raise RuntimeError("bootstrap is not complete")
    if verification.get("status") != "PASS":
        raise RuntimeError("hybrid-only verification did not pass")

    overall = results["overall"]
    issued = oof[oof["issued"] == 1].copy()
    supported_stage = (
        issued.groupby("stage", observed=True)["silver_positive"].agg(["mean", "count"])
        if len(issued)
        else pd.DataFrame(columns=["mean", "count"])
    )
    supported_stage = supported_stage[supported_stage["count"] >= 50]
    worst_stage = float(supported_stage["mean"].min()) if len(supported_stage) else 0.0
    minimum_fold = float(fold_frame["precision_at_1"].min()) if len(fold_frame) else 0.0

    checks = {
        "top1_precision": _gate(
            float(overall["precision_at_1"])
            >= float(gates_config["top1_precision_minimum"]),
            float(overall["precision_at_1"]),
            float(gates_config["top1_precision_minimum"]),
            "Top-1 silver-label precision on issued recommendations",
        ),
        "top1_precision_bootstrap_lower": _gate(
            float(bootstrap["precision_at_1"]["lower_95"])
            >= float(gates_config["top1_precision_bootstrap_lower_minimum"]),
            float(bootstrap["precision_at_1"]["lower_95"]),
            float(gates_config["top1_precision_bootstrap_lower_minimum"]),
            "Learner-cluster bootstrap lower confidence bound",
        ),
        "actionable_coverage": _gate(
            float(overall["actionable_coverage"])
            >= float(gates_config["actionable_coverage_minimum"]),
            float(overall["actionable_coverage"]),
            float(gates_config["actionable_coverage_minimum"]),
            "Coverage among groups with at least one positive future action",
        ),
        "outer_fold_precision": _gate(
            minimum_fold >= float(gates_config["each_outer_fold_precision_minimum"]),
            minimum_fold,
            float(gates_config["each_outer_fold_precision_minimum"]),
            "Minimum held-out outer-fold precision",
        ),
        "supported_stage_precision": _gate(
            worst_stage >= float(gates_config["supported_stage_precision_minimum"]),
            worst_stage,
            float(gates_config["supported_stage_precision_minimum"]),
            "Minimum stage precision where at least 50 recommendations were issued",
        ),
        "action_diversity": _gate(
            int(overall["action_diversity"])
            >= int(gates_config["action_diversity_minimum"]),
            int(overall["action_diversity"]),
            int(gates_config["action_diversity_minimum"]),
            "Distinct scientific action families issued",
        ),
        "top_action_concentration": _gate(
            float(overall["top_action_concentration"])
            <= float(gates_config["top_action_concentration_maximum"]),
            float(overall["top_action_concentration"]),
            float(gates_config["top_action_concentration_maximum"]),
            "Maximum share of the most common top action",
        ),
        "protected_feature_use": _gate(
            verification["gates"]["protected_features_in_scoring"],
            0,
            int(gates_config["protected_feature_use"]),
            "Protected features are absent from scoring",
        ),
        "temporal_leakage": _gate(
            verification["gates"]["future_features_in_scoring"],
            0,
            int(gates_config["temporal_leakage"]),
            "Future trajectory columns are absent from scoring",
        ),
        "constraint_violations": _gate(
            verification["gates"]["constraint_violations"],
            int(verification["constraint_violation_count"]),
            int(gates_config["constraint_violations"]),
            "Availability and prerequisite constraints",
        ),
        "deterministic_replay": _gate(
            verification["gates"]["deterministic_replay"],
            bool(verification["gates"]["deterministic_replay"]),
            bool(gates_config["deterministic_replay_required"]),
            "Exact deterministic OOF replay",
        ),
        "hybrid_only_architecture": _gate(
            verification["gates"]["forbidden_runtime_models"]
            and verification["gates"]["protocol_forbids_learned_ranker"],
            "frozen_residual_cnn_bilstm_only",
            "frozen_residual_cnn_bilstm_only",
            "No learned recommendation ranker is used",
        ),
    }

    passed = all(item["status"] == "PASS" for item in checks.values())
    status = (
        "HYBRID_ONLY_OFFLINE_SILVER_VALIDATED"
        if passed
        else "HYBRID_ONLY_SILVER_EVIDENCE_BELOW_GATE"
    )
    completion = "RECOMMENDATION_MODULE_COMPLETE" if passed else "RECOMMENDATION_MODULE_NOT_COMPLETE"

    release = {
        "schema_version": "hybrid_only_release_v1",
        "status": status,
        "thesis_scope_completion": completion,
        "runtime_authorized": passed,
        "merge_allowed": False,
        "claim_boundary": protocol["claim_boundary"],
        "primary_metric_interpretation": (
            "agreement with direct future OULAD silver labels on issued recommendations; "
            "not causal effectiveness or guaranteed grade improvement"
        ),
        "gates": checks,
        "overall": overall,
        "bootstrap": bootstrap,
        "supported_stage_precision": supported_stage.reset_index().to_dict(orient="records"),
        "selected_config": selected,
        "artifacts": {
            "protocol_sha256": _sha256(PROTOCOL_PATH),
            "candidate_dataset_sha256": _sha256(OUT / "dataset/candidate_rows.parquet"),
            "oof_predictions_sha256": _sha256(OUT / "evaluation/OOF_PREDICTIONS.parquet"),
            "verification_sha256": _sha256(OUT / "evaluation/VERIFICATION.json"),
        },
    }
    _write_json(RELEASE_PATH, release)

    if passed:
        runtime_payload = {
            "schema_version": "hybrid_only_runtime_config_v1",
            "status": "RELEASED",
            "release_status": status,
            "learned_model": "frozen_residual_cnn_bilstm",
            "additional_learned_ranker": False,
            "claim_boundary": protocol["claim_boundary"],
            "config": selected["config"],
            "normalization_scales": selected["normalization_scales"],
            "release_artifact": str(RELEASE_PATH.relative_to(ROOT)).replace("\\", "/"),
            "release_sha256": _sha256(RELEASE_PATH),
        }
        RUNTIME_CONFIG_PATH.write_text(
            yaml.safe_dump(runtime_payload, sort_keys=False), encoding="utf-8"
        )
    elif RUNTIME_CONFIG_PATH.exists():
        raise RuntimeError(
            "runtime config exists although the current fail-closed release did not pass"
        )

    print(json.dumps(release, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
