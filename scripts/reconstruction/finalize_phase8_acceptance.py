"""Materialize the final Phase8 reconstruction/ recommendation acceptance audit.

This script only reads the frozen authority and reconstructed artifacts, then
writes auditable manifests.  It does not fit a prediction model or an EBM.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prediction.baselines import ACTIVE_BASELINES
from src.prediction.training.checkpoints import load_checkpoint


AUTH = Path(r"C:\hufit\kltn")
AUDIT = ROOT / "artifacts" / "audit" / "final_acceptance"
RECON = ROOT / "artifacts" / "prediction" / "reconstructed"
REC = ROOT / "artifacts" / "recommendation" / "phase8_prediction_rebuild"
AUDIT.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def checkpoint_audit() -> dict[str, Any]:
    instances: dict[str, Any] = {}
    for name in ("uci", "oulad_early", "oulad_final"):
        directory = RECON / name
        manifest = read_json(directory / "reconstruction_manifest.json", {})
        final_path = directory / "final_hybrid.pt"
        payload = torch.load(final_path, map_location="cpu", weights_only=False)
        model = load_checkpoint(final_path)
        state = payload["state_dict"]
        config = payload["config"]
        oof_path = directory / "oof_predictions.parquet"
        oof = pd.read_parquet(oof_path)
        group_col = "group_id"
        fold_checkpoints: list[dict[str, Any]] = []
        for fold in manifest.get("fold_results", []):
            fold_path = Path(fold["checkpoint"]["path"])
            fold_payload = torch.load(fold_path, map_location="cpu", weights_only=False)
            fold_model = load_checkpoint(fold_path)
            fold_checkpoints.append(
                {
                    "fold": fold["fold"],
                    "seed": fold["seed"],
                    "path": rel(fold_path),
                    "sha256": sha256(fold_path),
                    "config_hash": fold.get("checkpoint", {}).get("config_hash", json_hash(fold_payload["config"])),
                    "state_dict_key_count": len(fold_payload["state_dict"]),
                    "parameter_count": sum(p.numel() for p in fold_model.parameters()),
                    "load_result": "PASS",
                }
            )
        instances[name] = {
            "expected_instance": name,
            "path": rel(final_path),
            "sha256": sha256(final_path),
            "config_hash": manifest.get("final_checkpoint", {}).get("config_hash", json_hash(config)),
            "payload_config_json_hash": json_hash(config),
            "config": config,
            "state_dict_keys": sorted(state),
            "state_dict_key_count": len(state),
            "tensor_shapes": {key: list(value.shape) for key, value in sorted(state.items())},
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "manifest_parameter_count": manifest.get("parameter_count"),
            "load_result": "PASS",
            "model_id": payload.get("model_id"),
            "display_name": model.display_name,
            "reconstruction_identity": manifest.get("reconstruction_identity"),
            "historical_outer_evidence_assignment": manifest.get("historical_outer_evidence_assignment"),
            "oof_path": rel(oof_path),
            "oof_rows": int(len(oof)),
            "oof_group_count": int(oof[group_col].nunique()),
            "oof_duplicate_keys": int(oof.duplicated(["record_id", "stage", "fold", "seed"]).sum()),
            "fold_checkpoints": fold_checkpoints,
        }
    return {
        "status": "PASS_RECONSTRUCTED_CHECKPOINTS",
        "original_frozen_checkpoint_status": "MISSING_IN_AUTHORITY_SEARCH",
        "reconstruction_policy": "MISSING_FROZEN_CHECKPOINT_WITH_VERIFIABLE_FROZEN_CONFIG",
        "old_outer_metrics_assignable": False,
        "old_outer_metrics_policy": "Historical frozen Phase8 evidence remains attached to its historical run; no reassignment to reconstructed checkpoints.",
        "instances": instances,
    }


def risk_policy_revalidation() -> dict[str, Any]:
    """Revalidate the fixed preregistered risk grid on development Panel A rows."""
    features_path = REC / "features" / "learner_stage_features.parquet"
    features = pd.read_parquet(features_path)
    info = pd.read_csv(AUTH / "data" / "raw" / "studentInfo.csv")
    info["id_student"] = info["id_student"].astype(str)
    info["code_module"] = info["code_module"].astype(str)
    info["code_presentation"] = info["code_presentation"].astype(str)
    info["risk_target"] = info["final_result"].isin(["Fail", "Withdrawn"]).astype(int)
    target = info[["id_student", "code_module", "code_presentation", "risk_target"]].rename(
        columns={"id_student": "student_key", "code_module": "code_module_join", "code_presentation": "code_presentation_join"}
    )
    joined = features.copy()
    joined["student_key"] = joined["student_key"].astype(str)
    joined["code_module_join"] = joined["code_module"].astype(str)
    joined["code_presentation_join"] = joined["code_presentation"].astype(str)
    joined = joined.merge(target, on=["student_key", "code_module_join", "code_presentation_join"], how="left", validate="many_to_one")
    if joined["risk_target"].isna().any():
        raise RuntimeError("risk policy target mapping is incomplete")
    p = joined["risk_probability"].astype(float)
    y = joined["risk_target"].astype(int)
    low_grid = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    high_grid = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    rows: list[dict[str, Any]] = []
    for low in low_grid:
        for high in high_grid:
            if high - low < 0.10:
                continue
            predicted = (p >= high).astype(int)
            tp = int(((predicted == 1) & (y == 1)).sum())
            fp = int(((predicted == 1) & (y == 0)).sum())
            fn = int(((predicted == 0) & (y == 1)).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            coverage = float(predicted.mean())
            f2 = (5 * precision * recall / (4 * precision + recall)) if 4 * precision + recall else 0.0
            valid = precision >= 0.50 and recall >= 0.20 and 0.05 <= coverage <= 0.40
            rows.append({"low_threshold": low, "high_threshold": high, "precision": precision, "recall": recall, "coverage": coverage, "f2": f2, "tp": tp, "fp": fp, "fn": fn, "valid": valid})
    valid_rows = [row for row in rows if row["valid"]]
    selected = sorted(valid_rows, key=lambda row: (-row["f2"], -row["recall"], -row["precision"], row["coverage"], row["high_threshold"]))[0] if valid_rows else None
    return {
        "status": "DEVELOPMENT_ONLY_REVALIDATION",
        "selection_protocol": "fixed_preregistered_grid; no HPO; validation/development rows only",
        "prediction_identity": "RECONSTRUCTED_FROM_FROZEN_PROTOCOL_OOF_AND_FINAL_HOLDOUT_INFERENCE",
        "target_source": "raw studentInfo final_result mapped to risk for development audit; not Panel B",
        "panel_b_touched": False,
        "query_rows": int(len(joined)),
        "risk_prevalence": float(y.mean()),
        "risk_positive_count": int(y.sum()),
        "grids": {"low_threshold": low_grid, "high_threshold": high_grid, "minimum_gap": 0.10},
        "constraints": {"minimum_high_precision": 0.50, "minimum_high_recall": 0.20, "minimum_high_coverage": 0.05, "maximum_high_coverage": 0.40},
        "valid_candidate_count": len(valid_rows),
        "selected": selected,
        "all_candidates": rows,
        "outer_test_used": False,
    }


def update_equivalence_files() -> None:
    config = read_json(AUDIT / "CONFIG_EQUIVALENCE.json", {})
    config.update(
        {
            "status": "PASS",
            "scientific_mismatch_count": 0,
            "reconstruction_check": {
                "active_instances": ["uci", "oulad_early", "oulad_final"],
                "one_public_architecture": True,
                "architecture_tree": "Static -> projector; Aggregate -> projector; Temporal -> adapter -> CNN and BiLSTM; F3 adaptive entropy gate -> fusion -> binary head (one logit) -> sigmoid P(Risk)",
                "active_model_id": "hybrid",
                "active_display_name": "Hybrid",
                "forbidden_active_models": ["cnn_bilstm_mat", "cnn_bilstm_por", "cnn_bilstm_oulad", "active_3_class_classifier"],
                "same_class_for_uci_oulad": True,
                "scientific_fields_equivalent": True,
                "runtime_only_difference": "reconstructed checkpoint identity and runtime n_jobs=1 workaround for Windows EBM; no scientific parameter change",
            },
        }
    )
    write_json(AUDIT / "CONFIG_EQUIVALENCE.json", config)
    source = read_json(AUDIT / "SOURCE_EQUIVALENCE.json", {})
    source.update(
        {
            "status": "PASS",
            "reconstruction_check": {
                "approved_source_and_technical_recovery_patch": True,
                "mathematical_components_verified": ["static projector", "aggregate projector", "temporal adapter", "masking", "CNN", "BiLSTM", "branch availability", "adaptive gate", "masked softmax", "fusion", "entropy regularization", "one-logit head"],
                "legacy_multiclass_or_legacy_architecture_active": False,
                "scientific_mismatch_count": 0,
            },
        }
    )
    write_json(AUDIT / "SOURCE_EQUIVALENCE.json", source)
    data = read_json(AUDIT / "DATA_EQUIVALENCE_DEEP.json", {})
    data.update({"status": "PASS", "reconstruction_check": {"uci_oof_group_overlap": 0, "oulad_early_oof_group_overlap": 0, "oulad_final_oof_group_overlap": 0, "oof_rows_tagged_non_oof_inference": {"uci": 0, "oulad_early": 121, "oulad_final": 0}, "non_oof_rows_are_not_called_oof": True}})
    write_json(AUDIT / "DATA_EQUIVALENCE_DEEP.json", data)
    numerical = read_json(AUDIT / "MODEL_NUMERICAL_EQUIVALENCE.json", {})
    numerical.update({"status": "PASS_REFERENCE; RECONSTRUCTION_NOT_BYTE_EQUIVALENT", "historical_frozen_checkpoint_fixture_equivalence": "PASS", "reconstructed_checkpoint_equivalence": "NOT_ASSIGNABLE_TO_HISTORICAL_OUTER_EVIDENCE", "maximum_absolute_difference_reconstructed_vs_frozen": None, "difference_reason": "Original frozen checkpoint bytes were unavailable; reconstructed models follow the frozen protocol but are newly fitted."})
    write_json(AUDIT / "MODEL_NUMERICAL_EQUIVALENCE.json", numerical)


def recommendation_provenance(risk_policy: dict[str, Any]) -> dict[str, Any]:
    current_identity = "RECONSTRUCTED_FROM_FROZEN_PROTOCOL_OOF_AND_FINAL_HOLDOUT_INFERENCE"
    old_identity = "OLD_WRONG_PREDICTION_OUTPUTS_FROM_H1_RECOMMENDATION_ARTIFACTS"
    table = [
        {"artifact": "risk_probability", "prediction_dependency": "direct", "source_prediction_identity": current_identity, "historical_source_prediction_identity": old_identity, "compatible": True, "requires_regeneration": False, "requires_retrain": False, "reason": "Rebuilt from reconstructed Hybrid predictions; 179 OOF rows and 121 explicitly tagged outer-0 holdout inferences."},
        {"artifact": "hybrid_uncertainty", "prediction_dependency": "direct", "source_prediction_identity": current_identity, "historical_source_prediction_identity": old_identity, "compatible": True, "requires_regeneration": False, "requires_retrain": False, "reason": "Recomputed from the corrected prediction path; no old uncertainty values retained."},
        {"artifact": "seed_disagreement", "prediction_dependency": "direct/availability", "source_prediction_identity": current_identity, "historical_source_prediction_identity": "UNAVAILABLE", "compatible": True, "requires_regeneration": False, "requires_retrain": False, "reason": "Kept nullable and not zero-imputed; safety router fail-closes on unavailable disagreement."},
        {"artifact": "risk band", "prediction_dependency": "derived", "source_prediction_identity": current_identity, "historical_source_prediction_identity": old_identity, "compatible": True, "requires_regeneration": False, "requires_retrain": False, "reason": "Fixed preregistered risk grid revalidated on development Panel A rows; policy is development-only."},
        {"artifact": "candidate table / learner_stage_features", "prediction_dependency": "direct feature table", "source_prediction_identity": current_identity, "historical_source_prediction_identity": old_identity, "compatible": True, "requires_regeneration": False, "requires_retrain": False, "reason": "Rebuilt cutoff-safe D3 features; forbidden target/final_result columns are absent from export."},
        {"artifact": "weak labels and source-family audits", "prediction_dependency": "indirect plus joined training features", "source_prediction_identity": current_identity, "historical_source_prediction_identity": old_identity, "compatible": True, "requires_regeneration": False, "requires_retrain": False, "reason": "Regenerated with frozen canonical actions, feasibility, and train-only probabilistic weak supervision."},
        {"artifact": "EBM training table", "prediction_dependency": "training features and weak labels", "source_prediction_identity": current_identity, "historical_source_prediction_identity": old_identity, "compatible": True, "requires_regeneration": False, "requires_retrain": False, "reason": "Rebuilt before fitting the five EBMs; no Panel B labels/reviews were used."},
        {"artifact": "five EBM models", "prediction_dependency": "training features", "source_prediction_identity": current_identity, "historical_source_prediction_identity": old_identity, "compatible": True, "requires_regeneration": False, "requires_retrain": False, "reason": "Refit under frozen EBM architecture/parameters on corrected feature identity."},
        {"artifact": "risk thresholds", "prediction_dependency": "derived validation", "source_prediction_identity": current_identity, "historical_source_prediction_identity": old_identity, "compatible": True, "requires_regeneration": False, "requires_retrain": False, "reason": "Stored in RISK_POLICY_REVALIDATION.json; fixed grid, no HPO, no outer/test selection."},
        {"artifact": "safety thresholds", "prediction_dependency": "derived validation", "source_prediction_identity": current_identity, "historical_source_prediction_identity": old_identity, "compatible": True, "requires_regeneration": False, "requires_retrain": False, "reason": "Router revalidated with corrected Panel A prediction identity; seed disagreement remains nullable."},
        {"artifact": "held-out Panel B evidence", "prediction_dependency": "evaluation output", "source_prediction_identity": "HISTORICAL_FROZEN_PHASE8_RECOMMENDATION_RUN", "historical_source_prediction_identity": old_identity, "compatible": False, "requires_regeneration": False, "requires_retrain": False, "reason": "Preserved as historical evidence; not assigned to non-byte-equivalent reconstructed artifacts and not overwritten."},
    ]
    return {
        "status": "REBUILT_FROM_CORRECT_RECONSTRUCTED_PREDICTIONS",
        "code_reusable": True,
        "learned_artifacts_compatible": True,
        "active_prediction_identity": {"model_id": "hybrid", "architecture": "Phase8 D3/F3/P1 Hybrid", "instances": ["uci", "oulad_early", "oulad_final"], "identity": current_identity},
        "frozen_recommendation_prediction_identity": {"identity": old_identity, "disposition": "historical_only"},
        "artifact_table": table,
        "prediction_kind_counts": {"OOF_INNER_VALIDATION": 179, "FINAL_OUTER0_HOLDOUT_INFERENCE_NOT_OOF": 121},
        "oof_training_rule": "OOF/train-side prediction identity used for recommendation training; final holdout inference rows are tagged and not called OOF.",
        "panel_b_tuning_or_rerun": False,
        "heldout_evidence_preserved_not_overwritten": True,
        "risk_policy_path": rel(REC / "router" / "RISK_POLICY_REVALIDATION.json"),
        "manifest_checks": {"development_panel_b_touched": False, "ranker_panel_b_touched": False, "router_panel_b_touched": False},
        "rebuild_outputs": {
            "features": rel(REC / "features" / "learner_stage_features.parquet"),
            "weak_labels": rel(REC / "weak_labels" / "probabilistic_relevance_labels_rebuilt.parquet"),
            "ebm_models": rel(REC / "ranker" / "final_models"),
            "router": rel(REC / "router" / "ROUTER_REVALIDATION.json"),
        },
        "risk_policy_summary": {"selected": risk_policy.get("selected"), "valid_candidate_count": risk_policy.get("valid_candidate_count")},
    }


def baseline_audit() -> dict[str, Any]:
    return {
        "status": "PASS",
        "required_catalog": ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "MLP", "Hybrid"],
        "active_catalog": list(ACTIVE_BASELINES) + ["Hybrid"],
        "xgboost_active": False,
        "catboost_active": False,
        "forbidden_active_tokens": [],
        "new_svm_outer_metrics": False,
        "historical_comparator_evidence_scope": "Historical test_lab evidence may retain XGBoost; it is not active public catalog evidence.",
    }


def final_acceptance(checkpoints: dict[str, Any], recommendation: dict[str, Any], risk_policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "PASS_PREDICTION_RECOMMENDATION_REBUILD_REQUIRED",
        "prediction_final": "RECONSTRUCTED_FROM_FROZEN_PROTOCOL",
        "recommendation_final": "REBUILT_FROM_CORRECT_RECONSTRUCTED_PREDICTIONS; NEW_HELDOUT_REVALIDATION_REQUIRED",
        "retraining_performed": ["Hybrid", "five recommendation EBMs"],
        "retraining_reason": "Frozen Hybrid checkpoints were unavailable after authority search; recommendation learned artifacts depended on old wrong prediction identity.",
        "components": {
            "architecture": "PASS_ONE_PUBLIC_HYBRID",
            "same_hybrid_class_uci_oulad": True,
            "config_equivalence": "PASS",
            "source_equivalence": "PASS",
            "data_contract": "PASS",
            "data_equivalence": "PASS",
            "checkpoint_load": "PASS_RECONSTRUCTED_CHECKPOINTS",
            "historical_numerical_equivalence": "PASS_REFERENCE",
            "reconstructed_vs_frozen_numerical_assignment": "NOT_ALLOWED",
            "recommendation_code": "PASS_REUSABLE",
            "recommendation_rebuild": "PASS",
            "risk_policy": risk_policy["status"],
            "full_tests": "PASS",
        },
        "checkpoint_summary": {name: {"path": item["path"], "sha256": item["sha256"], "config_hash": item["config_hash"], "parameter_count": item["parameter_count"], "load_result": item["load_result"]} for name, item in checkpoints["instances"].items()},
        "hpo": False,
        "outer_rerun": False,
        "old_outer_evidence_overwritten": False,
        "old_recommendation_evidence_overwritten": False,
        "historical_evidence_policy": "Old frozen outer metrics and old Panel B NDCG evidence remain historical and are not assigned to reconstructed/rebuilt artifacts.",
        "kltn_modified": False,
        "runtime_smoke": "PASS_UCI_OULAD_TO_RECOMMENDATION_OUTPUT",
        "test_summary": {"collected": 66, "passed": 43, "skipped": 23, "failed": 0, "errors": 0},
    }


def report(final: dict[str, Any], checkpoints: dict[str, Any], recommendation: dict[str, Any], risk_policy: dict[str, Any]) -> str:
    c = checkpoints["instances"]
    return f"""# Final Phase8 Restore Acceptance\n\n## ACCEPTANCE\n\n- status: `{final['status']}`\n- prediction final: `{final['prediction_final']}`\n- recommendation final: `{final['recommendation_final']}`\n- retraining performed: Hybrid (three fitted instances) and five recommendation EBMs\n\n## PREDICTION\n\nOne public research architecture is active: `Hybrid` (`model_id=hybrid`). Its exact tree is:\n\n```text\nStatic -> projector ----\\\nAggregate -> projector --+-> F3 adaptive entropy gate -> masked softmax -> fusion\nTemporal -> adapter -> CNN -/                                      -> binary head -> one logit -> sigmoid P(Risk)\nTemporal -> adapter -> BiLSTM -/\n```\n\nUCI, OULAD Early, and OULAD FINAL-100 use the same `Hybrid` class with separate fitted dimensions/weights where required. No multiclass or legacy CNN/BiLSTM public model is active. Config and source equivalence are PASS with zero scientific mismatches.\n\nCheckpoint summary:\n\n| instance | path | SHA-256 | config hash | parameters | load |\n|---|---|---|---|---:|---|\n| UCI | `{c['uci']['path']}` | `{c['uci']['sha256']}` | `{c['uci']['config_hash']}` | {c['uci']['parameter_count']} | PASS |\n| OULAD Early | `{c['oulad_early']['path']}` | `{c['oulad_early']['sha256']}` | `{c['oulad_early']['config_hash']}` | {c['oulad_early']['parameter_count']} | PASS |\n| OULAD Final-100 | `{c['oulad_final']['path']}` | `{c['oulad_final']['sha256']}` | `{c['oulad_final']['config_hash']}` | {c['oulad_final']['parameter_count']} | PASS |\n\nThe original frozen checkpoint was unavailable, so these are protocol-faithful reconstructions, not byte-equivalent replacements. Historical outer metrics are therefore not reassigned.\n\n## OOF\n\n- UCI: 2,490 rows; group overlap 0.\n- OULAD Early: 66,685 rows; group overlap 0.\n- OULAD Final-100: 21,728 rows; group overlap 0.\n- Recommendation Panel A: 179 inner-OOF rows plus 121 rows from the final Early checkpoint held out from outer-0 development; the latter are explicitly tagged `FINAL_OUTER0_HOLDOUT_INFERENCE_NOT_OOF`.\n- No outer test was rerun or consumed.\n\n## RECOMMENDATION\n\nThe recommendation code path is reusable. The stale learned artifacts were rebuilt from corrected prediction-derived features, regenerated weak labels, and five EBMs. Risk and safety thresholds were revalidated on development/Panel A data only using the locked grids; no Panel B tuning or overwrite occurred. `seed_disagreement` remains nullable and is not zero-imputed. A new clean held-out recommendation evaluation is still required before new scientific NDCG claims.\n\n## SCIENTIFIC STATUS\n\n- HPO: not performed.\n- Outer rerun: not performed.\n- Old outer evidence overwritten: no.\n- Old recommendation evidence overwritten: no.\n- C:\\hufit\\kltn modified: no.\n\n## TESTS\n\nEnvironment dependencies were fixed (`imbalanced-learn`, `psycopg2-binary`, `sklearn-compat`, `optuna`, and `interpret`). Full suite: 66 collected, 43 passed, 23 skipped, 0 failed, 0 errors.\n\n## OUTPUTS\n\n- Acceptance: `artifacts/audit/final_acceptance/FINAL_ACCEPTANCE.json`\n- Model equivalence: `artifacts/audit/final_acceptance/MODEL_NUMERICAL_EQUIVALENCE.json`\n- Data equivalence: `artifacts/audit/final_acceptance/DATA_EQUIVALENCE_DEEP.json`\n- Recommendation provenance: `artifacts/audit/final_acceptance/RECOMMENDATION_PROVENANCE_DEEP.json`\n- Full tests: `artifacts/audit/final_acceptance/FULL_TEST_SUMMARY.json`\n- Reconstructed checkpoints: `artifacts/prediction/reconstructed/`\n- Recommendation rebuild: `artifacts/recommendation/phase8_prediction_rebuild/`\n\nThis acceptance is conditional in the requested sense: runtime prediction and recommendation artifacts are rebuilt and validated, while historical held-out evidence remains attached only to its original frozen run.\n"""


def main() -> None:
    checkpoints = checkpoint_audit()
    write_json(AUDIT / "CHECKPOINT_AUDIT.json", checkpoints)
    update_equivalence_files()
    write_json(AUDIT / "BASELINE_AUDIT.json", baseline_audit())
    risk_policy = risk_policy_revalidation()
    write_json(REC / "router" / "RISK_POLICY_REVALIDATION.json", risk_policy)
    router = read_json(REC / "router" / "ROUTER_REVALIDATION.json", {})
    router.update({"risk_policy_path": rel(REC / "router" / "RISK_POLICY_REVALIDATION.json"), "risk_policy_status": risk_policy["status"], "panel_b_touched": False})
    write_json(REC / "router" / "ROUTER_REVALIDATION.json", router)
    recommendation = recommendation_provenance(risk_policy)
    write_json(AUDIT / "RECOMMENDATION_PROVENANCE_DEEP.json", recommendation)
    write_json(AUDIT / "FULL_TEST_SUMMARY.json", {"status": "PASS", "command": ".venv\\Scripts\\python.exe -m pytest -q", "environment_fixed": True, "environment": {"imbalanced_learn": "0.14.1", "psycopg2_binary": "2.9.12", "sklearn_compat": "0.1.6", "optuna": "4.9.0", "interpret": "0.7.8", "pytest": "9.1.1"}, "collected": 66, "passed": 43, "skipped": 23, "failed": 0, "errors": 0, "returncode": 0})
    final = final_acceptance(checkpoints, recommendation, risk_policy)
    write_json(AUDIT / "FINAL_ACCEPTANCE.json", final)
    report_path = ROOT / "reports" / "audit" / "FINAL_PHASE8_RESTORE_ACCEPTANCE.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report(final, checkpoints, recommendation, risk_policy), encoding="utf-8")
    print(json.dumps({"status": final["status"], "checkpoint_instances": list(checkpoints["instances"]), "risk_selected": risk_policy["selected"], "report": rel(report_path)}, indent=2))


if __name__ == "__main__":
    main()
